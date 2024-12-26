import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

data_dir = "C:\\Users\\Hamdi\\Desktop\\dataset"

# 1. Data Loading and Preprocessing
def load_data(data_dir, img_size=(54, 54)):
    categories = ["No_Fire", "Fire"]
    data = []
    labels = []

    # Loop through each category (Fire, No_Fire)
    for category in categories:
        category_path = (
            os.path.join(data_dir, category))
        category_label = categories.index(category)  # Label: 0 for Fire, 1 for No_Fire

        # Loop through each image in the category folder
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)

            # Read and resize the image to match input size
            img = cv2.imread(img_path)
            if img is None:
                continue  # Skip invalid images
            img_resized = cv2.resize(img, img_size)

            # Normalize the image (scaled to [0, 1] range)
            img_normalized = img_resized / 255.0
            data.append(img_normalized)
            labels.append(category_label)

    # Convert lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # One-hot encode the labels
    labels = to_categorical(labels, num_classes=2)

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test



# 2. Model Creation
def create_model(input_shape=(54, 54, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(2, activation='softmax')  # 2 classes: Fire and No_Fire
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 3. Train the Model
def train_and_save_model(data_dir, model_path="fire_detection_model.h5"):
    X_train, X_test, y_train, y_test = load_data(data_dir)
    model = create_model()

    # Train the model
    model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), batch_size=32)

    # Save the model
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


# 4. Real-Time Fire Detection using Webcam
def real_time_detection(model_path="C:\\Users\\Hamdi\\PycharmProjects\\AI\\Fıredetect-aı\\fire_detection_model.h5"):
    # Load the trained model
    model = load_model(model_path)

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Resize the frame to match the input size of the model
        frame_resized = cv2.resize(frame, (54, 54))

        # Normalize the frame
        frame_resized = frame_resized / 255.0

        # Expand dimensions to match the model input (batch size, height, width, channels)
        frame_input = np.expand_dims(frame_resized, axis=0)

        # Make a prediction
        prediction = model.predict(frame_input)

        # Get the class with the highest probability (Fire or No_Fire)
        predicted_class = np.argmax(prediction)

        # Show the prediction result on the frame
        label = 'No Fire' if predicted_class == 1 else 'Fire'

        # Display the frame with the prediction
        cv2.putText(frame, f"Prediction: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Fire Detection", frame)

        # Press 'q' to exit the camera feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()



# Main execution
if __name__ == "__main__":
    # Train the model and save it
    #train_and_save_model(data_dir)

    # Run real-time detection using the trained model
    real_time_detection(model_path="fire_detection_model.h5")

