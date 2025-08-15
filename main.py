import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load the dataset
data = pd.read_csv('fer2013.csv')  # Update with the path to your dataset

# Preprocess the data
X = []
y = []

for index, row in data.iterrows():
    img = np.array(row['pixels'].split(), dtype='float32').reshape(48, 48) / 255.0

    X.append(img)
    y.append(row['emotion'])

X = np.array(X)
y = to_categorical(np.array(y))  # One-hot encode the labels

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for CNN input
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 classes for emotions
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=300, batch_size=64, validation_data=(X_test, y_test))

# Save the model
model.save('emotion_recognition_model.h5')


df = pd.read_csv('fer2013.csv')
emotions = {0: 'Angry', 1: 'Digust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
emotion_counts = df['emotion'].value_counts(sort=False).reset_index()
emotion_counts.columns = ['emotion', 'number']
emotion_counts['emotion'] = emotion_counts['emotion'].map(emotions)
emotion_counts
def pictures(row):
    pixels, emotion = row['pixels'], emotions[row['emotion']]
    img = np.array(pixels.split(), dtype=int)
    img = img.reshape(48,48)
    return img, emotion

plt.figure(0, figsize=(16,10))
for i in range(1,9):
    if len(df[df['emotion'] == i-1]) == 0:
        continue

    face = df[df['emotion'] == i-1].iloc[0]
    img, emotion = pictures(face)
    plt.subplot(2,4,i)
    plt.imshow(img, cmap='gray')
    plt.title(emotion)

plt.show()





# Load the trained model from .h5 file
model = load_model('/content/emotion_recognition_model.h5')  # Replace with your .h5 file path

# Define the emotions corresponding to the model's output
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    plt.imshow(img, cmap='gray')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(img, (48, 48))  # Resize to the expected input size
    img = img / 255.0  # Normalize the image
    img = img.reshape(1, 48, 48, 1)  # Add batch and channel dimensions
    return img

# Function to predict emotion from an image
def predict_emotion(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)  # Get the index of the highest probability
    emotion_name = emotions[predicted_index]  # Map the index to the emotion name
    return emotion_name

# Example usage
if __name__ == "__main__":
    image_path = 'Angry.jpg'  # Replace with your image path
    predicted_emotion = predict_emotion(image_path)
    print(f'Predicted Emotion: {predicted_emotion}')
