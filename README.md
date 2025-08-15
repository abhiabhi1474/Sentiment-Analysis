# Sentiment-Analysis
# Facial Emotion Recognition using CNN

This project implements a Convolutional Neural Network (CNN) to recognize human facial expressions from images using the FER2013 dataset. The model is capable of classifying emotions into seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## 📁 Dataset

The project uses the **FER2013** dataset, which contains 48x48 grayscale facial images labeled with the following emotion categories:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

Each image is represented as a flattened string of pixel values in CSV format.

## 🚀 Features

* Preprocessing of facial image data (normalization and reshaping)
* CNN-based architecture for emotion classification
* Visualization of example emotion images
* Model training with validation
* Model saving in `.h5` format
* Real-time image emotion prediction from static images

## 🧠 Model Architecture

```
Sequential CNN Model:
- Conv2D (32 filters, 3x3 kernel, ReLU)
- MaxPooling2D (2x2)
- Conv2D (64 filters, 3x3 kernel, ReLU)
- MaxPooling2D (2x2)
- Flatten
- Dense (128 units, ReLU)
- Dropout (0.5)
- Dense (7 units, Softmax)
```

## 📊 Training

* **Optimizer**: Adam
* **Loss Function**: Categorical Crossentropy
* **Batch Size**: 64
* **Epochs**: 300
* **Validation Split**: 20%

Example training log:

```
Epoch 85/300 - accuracy: 0.8611 - val_accuracy: 0.5421
```

## 🖼️ Emotion Prediction Example

```python
predicted_emotion = predict_emotion('img.jpg')
print(f'Predicted Emotion: {predicted_emotion}')
```

## 🧪 Testing

To test the model on a new image:

1. Ensure the image is a frontal face.
2. Convert it to grayscale and resize to 48x48.
3. Use the `predict_emotion()` function to classify the emotion.

## 🛠 Requirements

* Python 3.x
* TensorFlow
* NumPy
* Pandas
* OpenCV
* Matplotlib
* Scikit-learn

## 🧹 How to Run

1. Clone this repository.
2. Unzip and place the `fer2013.csv` in your project directory.
3. Run the file (`main.py`) or Python script.
4. Train the model or load the pre-trained `.h5` model.
5. Use the image prediction function for real-time emotion classification.

## 📌 Notes

* The `Disgust` class has significantly fewer samples compared to other emotions, which might affect classification performance.
* Validation accuracy stabilizes around \~54%, which is expected for this dataset and a basic CNN architecture.

## 📂 Files

* `main.py` – Main Jupyter notebook with all code
* `fer2013.csv` – Dataset file
* `emotion_recognition_model.h5` – Trained model
* `img.jpg` – Sample image for prediction

## 📸 Output

Sample output:

```
Predicted Emotion: Fear
```

## 📄 License

This project is for educational purposes. Feel free to use and modify.
