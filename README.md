# Multimodal_Emotional_Analysis-2-

---

## 📥 1. **Dataset Download & Extraction**

Downloads and unzips the RAVDESS emotional speech dataset:

```python
!wget -O ravdess.zip "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
!unzip -q ravdess.zip -d /content/ravdess/
```

---

## 🎧 2. **Audio Preview**

Plays an audio sample using `IPython.display.Audio`:

```python
audio_path = "/content/ravdess/Actor_03/03-01-02-01-02-01-03.wav"
ipd.Audio(audio_path)
```

---

## 🧪 3. **Feature Extraction (MFCC)**

Extracts **Mel-frequency cepstral coefficients (MFCCs)** from `.wav` files:

```python
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs.mean(axis=1)
```

Creates a dataset of features + emotion labels:

```python
label = file.split("/")[-1].split("-")[2]  # Emotion label from filename
```

---

## 🤖 4. **SVM Model for Emotion Classification**

Builds a **Support Vector Machine (SVM)** model using scikit-learn:

```python
from sklearn.svm import SVC

model = SVC(kernel="rbf", C=1.0, gamma="scale")
model.fit(X_train, y_train)
```

Evaluates using accuracy and classification report:

```python
accuracy_score(y_test, y_pred)
classification_report(y_test, y_pred)
```

---

## 🧠 5. **Prediction Function**

Predicts emotion from a new audio file:

```python
def predict_emotion(file_path):
    ...
    return emotions[predicted_label]
```

---

## 🔬 6. **Deep Learning (CNN/LSTM Hybrid)**

Later in the notebook, it shifts to a deep learning approach using Keras:

* Feature extraction with `librosa`
* Emotion label parsing from file names
* Model using `Conv1D`, `LSTM`, `Dense` layers

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
```

---


