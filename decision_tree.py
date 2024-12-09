import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
import joblib

# Load and preprocess audio files
def load_audio_files(directory):
    audio_data = []
    labels = []
    
    for label_dir in os.listdir(directory):
        label_path = os.path.join(directory, label_dir)
        if os.path.isdir(label_path):
            for file_name in os.listdir(label_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(label_path, file_name)
                    try:
                        y, sr = librosa.load(file_path, sr=None)
                        y = librosa.util.normalize(y)

                        if len(y) < 1024:
                            y = np.pad(y, (0, max(0, 1024 - len(y))), mode='constant')

                        n_fft = min(1024, len(y))
                        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                        mfccs_mean = np.mean(mfccs.T, axis=0)

                        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
                        chroma_mean = np.mean(chroma.T, axis=0)

                        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft)
                        spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)

                        try:
                            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
                            tonnetz_mean = np.mean(tonnetz.T, axis=0)
                        except ValueError:
                            tonnetz_mean = np.zeros(6)

                        features = np.concatenate((mfccs_mean, chroma_mean, spectral_contrast_mean, tonnetz_mean))
                        audio_data.append(features)
                        labels.append(label_dir)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

    return np.array(audio_data), np.array(labels)

# Load data
data_dir = 'dataset-hb'  # Change this to your actual dataset directory
X, y = load_audio_files(data_dir)

# Label encoding and normalization
le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_encoded)

# Optional: Augment data if necessary
unique_classes = np.unique(y_resampled)
X_augmented = []
y_augmented = []

for cls in unique_classes:
    X_cls = X_resampled[y_resampled == cls]
    y_cls = y_resampled[y_resampled == cls]
    
    if len(X_cls) < 100:  # Threshold for augmentation
        X_cls, y_cls = resample(X_cls, y_cls, replace=True, n_samples=100, random_state=42)
    
    X_augmented.append(X_cls)
    y_augmented.append(y_cls)

X_augmented = np.vstack(X_augmented)
y_augmented = np.hstack(y_augmented)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=42)

# Hyperparameter tuning for Decision Tree
dt = DecisionTreeClassifier()

# Define parameter grid for Decision Tree
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [9, 14, None],
    'min_samples_split': [2, 5, 10]
}

# Grid Search for Decision Tree
dt_cv = GridSearchCV(dt, param_grid_dt, cv=5)
dt_cv.fit(X_train, y_train)

# Print best parameters for Decision Tree
print("Decision Tree Best Parameters:", dt_cv.best_params_)

# Fit the best Decision Tree model
best_dt = dt_cv.best_estimator_
best_dt.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = best_dt.predict(X_test)

train_accuracy = best_dt.score(X_train, y_train)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"Training Accuracy: {train_accuracy}")
print(f"Test Set Accuracy: {test_accuracy}")

# Detailed classification report
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save the model and label encoder
joblib.dump(best_dt, 'heart_disease_decision_tree_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

# Function to extract features from a new audio file
def extract_features_from_audio(audio_file_path, n_mfcc=13):
    try:
        audio, sample_rate = librosa.load(audio_file_path, sr=None)

        if len(audio) < 1024:
            audio = np.pad(audio, (0, max(0, 1024 - len(audio))), mode='constant')

        n_fft = min(1024, len(audio))
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs.T, axis=0)

        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, n_fft=n_fft)
        chroma_mean = np.mean(chroma.T, axis=0)

        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate, n_fft=n_fft)
        spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)

        try:
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sample_rate)
            tonnetz_mean = np.mean(tonnetz.T, axis=0)
        except ValueError:
            tonnetz_mean = np.zeros(6)

        features = np.concatenate((mfccs_mean, chroma_mean, spectral_contrast_mean, tonnetz_mean))
        return features

    except Exception as e:
        print(f"Error processing {audio_file_path}: {e}")
        return None

# Load model and label encoder
best_dt = joblib.load('heart_disease_decision_tree_model.pkl')
le = joblib.load('label_encoder.pkl')

# Sample test prediction
audio_file_path = 'dataset-hb/Atraining_murmur/201101051114.wav'
features = extract_features_from_audio(audio_file_path)

if features is not None:
    features = features.reshape(1, -1)
    predicted_label = best_dt.predict(features)
    predicted_class = le.inverse_transform(predicted_label)
    print("Predicted Heart Condition:", predicted_class[0])
else:
    print("Feature extraction failed for the provided audio file.")