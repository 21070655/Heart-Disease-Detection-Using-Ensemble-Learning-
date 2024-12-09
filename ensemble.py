import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import joblib

# Step 1: Load audio files and extract features (MFCC, Chroma, Spectral Contrast, Tonnetz)
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
                            # Pad short signals
                            y = np.pad(y, (0, max(0, 1024 - len(y))), mode='constant')
                        
                        # Extract MFCCs
                        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                        mfccs_mean = np.mean(mfccs.T, axis=0)
                        
                        # Extract Chroma features
                        n_fft = min(2048, len(y))  # Adjust n_fft based on length of y
                        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
                        chroma_mean = np.mean(chroma.T, axis=0)
                        
                        # Extract Spectral Contrast features
                        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft)
                        spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
                        
                        # Extract Tonnetz features
                        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
                        tonnetz_mean = np.mean(tonnetz.T, axis=0)
                        
                        # Concatenate all features
                        features = np.concatenate((mfccs_mean, chroma_mean, spectral_contrast_mean, tonnetz_mean))
                        audio_data.append(features)
                        labels.append(label_dir)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
    
    return np.array(audio_data), np.array(labels)

# Step 2: Define paths and load data
data_dir = 'dataset-hb'  
X, y = load_audio_files(data_dir)

# Step 3: Label encoding and data normalization
le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_encoded)

# Optional: Augment data if necessary
# Split resampled data into different classes and perform resampling
unique_classes = np.unique(y_resampled)
X_augmented = []
y_augmented = []

for cls in unique_classes:
    X_cls = X_resampled[y_resampled == cls]
    y_cls = y_resampled[y_resampled == cls]
    
    if len(X_cls) < 100:  # Arbitrary threshold for augmentation
        X_cls, y_cls = resample(X_cls, y_cls, replace=True, n_samples=100, random_state=42)
    
    X_augmented.append(X_cls)
    y_augmented.append(y_cls)

X_augmented = np.vstack(X_augmented)
y_augmented = np.hstack(y_augmented)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=42)

# Step 6: Define the BaggingClassifier with RandomForest as the base estimator
base_rf = RandomForestClassifier(n_estimators=40, max_depth=20, min_samples_split=8, min_samples_leaf=4, random_state=42)
bagging_model = BaggingClassifier(base_estimator=base_rf, n_estimators=12, random_state=42)

# Step 7: Train the Bagging model
bagging_model.fit(X_train, y_train)

# Step 8: Perform cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cross_val_accuracies = cross_val_score(bagging_model, X_train, y_train, cv=kf, scoring='accuracy')
mean_cv_accuracy = np.mean(cross_val_accuracies)

print(f"Mean CV Accuracy: {mean_cv_accuracy}")

# Step 9: Evaluate the model on the test set
y_pred = bagging_model.predict(X_test)

train_accuracy = bagging_model.score(X_train, y_train)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"Training Accuracy: {train_accuracy}")
print(f"Test Set Accuracy: {test_accuracy}")

# Step 10: Detailed classification report
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Step 11: Save the model and label encoder
joblib.dump(bagging_model, 'heart_disease_bagging_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

# Step 12: Test function to make predictions with a new audio file
def extract_features_from_audio(audio_file_path, n_mfcc=13):
    audio, sample_rate = librosa.load(audio_file_path, sr=None)
    
    if len(audio) < 1024:
        # Pad short signals
        audio = np.pad(audio, (0, max(0, 1024 - len(audio))), mode='constant')
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    n_fft = min(2048, len(audio))  # Adjust n_fft based on length of audio
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, n_fft=n_fft)
    chroma_mean = np.mean(chroma.T, axis=0)
    
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate, n_fft=n_fft)
    spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
    
    tonnetz = librosa.feature.tonnetz(y=audio, sr=sample_rate)
    tonnetz_mean = np.mean(tonnetz.T, axis=0)
    
    features = np.concatenate((mfccs_mean, chroma_mean, spectral_contrast_mean, tonnetz_mean))
    return features

# Load model and label encoder
bagging_model = joblib.load('heart_disease_bagging_model.pkl')
le = joblib.load('label_encoder.pkl')

# Sample test prediction
audio_file_path = 'dataset-hb/Atraining_murmur/201101051114.wav'
features = extract_features_from_audio(audio_file_path)

if features is not None:
    features = features.reshape(1, -1)  # Reshape for the model
    predicted_label = bagging_model.predict(features)
    predicted_class = le.inverse_transform(predicted_label)
    print("Predicted Heart Condition:", predicted_class[0])
else:
    print("Feature extraction failed for the provided audio file.")
