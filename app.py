import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import pywt
from scipy.stats import skew, kurtosis
from scipy.signal import welch, butter, filtfilt
from sklearn.preprocessing import StandardScaler

feature_columns = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8', 'AccX', 'AccY', 'AccZ', 'Gyro1', 'Gyro2', 'Gyro3']

# -----------------------
# MODEL PATHS
# -----------------------
MI_MODEL_PATH = "Models/MI_Model.pkl"
MI_indices = np.load("Models/top100_indices.npy")

SSVEP_MODEL_PATH = "Models/SSVEP_Model.pkl"
SSVEP_indices = np.load("Models/SSVEP_indices.npy")

# -----------------------
# EEG Reshaping Function
# -----------------------
def reshape_data(df_data, feature_columns, group_size=2250):
    X = df_data[feature_columns].values
    n_samples = X.shape[0] // group_size
    X_reshaped = X[:n_samples * group_size].reshape(n_samples, group_size, len(feature_columns))
    return X_reshaped

# -----------------------
# Feature Extraction - Common
# -----------------------
def extract_features_from_sequence(seq):
    features = {}
    for i in range(seq.shape[1]):
        channel = seq[:, i]
        features[f'ch{i}_mean'] = np.mean(channel)
        features[f'ch{i}_std'] = np.std(channel)
        features[f'ch{i}_min'] = np.min(channel)
        features[f'ch{i}_max'] = np.max(channel)
        features[f'ch{i}_median'] = np.median(channel)
        features[f'ch{i}_range'] = np.ptp(channel)
        features[f'ch{i}_energy'] = np.sum(channel**2)
        features[f'ch{i}_skew'] = skew(channel)
        features[f'ch{i}_kurtosis'] = kurtosis(channel)
        features[f'ch{i}_zcr'] = ((np.diff(np.sign(channel)) != 0).sum()) / len(channel)
    return features

def extract_fft_features(seq):
    features = {}
    for i in range(seq.shape[1]):
        channel = seq[:, i]
        fft_vals = np.fft.rfft(channel)
        fft_power = np.abs(fft_vals) ** 2
        features[f'ch{i}_fft_mean'] = np.mean(fft_power)
        features[f'ch{i}_fft_std'] = np.std(fft_power)
        features[f'ch{i}_fft_max'] = np.max(fft_power)
        features[f'ch{i}_fft_freq_max'] = np.argmax(fft_power)
    return features

def extract_wavelet_features(seq):
    features = {}
    for i in range(seq.shape[1]):
        coeffs = pywt.wavedec(seq[:, i], 'db4', level=3)
        for j, c in enumerate(coeffs):
            features[f'ch{i}_w{j}_mean'] = np.mean(c)
            features[f'ch{i}_w{j}_std'] = np.std(c)
    return features

def extract_target_freq_power(seq, fs, target_freqs):
    features = {}
    n = seq.shape[0]
    freqs = np.fft.rfftfreq(n, d=1/fs)
    window = np.hanning(n)
    for i in range(seq.shape[1]):
        windowed_signal = seq[:, i] * window
        fft_vals = np.fft.rfft(windowed_signal)
        power = np.abs(fft_vals) ** 2
        for f in target_freqs:
            idx_range = np.where((freqs >= f-0.5) & (freqs <= f+0.5))[0]
            if len(idx_range) > 0:
                features[f'ch{i}_power_{f}Hz'] = np.mean(power[idx_range])
                features[f'ch{i}_power_{f}Hz_peak'] = np.max(power[idx_range])
    return features

# -----------------------
# MI Feature Extractors
# -----------------------
mi_freqs = [10, 13]

def extract_mi_power_features(seq, fs=250):
    features = {}
    mi_channels = ['C3', 'C4', 'CZ']
    channel_indices = [feature_columns.index(ch) for ch in mi_channels if ch in feature_columns]
    for i in channel_indices:
        f, pxx = welch(seq[:, i], fs=fs, nperseg=min(256, seq.shape[0]))
        for freq in mi_freqs:
            idx = np.argmin(np.abs(f - freq))
            features[f'ch{i}_{freq}Hz_power'] = pxx[idx]
            neighbors = np.where((f >= freq-2) & (f <= freq+2) & (np.abs(f - freq) > 0.5))[0]
            if len(neighbors) > 0:
                neighbor_power = np.mean(pxx[neighbors])
                features[f'ch{i}_{freq}Hz_ratio'] = pxx[idx] / (neighbor_power + 1e-10)
    return features

def extract_combined_mi_features(seq, fs=250):
    features = extract_features_from_sequence(seq)
    features.update(extract_fft_features(seq))
    features.update(extract_wavelet_features(seq))
    features.update(extract_target_freq_power(seq, fs, mi_freqs))
    features.update(extract_mi_power_features(seq, fs))
    return features

def transform_mi_sequences(X):
    return pd.DataFrame([extract_combined_mi_features(seq) for seq in X])

# -----------------------
# SSVEP Feature Extractors
# -----------------------
def extract_ssvep_features(seq, fs=250):
    features = {}
    ssvep_freqs = {'left': 10, 'right': 13, 'forward': 7, 'backward': 8}
    ssvep_channels = ['PZ', 'PO7', 'OZ', 'PO8']
    channel_indices = [feature_columns.index(ch) for ch in ssvep_channels if ch in feature_columns]
    for i in channel_indices:
        f, pxx = welch(seq[:, i], fs=fs, nperseg=min(256, seq.shape[0]))
        for label, freq in ssvep_freqs.items():
            idx = np.argmin(np.abs(f - freq))
            features[f'ch{i}_{label}_power'] = pxx[idx]
            neighbors = np.where((f >= freq-2) & (f <= freq+2) & (np.abs(f - freq) > 0.5))[0]
            if len(neighbors) > 0:
                neighbor_power = np.mean(pxx[neighbors])
                features[f'ch{i}_{label}_power_ratio'] = pxx[idx] / (neighbor_power + 1e-10)
    return features

def extract_combined_ssvep_features(seq, fs=250):
    features = extract_features_from_sequence(seq)
    features.update(extract_fft_features(seq))
    features.update(extract_wavelet_features(seq))
    features.update(extract_target_freq_power(seq, fs, [7, 8, 10, 13]))
    features.update(extract_ssvep_features(seq, fs))
    return features

def transform_ssvep_sequences(X):
    return pd.DataFrame([extract_combined_ssvep_features(seq) for seq in X])

# -----------------------
# Preprocessing
# -----------------------
def bandpass_filter(data, lowcut=5, highcut=40, fs=250, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def preprocess_eeg_data(X, feature_columns):
    eeg_channels = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
    eeg_indices = [feature_columns.index(ch) for ch in eeg_channels if ch in feature_columns]
    X_processed = X.copy()
    for i in range(X.shape[0]):
        for j in eeg_indices:
            X_processed[i, :, j] = bandpass_filter(X[i, :, j], 5, 40)
            X_processed[i, :, j] = bandpass_filter(X[i, :, j], 6, 15)
    return X_processed

# -----------------------
# Streamlit UI
# -----------------------
st.title("🧠 EEG MI & SSVEP Classifier")

# Upload EEG CSV files
mi_files = st.file_uploader("📁 Upload up to 5 MI EEG CSV Files", type=["csv"], accept_multiple_files=True, key="mi")
ssvep_files = st.file_uploader("📁 Upload up to 5 SSVEP EEG CSV Files", type=["csv"], accept_multiple_files=True, key="ssvep")

predict_button = st.button("🚀 Run Prediction")

if predict_button:
    results = []

    if mi_files:
        if len(mi_files) > 5:
            st.warning(f"⚠️ {len(mi_files)} MI files uploaded. Only the first 5 will be used.")
            mi_files = mi_files[:5]

    if mi_files and 1 <= len(mi_files) <= 5:
        df_mi = pd.concat([pd.read_csv(f) for f in mi_files], ignore_index=True)
        X_mi = reshape_data(df_mi, feature_columns, 2250)
        mi_features = transform_mi_sequences(X_mi)
        X_mi_scaled = StandardScaler().fit_transform(mi_features)
        X_mi_final = X_mi_scaled[:, MI_indices]
        mi_model = joblib.load(MI_MODEL_PATH)
        mi_preds = mi_model.predict(X_mi_final)
        results.append(pd.DataFrame({
            'id': range(4901, 4901 + len(mi_preds)),
            'Class': ["Left" if p == 0 else "Right" for p in mi_preds]
        }))

    if ssvep_files:
        if len(ssvep_files) > 5:
            st.warning(f"⚠️ {len(ssvep_files)} MI files uploaded. Only the first 5 will be used.")
            ssvep_files = ssvep_files[:5]

    if ssvep_files and 1 <= len(ssvep_files) <= 5:
        df_ssvep = pd.concat([pd.read_csv(f) for f in ssvep_files], ignore_index=True)
        X_ssvep = reshape_data(df_ssvep, feature_columns, 1750)
        X_ssvep_filtered = preprocess_eeg_data(X_ssvep, feature_columns)
        ssvep_features = transform_ssvep_sequences(X_ssvep_filtered)
        X_ssvep_scaled = StandardScaler().fit_transform(ssvep_features)
        X_ssvep_final = X_ssvep_scaled[:, SSVEP_indices]
        ssvep_model = joblib.load(SSVEP_MODEL_PATH)
        ssvep_preds = ssvep_model.predict(X_ssvep_final)
        label_map = {0: "Left", 1: "Right", 2: "Forward", 3: "Backward"}
        results.append(pd.DataFrame({
            'id': range(4951, 4951 + len(ssvep_preds)),
            'Class': [label_map[p] for p in ssvep_preds]
        }))

    if results:
        final_df = pd.concat(results, ignore_index=True)
        st.subheader("📊 Combined MI + SSVEP Predictions")
        st.write(final_df)
        csv = final_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", csv, "final_predictions.csv", "text/csv")