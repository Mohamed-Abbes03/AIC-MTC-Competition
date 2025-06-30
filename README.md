# EEG-Based Classification System for Motor Imagery (MI) and SSVEP Tasks

This repository contains a full pipeline for EEG-based classification using two tasks: Motor Imagery (MI) and Steady-State Visual Evoked Potentials (SSVEP). The project was part of the MTC-AIC3 AI competition.

## 🖥️ System Architecture Overview
1. *EEG Signals (CSV Files)*
   -
   ↓  
3. *Preprocessing*  
   - Filtering (bandpass)  
   - Denoising  
   ↓  
4. *Feature Extraction*  
   - Statistical features  
   - Frequency domain (FFT)  
   - Wavelet features  
   ↓  
5. *Feature Selection*  
   - Top 100 features using GradientBoosting importance  
   ↓  
6. *Model Training*  
   - XGBoost Classifier  
   ↓  
7. *Prediction and Evaluation*

The project is divided into two main pipelines:

- *Motor Imagery (MI)* for binary classification: Left vs. Right hand imagination
- *SSVEP* for multi-class classification: Left / Right / Forward / Backward

Each pipeline follows a structured flow:

1. Data Loading & Label Merging
2. Preprocessing (denoising, normalization, etc.)
3. Feature Extraction (statistical, FFT, wavelet, frequency-band)
4. Feature Selection using GradientBoosting
5. Model Training using XGBoost
6. Prediction & Export

## 🧪 Methodology

### 1. Preprocessing

*MI Task:*
- Drop irrelevant columns: Time, Battery, Counter
- Standardize numeric features using validation data statistics
- Select low-noise experiments using mean variance from IMU signals
- Remove trials with incomplete data

*SSVEP Task:*
- Clip training values to the range of validation data to reduce drift
- Bandpass filtering (5–40 Hz, then 6–15 Hz) for visual EEG channels

### 2. Segmentation
EEG data was segmented into trials:
- 2250 samples for MI trials
- 1750 samples for SSVEP trials

### 3. Feature Extraction

*Common features for both tasks:*
- Statistical: mean, std, min, max, median, range, energy, skew, kurtosis, zero-crossing rate
- FFT features: power mean/std/max + peak frequency
- Wavelet features using db4
- Band-specific frequency power (e.g., 10Hz, 13Hz)

*MI-specific:*
- Power and power-ratio in Mu/Beta bands from C3/C4/CZ using Welch's method

*SSVEP-specific:*
- Power and power-ratio from OZ, PO7, PO8, PZ at stimulus frequencies (7, 8, 10, 13 Hz)

### 4. Feature Selection
- Used a GradientBoostingClassifier to rank feature importance
- Top 100 features were retained for both tasks

### 5. Model Training
Trained XGBoost classifiers with the following config:

**MI XGBoost Config:**
```python
XGBClassifier(
    n_estimators=550,
    learning_rate=0.03,
    max_depth=1,
    subsample=0.7,
    colsample_bytree=0.5,
    min_child_weight=4,
    reg_alpha=1
)
```


**SSVEP XGBoost Config:**
```python
XGBClassifier(
    n_estimators=900,
    learning_rate=0.03,
    max_depth=1,
    subsample=0.8,
    colsample_bytree=0.8
)
```


### 6. Data Augmentation

MI:

    Oversampling with noise: random Gaussian noise added to under-represented classes

SSVEP:

    Frequency-related augmentation: feature values multiplied by small random factors


## 📊 Results

    MI Accuracy (Validation): ~76%

    SSVEP Accuracy (Validation): ~62%

Metrics include:

    Accuracy

    Confusion Matrix

    Classification Report

    Cross-validation accuracy


## 🚧 Challenges Faced

| Challenge                          | Solution |
|------------------------------------|----------|
| High variance in raw EEG signals   | Low-noise experiment selection based on IMU signal variance |
| Overfitting on small datasets      | Feature selection + strong regularization + data augmentation |
| Inconsistent feature ranges        | Standardization using validation set only |
| Noisy SSVEP test data              | Bandpass filtering and signal clipping |


## 📁 Outputs

- MI_Model.pkl → Trained MI XGBoost model  
- SSVEP_Model.pkl → Trained SSVEP XGBoost model  
- top100_indices.npy → Feature indices used in MI  
- SSVEP_indices.npy → Feature indices used in SSVEP  
- predictions.csv → Final predictions combining MI and SSVEP results  

## 🚀 How to Run

1. Upload EEG CSV files for MI and SSVEP  
2. Run the Streamlit app or use the prediction script  
3. Models and feature indices will be used automatically  

## 📚 Acknowledgments

- MTC AIC-3 EEG dataset provided for competition purposes  
- Scikit-learn, XGBoost, PyWavelets, and SciPy were critical libraries used
