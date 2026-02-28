
# VIGIL-AI  
AI-Powered Audio Scene Understanding & Threat Detection System

##  Overview

VIGIL-AI is a web-based audio intelligence prototype that analyzes environmental sounds and converts them into a calibrated **Threat Score (0–100)** with controlled alert escalation.

The system classifies acoustic events (e.g., gunshot, siren, street activity), evaluates their risk level, and determines whether the situation is **SAFE, WARNING, or ALERT**, while reducing false alarms using explainable decision logic and a cancel window.

This project was developed as a hackathon prototype to demonstrate practical AI-based safety monitoring.

## Problem Statement

Traditional emergency systems depend heavily on manual triggers.  
There is a need for an intelligent solution that can:

- Analyze environmental audio
- Detect potentially dangerous acoustic events
- Estimate threat severity
- Escalate alerts responsibly
- Prevent false emergency dispatch

VIGIL-AI addresses this using machine learning and rule-based fusion logic.

##  Key Features

- Audio Scene Classification (Top-3 predictions with confidence)
- Threat Score Calculation (0–100 scale)
- Explainable AI (shows WHY a score was generated)
- Context-aware threshold (Home / Outdoor mode)
- False-alarm safeguards (confidence & energy checks)
- 5-Second Cancel Window before escalation
- Local History Logging (Demo simulation)

## System Architecture

Audio Input  
→ Feature Extraction (MFCC, RMS, ZCR, Spectral Centroid)  
→ Machine Learning Model  
→ Rule-Based Fusion Engine  
→ Threat Score  
→ Alert Decision (SAFE / WARNING / ALERT)

## Tech Stack

### Backend
- Python
- Flask
- Librosa (audio feature extraction)
- Scikit-learn (Logistic Regression model)
- Joblib (model storage)

### Frontend
- HTML
- CSS
- JavaScript (Fetch API)
- Local Storage (history simulation)

## Dataset

The model is trained using **UrbanSound8K**, a public dataset containing 8,732 labeled urban sound clips including:

- gun_shot
- siren
- street_music
- children_playing
- dog_bark
- drilling
- and other environmental sounds

The dataset helps the model learn acoustic feature patterns for urban audio scenes.

Dataset is not included in this repository due to size.

## How It Works

1. User uploads an audio file.
2. Backend extracts acoustic features:
   - MFCC (Mel-frequency cepstral coefficients)
   - RMS (energy)
   - Zero Crossing Rate
   - Spectral Centroid
3. The trained model predicts class probabilities.
4. A rule-based fusion engine calculates:
   - Base risk from predicted class
   - Confidence contribution
   - Energy contribution
   - Context adjustment
5. A final Threat Score is generated.
6. The system outputs:
   - SAFE
   - WARNING
   - ALERT
7. If ALERT → a 5-second cancel window activates before dispatch.

## False Alert Protection

To reduce incorrect emergency triggers:

- Low-confidence predictions are suppressed
- Context-based thresholding adjusts sensitivity
- Cancel window allows manual override
- Explainable scoring improves transparency

## Future Improvements

- Replace baseline classifier with CNN (YAMNet / PANNs)
- Real-time microphone streaming
- Mobile app deployment
- GPS-based risk calibration
- Cloud deployment with scalable inference

## Project Highlights

- Lightweight ML prototype
- Explainable AI decision pipeline
- Ethical alert escalation mechanism
- Real working backend + frontend integration

## License

This project is developed for educational and hackathon demonstration purposes.

