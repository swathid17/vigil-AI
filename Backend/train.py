import os
import pandas as pd
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Update this path
ESC_ROOT = os.path.join("..", "datasets", "ESC-50")
CSV_PATH = os.path.join(ESC_ROOT, "meta", "esc50.csv")
AUDIO_DIR = os.path.join(ESC_ROOT, "audio")

# Choose classes for demo
TARGET = {
    "glass_breaking": "glass",
    "siren": "siren"
    # Add more if you want: "crowd": "normal" etc.
}

def extract(path):
    y, sr = librosa.load(path, sr=16000, mono=True)
    y = librosa.util.normalize(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).mean(axis=1)
    rms = librosa.feature.rms(y=y).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    return np.concatenate([mfcc, [rms, zcr, centroid]])

def main():
    df = pd.read_csv(CSV_PATH)

    X, y = [], []
    for _, row in df.iterrows():
        cat = row["category"]
        if cat not in TARGET:
            continue
        label = TARGET[cat]
        wav = row["filename"]
        path = os.path.join(AUDIO_DIR, wav)
        if os.path.exists(path):
            X.append(extract(path))
            y.append(label)

    X = np.array(X); y = np.array(y)
    print("Samples:", len(y), "Classes:", set(y))

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print(classification_report(yte, pred))

    out = os.path.join("..", "models", "sed_model.joblib")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    joblib.dump(model, out)
    print("Saved:", out)

if __name__ == "__main__":
    main()
