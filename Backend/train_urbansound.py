import os
import numpy as np
import pandas as pd
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

print("TRAIN SCRIPT STARTED")

DATASET_ROOT = os.path.join("..", "datasets", "UrbanSound8K")
CSV_PATH = os.path.join(DATASET_ROOT, "metadata", "UrbanSound8K.csv")
AUDIO_ROOT = os.path.join(DATASET_ROOT, "audio")

TARGET_CLASSES = {"gun_shot", "siren", "street_music", "children_playing"}

def extract_features(path):
    y, sr = librosa.load(path, sr=16000, mono=True)
    if y.size == 0:
        return None

    y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).mean(axis=1)
    rms = float(librosa.feature.rms(y=y).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())
    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())

    return np.concatenate([mfcc, [rms, zcr, centroid]])

def main():
    print("MAIN RUNNING")

    df = pd.read_csv(CSV_PATH)
    df = df[df["class"].isin(TARGET_CLASSES)].copy()

    X, y = [], []

    print("Total rows for selected classes:", len(df))

    for _, row in df.iterrows():
        fold = f"fold{int(row['fold'])}"
        filename = row["slice_file_name"]
        label = row["class"]

        wav_path = os.path.join(AUDIO_ROOT, fold, filename)

        if not os.path.exists(wav_path):
            continue

        feat = extract_features(wav_path)
        if feat is None:
            continue

        X.append(feat)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    print("Final samples:", len(y))
    print("Labels:", sorted(set(y.tolist())))

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2500))
    ])

    model.fit(Xtr, ytr)

    preds = model.predict(Xte)
    acc = accuracy_score(yte, preds)

    print("\nAccuracy:", round(acc, 4))
    print("\nClassification report:\n", classification_report(yte, preds))

    out_path = os.path.join("..", "models", "sed_model.joblib")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(model, out_path)

    print("\n✅ Saved model to:", out_path)

if __name__ == "__main__":
    main()