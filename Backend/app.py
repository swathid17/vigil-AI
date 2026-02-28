from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import joblib
import os
import tempfile

app = Flask(__name__)
CORS(app)

# Model will be created in Step 5 (training)
MODEL_PATH = os.path.join("..", "models", "sed_model.joblib")
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def extract_features(path):
    # Load audio
    y, sr = librosa.load(path, sr=16000, mono=True)
    if y.size == 0:
        raise ValueError("Empty audio")

    # Normalize
    y = librosa.util.normalize(y)

    # MFCC features (fast + works)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = mfcc.mean(axis=1)

    # Extra features
    rms = float(librosa.feature.rms(y=y).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())
    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())

    feat = np.concatenate([mfcc_mean, [rms, zcr, centroid]])
    dbg = {"rms": rms, "zcr": zcr, "centroid": centroid}
    return feat, dbg

def topk_from_probs(classes, probs, k=3):
    idxs = np.argsort(probs)[::-1][:k]
    return [{"label": str(classes[i]), "prob": float(probs[i])} for i in idxs]

def fuse_to_threat(pred_label, topk, feat_dbg, context):
    # Base risk by event label (you can adjust later)
    base = {
        "gun_shot": 92,
        "siren": 70,
        "glass_break": 78,
        "scream": 82,
        "street_music": 20,
        "children_playing": 15,
        "dog_bark": 18,
        "engine_idling": 25,
        "air_conditioner": 10,
        "jackhammer": 35,
        "normal": 10,
    }.get(pred_label, 35)

    conf = topk[0]["prob"] if topk else 0.5
    rms = feat_dbg.get("rms", 0.0)

    # Explainable contributions (simple, judge-friendly)
    reasons = []
    score = base
    reasons.append(f"Event='{pred_label}' base={base}")

    # confidence adds up to +10
    conf_add = int(min(10, round(conf * 10)))
    score += conf_add
    reasons.append(f"Confidence={conf:.2f} adds +{conf_add}")

    # high energy adds up to +10
    energy_add = 0
    if rms > 0.07:
        energy_add = 10
    elif rms > 0.04:
        energy_add = 6
    elif rms > 0.02:
        energy_add = 3
    score += energy_add
    reasons.append(f"Energy(RMS)={rms:.3f} adds +{energy_add}")

    # Context-based threshold + calibration
    # Home: stricter to avoid domestic false alarms
    # Outdoor: more sensitive
    if context == "home":
        threshold = 80
        score -= 8
        reasons.append("Context=home: score -8, threshold=80")
    else:
        threshold = 70
        score += 5
        reasons.append("Context=outdoor: score +5, threshold=70")

    score = int(max(0, min(100, score)))

    if score >= threshold:
        status = "ALERT"
        dispatch = "contacts_then_police"
    elif score >= threshold - 15:
        status = "WARNING"
        dispatch = "contacts"
    else:
        status = "SAFE"
        dispatch = "none"

    # Scene inference (PS alignment)
    scene_map = {
        "gun_shot": "Possible firearm discharge in the environment.",
        "glass_break": "Possible break-in / forced entry signal.",
        "siren": "Emergency vehicle / public safety signal detected.",
        "scream": "Distress vocalization detected.",
    }
    scene = scene_map.get(pred_label, "General audio scene detected.")

    why = " | ".join(reasons)
    return score, threshold, status, dispatch, scene, why

@app.post("/analyze")
def analyze():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    context = request.form.get("context", "outdoor").strip().lower()
    if context not in ["home", "outdoor"]:
        context = "outdoor"

    f = request.files["audio"]

    # Save to temp file
    suffix = os.path.splitext(f.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        f.save(tmp_path)

    try:
        feat, dbg = extract_features(tmp_path)

        if model is None:
            # Until training is done, show demo fallback but still V2 structure
            classes = np.array(["normal", "siren", "glass_break"])
            probs = np.array([0.70, 0.20, 0.10])
        else:
            probs = model.predict_proba([feat])[0]
            classes = model.classes_

        topk = topk_from_probs(classes, probs, k=3)
        pred_label = topk[0]["label"] if topk else "normal"

        score, threshold, status, dispatch, scene, why = fuse_to_threat(
            pred_label, topk, dbg, context
        )

        return jsonify({
            "status": status,
            "threatScore": score,
            "threshold": threshold,
            "predictedEvent": pred_label,
            "topK": topk,                 # NEW: shows model confidence (V2 feature)
            "features": dbg,              # NEW: explainability
            "scene": scene,               # NEW: PS alignment (reasoning)
            "why": why,                   # NEW: explainability text
            "dispatch": dispatch
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass

@app.get("/health")
def health():
    return jsonify({"ok": True, "modelLoaded": model is not None})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
