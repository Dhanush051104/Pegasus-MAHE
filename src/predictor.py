import joblib
import numpy as np
from tensorflow.keras.models import load_model
from .features import CANFeatureExtractor

class IDSPredictor:
    def __init__(self, model_dir='../models'):
        # 1. Load the Brains
        print("🧠 Loading models into memory...")
        self.rf_model = joblib.load(f'{model_dir}/rf_model.joblib')
        self.lstm_model = load_model(f'{model_dir}/lstm_model.keras')
        self.scaler = joblib.load(f'{model_dir}/scaler.joblib')
        
        # 2. Initialize the Translator
        self.extractor = CANFeatureExtractor(self.scaler)
        
        # 3. Set the Alarm Threshold (from our Colab test)
        self.anomaly_threshold = 0.1650 

    def analyze_frame(self, raw_frame):
        """
        The main entry point for the IDS.
        Takes a raw dict and returns a safety decision.
        """
        # Get features for both models
        rf_input, lstm_input = self.extractor.process_single_frame(raw_frame)
        
        # --- BRAIN 1: Random Forest (The Bouncer) ---
        is_known_attack = self.rf_model.predict(rf_input)[0]
        rf_confidence = np.max(self.rf_model.predict_proba(rf_input))

        # --- BRAIN 2: LSTM (The Heart Monitor) ---
        is_anomaly = False
        reconstruction_error = 0
        
        if lstm_input is not None:
            prediction = self.lstm_model.predict(lstm_input, verbose=0)
            reconstruction_error = np.mean(np.abs(prediction - lstm_input))
            if reconstruction_error > self.anomaly_threshold:
                is_anomaly = True

        # --- THE DECISION LOGIC (The Hybrid Shield) ---
        if is_known_attack == 1 or is_anomaly:
            return {
                "action": "BLOCK",
                "reason": "Known Attack" if is_known_attack == 1 else "Stealth Anomaly",
                "confidence": float(rf_confidence) if is_known_attack == 1 else float(reconstruction_error)
            }
        
        return {"action": "ALLOW", "reason": "Normal Traffic", "confidence": 1.0}