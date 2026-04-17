import numpy as np
import pandas as pd
from collections import deque

class CANFeatureExtractor:
    def __init__(self, scaler):
        self.scaler = scaler
        # A 'deque' is a special list that automatically drops old items.
        # We use it to keep exactly the last 10 messages for the LSTM.
        self.window = deque(maxlen=10)
        self.feature_cols = ['CAN_ID_INT', 'DLC', 'D1_INT', 'D2_INT', 
                             'D3_INT', 'D4_INT', 'D5_INT', 'D6_INT', 
                             'D7_INT', 'D8_INT']

    def hex_to_int(self, hex_val):
        try:
            return int(str(hex_val), 16) if pd.notnull(hex_val) and str(hex_val).strip() != '' else 0
        except:
            return 0

    def process_single_frame(self, can_msg):
        """
        Takes a raw CAN message (dict) and returns:
        1. Features for Random Forest (1D array)
        2. Sequence for LSTM (3D array or None)
        """
        # 1. Convert hex to integers
        raw_features = [
            self.hex_to_int(can_msg['CAN_ID']),
            int(can_msg['DLC']),
            *[self.hex_to_int(can_msg.get(f'D{i}', 0)) for i in range(1, 9)]
        ]
        
        # 2. Scale the features
        scaled_features = self.scaler.transform([raw_features])[0]
        
        # 3. Update the sliding window for the LSTM
        self.window.append(scaled_features)
        
        # 4. Prepare the LSTM sequence (only if we have 10 messages)
        sequence = None
        if len(self.window) == 10:
            sequence = np.array(self.window).reshape(1, 10, 10)
            
        return scaled_features.reshape(1, -1), sequence