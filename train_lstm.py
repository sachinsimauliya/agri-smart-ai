import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, Concatenate, Flatten

DATA_FILE = 'crop_yield_data.csv'

def get_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        column_mapping = {
            'N_req_kg_per_ha': 'Nitrogen',
            'P_req_kg_per_ha': 'Phosphorus',
            'K_req_kg_per_ha': 'Potassium',
            'Temperature_C': 'Temperature',
            'Humidity_%': 'Humidity',
            'pH': 'pH',
            'Rainfall_mm': 'Rainfall',
            'Crop': 'Crop',
            'Yield_kg_per_ha': 'Yield'
        }
        df.rename(columns=column_mapping, inplace=True)
        return df
    return None

df = get_data()

if df is not None:
    df = df.dropna()
    le = LabelEncoder()
    df['Crop_Encoded'] = le.fit_transform(df['Crop'])
    
    features = ['Temperature', 'Humidity', 'pH', 'Rainfall', 'Crop_Encoded']
    target = 'Yield'

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[features])
    y = df[target].values

    # Reshape for LSTM
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

    # Build LSTM Model
    input_layer = Input(shape=(1, len(features)))
    lstm_out = LSTM(64, return_sequences=True)(input_layer)
    lstm_out = Dropout(0.2)(lstm_out)
    attn_out = Attention()([lstm_out, lstm_out])
    concat = Concatenate()([lstm_out, attn_out])
    flat = Flatten()(concat)
    output_layer = Dense(1, activation='linear')(flat)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')

    print("Training Advanced AI (This may take a minute)...")
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    model.save('lstm_model.h5')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le, 'label_encoder_lstm.pkl')
    print("✅ Advanced AI Trained! (lstm_model.h5)")