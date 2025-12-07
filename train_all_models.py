import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, Concatenate, Flatten
import joblib
import os

print("🚀 Initializing Super Duper AI Training...")

# ==========================================
# 1. TRAIN YIELD MODELS (Regression)
# ==========================================
if os.path.exists('crop_yield_data.csv'):
    print("\n[1/3] Training Yield Models...")
    df_yield = pd.read_csv('crop_yield_data.csv')
    
    # Standardize Column Names
    yield_map = {
        'N_req_kg_per_ha': 'Nitrogen', 'P_req_kg_per_ha': 'Phosphorus', 'K_req_kg_per_ha': 'Potassium',
        'Temperature_C': 'Temperature', 'Humidity_%': 'Humidity', 'pH': 'pH', 'Rainfall_mm': 'Rainfall',
        'Crop': 'Crop', 'Yield_kg_per_ha': 'Yield'
    }
    df_yield.rename(columns=yield_map, inplace=True)
    df_yield = df_yield.dropna()

    # Shared Encoder (Fixes the "Maize not known" error)
    le_yield = LabelEncoder()
    df_yield['Crop_Encoded'] = le_yield.fit_transform(df_yield['Crop'])

    # Inputs: Weather + Soil pH + Crop (No N/P/K to prevent cheating)
    features = ['Temperature', 'Humidity', 'pH', 'Rainfall', 'Crop_Encoded']
    X = df_yield[features]
    y = df_yield['Yield']

    # --- A. SUPER ENSEMBLE MODEL (Stacking) ---
    # Combines Random Forest and Gradient Boosting
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42))
    ]
    super_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
    super_model.fit(X, y)
    
    joblib.dump(super_model, 'yield_model.pkl')
    joblib.dump(le_yield, 'yield_encoder.pkl')
    print("✅ Super Ensemble Model Saved (yield_model.pkl)")

    # --- B. DEEP LEARNING MODEL (Attention-LSTM) ---
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1])) # Reshape for LSTM

    input_layer = Input(shape=(1, len(features)))
    lstm_out = LSTM(64, return_sequences=True)(input_layer)
    lstm_out = Dropout(0.2)(lstm_out)
    attn_out = Attention()([lstm_out, lstm_out])
    concat = Concatenate()([lstm_out, attn_out])
    flat = Flatten()(concat)
    output_layer = Dense(1, activation='linear')(flat)

    lstm_model = Model(inputs=input_layer, outputs=output_layer)
    lstm_model.compile(optimizer='adam', loss='mse')
    
    print("Training Attention-LSTM (Deep Learning)...")
    lstm_model.fit(X_reshaped, y, epochs=10, batch_size=64, verbose=0)
    
    lstm_model.save('lstm_model.h5')
    joblib.dump(scaler, 'scaler.pkl')
    print("✅ Attention-LSTM Model Saved (lstm_model.h5)")

else:
    print("❌ Error: 'crop_yield_data.csv' not found!")

# ==========================================
# 2. TRAIN CROP RECOMMENDER (Classification)
# ==========================================
if os.path.exists('crop_recommendation.csv'):
    print("\n[2/3] Training Crop Recommender...")
    df_rec = pd.read_csv('crop_recommendation.csv')
    
    X_r = df_rec[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y_r = df_rec['label']

    model_rec = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rec.fit(X_r, y_r)

    joblib.dump(model_rec, 'recommender_model.pkl')
    print("✅ Crop Recommender Saved (recommender_model.pkl)")
else:
    print("⚠️ 'crop_recommendation.csv' not found. Skipping Recommender.")

print("\n🎉 All Brains Trained Successfully!")