import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load('yield_model.pkl')

# Get Feature Names (Must match training order)
feature_names = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall', 'Crop_Encoded']

# Get Importance
importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\n📊 WHAT YOUR AI CARES ABOUT:")
print("-----------------------------")
print(importance_df)
print("-----------------------------")
print("If 'Crop_Encoded' or 'Nitrogen' is near 0.99, the model is ignoring weather.\n")