from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import random
from datetime import datetime, timedelta
from io import BytesIO
import cv2 # Required for system setup, though imported via from...
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# --- MODEL LOADING ---
yield_model, yield_le, rec_model, lstm_model, scaler, diag_model = None, None, None, None, None, None

try:
    if os.path.exists('yield_model.pkl'): yield_model = joblib.load('yield_model.pkl')
    if os.path.exists('yield_encoder.pkl'): yield_le = joblib.load('yield_encoder.pkl')
    if os.path.exists('recommender_model.pkl'): rec_model = joblib.load('recommender_model.pkl')
    if os.path.exists('lstm_model.h5'):
        lstm_model = tf.keras.models.load_model('lstm_model.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
    
    # Loading the structure file saved by the enhanced training script
    if os.path.exists('diagnostics_model_structure.h5'):
        diag_model = tf.keras.models.load_model('diagnostics_model_structure.h5', compile=False)
        print("✅ Diagnostics Model Structure Loaded.")

except Exception as e:
    print(f"⚠️ Warning: Some models failed to load: {e}")

# --- GLOBAL CONSTANTS ---

# CROP NPK targets (N, P, K) - Nutrient requirements in kg/ha
CROP_NPK = {'rice': [100, 50, 40], 'maize': [120, 60, 40], 'cotton': [120, 60, 60], 'chickpea': [20, 70, 30], 'wheat': [120, 60, 30], 'sugarcane': [150, 75, 75], 'fruit trees': [50, 30, 80], 'coffee': [100, 40, 40]}

# DISEASE CLASSES (38 classes from PlantVillage Dataset)
DISEASE_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# CROP WATER REQUIREMENTS (CWR) for Smart Irrigation
CROP_WATER_FACTORS = {
    'Rice': 1.25, 'Maize': 1.00, 'Cotton': 0.85, 'Chickpea': 0.60, 'Fruits': 0.70
}

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

# 1. YIELD PREDICTION
@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    if not yield_model: return jsonify({'error': 'Yield Model not trained.'})
    d = request.json
    try:
        crop_enc = yield_le.transform([d['Crop']])[0]
        raw = [float(d['Temp']), float(d['Humidity']), float(d['pH']), float(d['Rainfall']), float(crop_enc)]
        
        result = 0
        mod = "Super Ensemble"

        if d.get('Model_Type') == 'LSTM' and lstm_model:
            scaled = scaler.transform([raw])
            result = float(lstm_model.predict(scaled.reshape(1, 1, 5), verbose=0)[0][0])
            mod = "Attention-LSTM"
        else:
            result = float(yield_model.predict(pd.DataFrame([raw], columns=['Temperature', 'Humidity', 'pH', 'Rainfall', 'Crop_Encoded']))[0])
        
        result = max(0, result) 
        
        return jsonify({
            'Result': round(result, 2), 
            'Model': mod
        })
    except Exception as e: 
        return jsonify({'error': str(e)})

# 2. CROP RECOMMENDATION
@app.route('/recommend_crop', methods=['POST'])
def recommend_crop():
    if not rec_model: return jsonify({'error': 'Recommender not trained.'})
    d = request.json
    try:
        feats = [[float(d['N']), float(d['P']), float(d['K']), float(d['Temp']), float(d['Humidity']), float(d['pH']), float(d['Rainfall'])]]
        return jsonify({'Result': rec_model.predict(feats)[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

# 3. SMART IRRIGATION
@app.route('/smart_irrigation', methods=['POST'])
def smart_irrigation():
    d = request.json
    temp = float(d['Temp'])
    hum = float(d['Humidity'])
    rain = float(d['Rainfall'])
    crop = d['Crop'] 
    
    cwr_factor = CROP_WATER_FACTORS.get(crop, 1.0)
    base_need = (temp * 0.5) + (100 - hum) * 0.2
    adjusted_need = base_need * cwr_factor

    required_water = (adjusted_need * 10) - rain

    if required_water <= 5: 
        status = "No Irrigation Needed"
        advice = f"Soil moisture is sufficient for {crop}. Check again tomorrow."
    else:
        status = "Irrigation Required"
        advice = f"Apply {round(required_water, 2)} mm water immediately for optimal {crop} health."
    
    return jsonify({'Status': status, 'Advice': advice})


# 4. FERTILIZER ADVISOR
@app.route('/fertilizer_advice', methods=['POST'])
def fertilizer_advice():
    d = request.json
    crop = d['Crop'].lower()
    
    t = CROP_NPK.get(crop, [80, 40, 40])
    diff = [t[0]-float(d['N']), t[1]-float(d['P']), t[2]-float(d['K'])]
    
    FULL_NAMES = ['Nitrogen', 'Phosphorus', 'Potassium']
    
    res = []
    
    for i, nutrient in enumerate(FULL_NAMES):
        if diff[i] > 5:
            res.append(f"Add {round(diff[i], 1)} kg/ha {nutrient}")
            
    return jsonify({'Result': "<br>".join(res) if res else "Optimal Nutrient Levels!"})

# 5. STORAGE ADVISOR
@app.route('/storage_life', methods=['POST'])
def storage_life():
    d = request.json
    temp = float(d['Temp'])
    hum = float(d['Humidity'])
    crop = d['Crop']

    BASE_LIFE = {'Rice': 365, 'Maize': 180, 'Cotton': 250, 'Chickpea': 150, 'Coffee': 300, 'Pomegranate': 45, 'Apple': 60, 'Grapes': 15}.get(crop, 90)
    
    decay_factor = 1.0
    if temp > 15:
        temp_loss = (temp - 15) * 0.015
        decay_factor -= min(0.5, temp_loss)
        
    if hum > 60:
        hum_loss = (hum - 60) * 0.008
        decay_factor -= min(0.3, hum_loss)
    
    decay_factor = max(0.2, decay_factor)
    real_life = int(BASE_LIFE * decay_factor)

    if real_life >= BASE_LIFE * 0.9: advice = "Optimal Conditions. Shelf life is projected to be maintained."
    elif real_life >= BASE_LIFE * 0.7: advice = "Moderate Risk. Check for initial spoilage symptoms."
    else: advice = "Critical Spoilage Risk. Ventilate immediately and consider selling within the next week."

    return jsonify({'Days': real_life, 'Advice': advice})

# 6. DIAGNOSTICS AI 
@app.route('/diagnose_disease', methods=['POST'])
def diagnose_disease():
    if diag_model is None:
        return jsonify({'result': 'Error: Diagnostics Model not loaded.', 'advice': 'Please run training script.'})

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'})

    file = request.files['image']
    
    # --- 1. Preprocessing ---
    try:
        img = load_img(BytesIO(file.read()), target_size=(128, 128))
        img_array = img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        processed_image = vgg16_preprocess_input(img_batch)
        
    except Exception as e:
        return jsonify({'result': 'Error during image processing.', 'advice': f'Failed to process image: {str(e)}'})

    # --- 2. Prediction ---
    try:
        prediction = diag_model.predict(processed_image)
        
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        confidence = float(prediction[0][predicted_class_index])
        
        if predicted_class_index < len(DISEASE_CLASSES):
            pred_class = DISEASE_CLASSES[predicted_class_index]
        else:
            pred_class = "Unidentified Disease/Healthy"
            
        # --- 3. Advice Lookup ---
        if 'healthy' in pred_class.lower():
            advice = 'No action required. Maintain nutrients and monitor for stress.'
        elif 'black_rot' in pred_class.lower():
            advice = 'Highly confident of Black Rot. Prune infected branches and apply fungicide (e.g., Mancozeb).'
        elif 'scab' in pred_class.lower():
            advice = 'Apple Scab detected. Apply early season fungicide.'
        elif 'blight' in pred_class.lower() or 'spot' in pred_class.lower():
            advice = 'Leaf Blight/Spot detected. Remove fallen debris and apply targeted fungicide.'
        else:
            advice = 'Consult a local agronomist. Immediate fungicide application may be necessary.'


        return jsonify({
            'result': pred_class.replace('___', ': '),
            'advice': advice,
            'confidence': round(confidence * 100, 1)
        })

    except Exception as e:
        return jsonify({'result': f'Error during model prediction: {e}', 'advice': 'Model prediction failed.'})

if __name__ == '__main__':
    app.run(debug=True)