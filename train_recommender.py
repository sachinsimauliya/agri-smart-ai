import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Define file
DATA_FILE = 'crop_recommendation.csv'

if os.path.exists(DATA_FILE):
    print(f"Loading {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)

    # Inputs: N, P, K, Temp, Humidity, pH, Rainfall
    # Output: Label (The Crop Name)
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Crop Recommender (Classification)...")
    # We use Classifier because we are predicting a Category (Name), not a Number
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, 'crop_recommender.pkl')
    print(f"✅ Success! Accuracy: {model.score(X_test, y_test):.2f}")
else:
    print(f"❌ Error: {DATA_FILE} not found. Please download it first.")