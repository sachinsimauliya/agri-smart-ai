import requests

url = 'http://127.0.0.1:5000/predict'

# IMPORTANT: Change "Rice" to a name that appeared in your train_model.py output
# Example: If training printed 'rice' (lowercase), change it here.
data = {
    "N": 100,
    "P": 40,
    "K": 30,
    "Temp": 25.5,
    "Humidity": 75,
    "pH": 6.2,
    "Rainfall": 180,
    "Crop": "Rice"  # <--- CHECK THIS NAME
}

try:
    response = requests.post(url, json=data)
    print("\nStatus Code:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print("Error:", e)