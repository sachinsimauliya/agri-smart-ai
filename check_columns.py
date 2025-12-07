import pandas as pd

# Load the csv
try:
    df = pd.read_csv('crop_yield_data.csv')
    print("\nSUCCESS! File found. Here are your column headers:\n")
    print(list(df.columns))
    print("\nCopy these names exactly into your train_model.py mapping.\n")
except FileNotFoundError:
    print("Error: crop_yield_data.csv still not found.")