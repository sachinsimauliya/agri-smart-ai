import pandas as pd

# Load the csv
try:
    df = pd.read_csv('crop_yield_data.csv')
    
    # Check what the 'Crop' column actually contains
    # We use the column name 'Crop' because you renamed it in your training script, 
    # but since we are reading raw CSV here, use the original header name.
    # Based on your previous screenshots, the header in CSV is 'Crop' or 'Crop_Name'.
    
    # Let's try to find the crop column automatically
    possible_names = ['Crop', 'Crop_Name', 'label', 'Label_Crop']
    crop_col = next((col for col in df.columns if col in possible_names), None)

    if crop_col:
        print("\n✅ HERE ARE THE EXACT CROP NAMES IN YOUR DATABASE:")
        print("----------------------------------------------------")
        unique_crops = sorted(df[crop_col].unique())
        print(unique_crops)
        print("----------------------------------------------------")
        print("👉 Update your index.html <option> values to match these EXACTLY.")
    else:
        print("Could not find a 'Crop' column. Here are the columns found:", list(df.columns))

except FileNotFoundError:
    print("Error: crop_yield_data.csv not found.")