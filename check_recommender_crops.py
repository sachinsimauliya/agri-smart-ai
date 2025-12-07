import joblib
import os

def check_recommender_crops():
    """Loads the recommender model to see which crops it knows."""
    REC_MODEL_FILE = 'recommender_model.pkl'

    if not os.path.exists(REC_MODEL_FILE):
        print("❌ Error: recommender_model.pkl not found.")
        print("Please ensure you have run the training script.")
        return

    try:
        # Load the Random Forest Classifier
        recommender_model = joblib.load(REC_MODEL_FILE)
        
        # The classes_ attribute holds the unique labels seen during training
        # For the recommender (a classifier), the output classes are the crop names
        crops = recommender_model.classes_
        
        print(f"\n✅ Crop Recommender knows {len(crops)} crops.")
        print("-" * 30)
        
        # Format the list for easy reading
        for i, crop in enumerate(crops):
            print(f"  {i+1}. {crop}")
            
    except Exception as e:
        print(f"❌ An error occurred while loading the model: {e}")

if __name__ == "__main__":
    check_recommender_crops()