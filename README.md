🌱 AgriSmart AI: Precision Agriculture DashboardThis project is a web-based, AI-powered dashboard designed for precision agriculture. It provides farmers and agronomists with predictive models and diagnostic tools to optimize crop yield, resource usage, and overall farm health. The application is built using Python (Flask) for the backend and TensorFlow/Scikit-learn for the machine learning models.✨ Key FeaturesFeatureTechnologyFunctionYield AIRandom Forest / LSTMPredicts crop yield (kg/ha) based on environmental conditions (Temp, Humidity, pH, Rainfall).Crop RecommenderRandom Forest ClassifierRecommends the optimal crop type based on current NPK levels and environmental metrics.Diagnostics AICNN (VGG16 Transfer Learning)Identifies plant diseases from leaf images and provides treatment advice.Fertilizer AdvisorBusiness LogicAdvises on NPK dosage required to meet optimal levels for a specific crop.Storage AdvisorDynamic Decay ModelPredicts the shelf-life of produce based on storage temperature and humidity.Smart IrrigationLogic-based CWR FactorCalculates immediate water needs based on ambient conditions and crop water requirements.Bilingual InterfaceFrontend JavaScript / Backend LogicSupports interface and output translation between English (EN) and Hindi (HI).🛠️ Technology StackBackend Framework: Python 3.10+ (Flask)Machine Learning: TensorFlow, Keras, scikit-learnData Handling: NumPy, Pandas, JoblibImage Processing: OpenCV (cv2)Frontend: HTML, CSS (Custom Styling), JavaScript🚀 Setup and InstallationFollow these steps to get the AgriSmart AI dashboard running locally.1. Clone the RepositoryBashgit clone https://github.com/YOUR-USERNAME/agri-smart-ai.git
cd agri-smart-ai
2. Create and Activate Virtual EnvironmentIt's highly recommended to use a virtual environment.Bash# Create the environment
python -m venv venv

# Activate the environment (Windows)
.\venv\Scripts\activate
# Activate the environment (macOS/Linux)
source venv/bin/activate
3. Install DependenciesYou need to install all required Python libraries.Bashpip install -r requirements.txt
# NOTE: If you haven't created a requirements.txt file, run:
# pip install Flask scikit-learn pandas numpy tensorflow keras joblib opencv-python Pillow
4. Run Model Training (Generate .h5 files)You must run the training script to generate the model structure files (.pkl and .h5) needed by the application.Bashpython train_model.py
(Note: If the train_model.py script requires external data, ensure that data is placed in the expected location before running this command.)5. Run the Flask ApplicationStart the development server:Bashpython app.py
The application will now be running. Access it in your web browser at:http://127.0.0.1:5000/💻 Project Structureagri-smart-ai/
├── app.py                      # Flask Application, API Endpoints, and Translation Logic
├── train_model.py              # Script to build and save ML models (.pkl, .h5)
├── templates/
│   └── index.html              # Main dashboard frontend (HTML, CSS, JS)
├── venv/                       # Virtual Environment (Ignored by Git)
├── *.pkl                       # Saved ML Models (Yield, Encoder, Recommender, Scaler)
└── *.h5                        # Saved Deep Learning Models (LSTM, Diagnostics CNN)
🤝 ContributingIf you'd like to contribute, please fork the repository and use a feature branch. Pull requests are welcome!📜 License(You should choose and mention a license here, e.g., MIT, Apache 2.0)
