# Diabetes Prediction Web Application

An interactive web-based dashboard that predicts the probability of a user having diabetes based on various clinical parameters. The application uses a machine learning model trained on the PIMA Diabetes dataset, exposes a Flask REST API, and features a responsive, user-friendly frontend.

🔗 **Live Demo**: [Diabetes Predictor Website](https://jeevanjacob1.github.io/ACM_WORKS/diabetes_prediction/)

## 📸 Screenshots

Here are some screenshots of the application:

<img width="800" alt="Dashboard Screenshot 1" src="https://github.com/user-attachments/assets/11587342-e7f1-427c-b10c-365845489d90" />
<br/>
<img width="800" alt="Dashboard Screenshot 2" src="https://github.com/user-attachments/assets/8a0d56c5-4ec7-4d49-89e3-4c10bff00b0a" />
<br/>
<img width="800" alt="Dashboard Screenshot 3" src="https://github.com/user-attachments/assets/d3d98b24-eb7e-4657-a5cd-bf3c5cf80d5b" />

---

## 🚀 Key Features
- **SVM Classifier**: Uses a Support Vector Machine classifier with a linear kernel.
- **Flask REST API**: Handles backend predictions seamlessly, processing user inputs and outputting results in real-time.
- **Responsive Dashboard**: Beautifully designed UI for users to easily enter their clinical metrics and instantly receive a prediction.
- **Robust Preprocessing**: Standardizes input health metrics using `StandardScaler` to ensure prediction accuracy.

---

## 📁 Repository Directory Structure

Below is the directory structure for this project:

```
diabetes_prediction/
├── Project_3_Diabetes_Prediction.ipynb  # Jupyter Notebook for Exploratory Data Analysis & training
├── app.py                              # Flask application (serves static UI & exposes /predict REST API)
├── diabetes.csv                        # PIMA Diabetes Dataset (768 patient records)
├── diabetes_model.sav                  # Saved serialized model and scaler dict (pickle format)
├── index.html                          # Frontend User Interface dashboard
├── requirements.txt                    # List of Python dependencies
├── script.js                           # Frontend logic for validation & API request handling
├── style.css                           # Modern stylesheet with dark-themed visual presentation
└── train_model.py                      # Python training script to retrain the SVM classifier
```

### File Breakdown
- **[app.py](./app.py)**: The web server script. Serves the web interface (`index.html`, `style.css`, `script.js`) and handles `POST` requests to `/predict`. It deserializes the scaler and classifier to process user input and return the result.
- **[train_model.py](./train_model.py)**: Loads `diabetes.csv`, splits the data into training/testing sets (80/20 split), standardizes the metrics, trains the SVM classifier, prints the model accuracy, and saves it to `diabetes_model.sav`.
- **[Project_3_Diabetes_Prediction.ipynb](./Project_3_Diabetes_Prediction.ipynb)**: An notebook illustrating the exploratory data analysis, data distribution, and step-by-step logic.
- **[diabetes.csv](./diabetes.csv)**: Dataset from the National Institute of Diabetes and Digestive and Kidney Diseases containing 8 health variables (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age).
- **[diabetes_model.sav](./diabetes_model.sav)**: Binary file generated using `pickle` containing the trained SVM model and the fitted `StandardScaler`.
- **[index.html](./index.html)**, **[style.css](./style.css)**, & **[script.js](./script.js)**: The user interface files. It contains form validation and communicates asynchronously via Fetch API to the backend to retrieve predictions without refreshing the page.

---

## 📊 Model & Dataset Details

The machine learning model is trained on the PIMA Diabetes dataset containing records of females of Pima Indian heritage.

### Health Parameters (Model Features)
1. **Pregnancies**: Number of times pregnant
2. **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3. **Blood Pressure**: Diastolic blood pressure (mm Hg)
4. **Skin Thickness**: Triceps skin fold thickness (mm)
5. **Insulin**: 2-Hour serum insulin (mu U/ml)
6. **BMI**: Body mass index (weight in kg/(height in m)^2)
7. **Diabetes Pedigree Function**: Diabetes pedigree function (genetic score)
8. **Age**: Age in years

### Model Performance
- **Algorithm**: Support Vector Classifier (SVC) with a linear kernel
- **Dataset Size**: 768 samples (8 features, 1 target label)
- **Train/Test Split**: 80% Train (614 samples) / 20% Test (154 samples)
- **Training Accuracy**: `78.66%`
- **Testing Accuracy**: `77.27%`

---

## 🛠️ Setup & Running Instructions

Follow these steps to run the application locally on your machine:

### 1. Clone the repository & navigate to the project directory
```bash
git clone https://github.com/jeevanjacob1/ACM_WORKS.git
cd ACM_WORKS/diabetes_prediction
```

### 2. Create and activate a Virtual Environment (Optional but Recommended)
**On Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```
**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install required dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Retrain the Model
If you want to retrain the SVM classifier and update the saved `.sav` file, run:
```bash
python train_model.py
```

### 5. Start the Web Server
Launch the Flask development server:
```bash
python app.py
```
By default, the server will start on `http://127.0.0.1:5000/`.

### 6. Access the Web App
Open your favorite web browser and navigate to:
```
http://127.0.0.1:5000/
```
Enter values for all 8 health parameters, and click on the **Predict** button to view the model's output prediction!
