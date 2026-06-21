from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

model_data = pickle.load(open('diabetes_model.sav', 'rb'))
classifier = model_data['model']
scaler = model_data['scaler']

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/style.css')
def serve_css():
    return send_file('style.css')

@app.route('/script.js')
def serve_js():
    return send_file('script.js')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        input_data = [
            float(data['Pregnancies']),
            float(data['Glucose']),
            float(data['BloodPressure']),
            float(data['SkinThickness']),
            float(data['Insulin']),
            float(data['BMI']),
            float(data['DiabetesPedigreeFunction']),
            float(data['Age'])
        ]
        
        input_array = np.asarray(input_data).reshape(1, -1)
        standardized_input = scaler.transform(input_array)
        prediction = classifier.predict(standardized_input)
        
        result = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'
        return jsonify({'prediction': result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
