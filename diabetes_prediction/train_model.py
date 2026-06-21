import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
diabetes_dataset = pd.read_csv('diabetes.csv')
print("Dataset shape:", diabetes_dataset.shape)

X = diabetes_dataset.drop(columns='Outcome')
Y = diabetes_dataset['Outcome']

# Standardize features
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")

# Train linear SVM classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Evaluate model
train_prediction = classifier.predict(X_train)
training_accuracy = accuracy_score(Y_train, train_prediction)
print(f"Training Accuracy : {training_accuracy:.4f}")

test_prediction = classifier.predict(X_test)
testing_accuracy = accuracy_score(Y_test, test_prediction)
print(f"Testing Accuracy  : {testing_accuracy:.4f}")

# Save classifier and scaler
model_data = {
    'model': classifier,
    'scaler': scaler
}

filename = 'diabetes_model.sav'
pickle.dump(model_data, open(filename, 'wb'))
print(f"Model saved successfully to {filename}")
