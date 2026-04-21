import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Set up logging for a clean, professional output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_clean_data(file_path):
    """
    Load the spam dataset, handle missing values, and create initial features.
    """
    logging.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path, encoding='latin-1')
    
    # Drop columns that are mostly NaNs
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True, errors='ignore')
    
    # Rename columns explicitly
    df.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)
    
    # Encode target variable
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Create an additional feature for scaling demonstration
    df['text_length'] = df['text'].apply(len)
    
    # Check for missing values
    if df.isnull().sum().any():
        df.dropna(inplace=True)
        
    return df

def perform_eda(df):
    """
    Perform Exploratory Data Analysis and save plots.
    """
    logging.info("Generating EDA visualizations...")
    
    sns.set_theme(style="whitegrid")
    
    # 1. Class distribution plot
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x='label', palette='viridis')
    plt.title('Distribution of Ham vs. Spam (0=Ham, 1=Spam)')
    ax.bar_label(ax.containers[0])
    plt.tight_layout()
    plt.savefig('eda_class_distribution.png', dpi=300)
    plt.close()
    
    # 2. Text length distribution plot
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='text_length', hue='label', bins=50, kde=True, palette='viridis')
    plt.title('Text Length Distribution by Class')
    plt.tight_layout()
    plt.savefig('eda_text_length_distribution.png', dpi=300)
    plt.close()

def build_preprocessing_pipeline():
    """
    Build a comprehensive pipeline including text vectorization, scaling, 
    and feature selection.
    """
    # Text vectorization using TF-IDF (implicit normalization)
    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(stop_words='english', min_df=2, max_df=0.9))
    ])
    
    # Numerical scaling
    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Column transformer to apply appropriate transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, 'text'),
            ('num', num_transformer, ['text_length'])
        ]
    )
    
    return preprocessor

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using appropriate metrics and save confusion matrix.
    """
    logging.info("Evaluating optimal model on the test set...")
    
    y_pred = model.predict(X_test)
    
    # Print the classification report
    report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
    print("\n--- Classification Report ---\n")
    print(report)
    print("-----------------------------\n")
    
    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('evaluation_confusion_matrix.png', dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Spam Classification Pipeline")
    parser.add_argument('--data', type=str, default='spam.csv', help='Path to the dataset')
    args = parser.parse_args()

    # 1. Provide custom preprocessing and EDA
    df = load_and_clean_data(args.data)
    perform_eda(df)
    
    X = df[['text', 'text_length']]
    y = df['label']
    
    # Stratified split to ensure class balance holds
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. Build Pipeline
    # Incorporates preprocessor, feature selection, and the classifier.
    preprocessor = build_preprocessing_pipeline()
    
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])
    
    # 3. Hyperparameter Tuning
    logging.info("Starting hyperparameter tuning with GridSearchCV...")
    param_grid = {
        'feature_selection__k': [500, 1000, 1500],
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 10, 20]
    }
    
    grid_search = GridSearchCV(
        full_pipeline, 
        param_grid=param_grid, 
        cv=3, 
        scoring='f1_macro', 
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    logging.info(f"Best Hyperparameters: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_
    
    # 4. Evaluation phase
    evaluate_model(best_model, X_test, y_test)
    
    # 5. Serialization
    model_filepath = 'spam_classifier_model.pkl'
    logging.info(f"Saving the finalized pipeline to {model_filepath}...")
    with open(model_filepath, 'wb') as f:
        pickle.dump(best_model, f)
        
    logging.info("Pipeline executed successfully.")

if __name__ == '__main__':
    main()
