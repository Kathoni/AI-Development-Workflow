"""
Hospital Readmission Risk Prediction System
AI Development Workflow Assignment - Part 2: Case Study Application

This script demonstrates a complete AI workflow for predicting patient readmission risk
within 30 days of discharge, including data preprocessing, model training, evaluation,
and a simple API for deployment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from flask import Flask, request, jsonify
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class HospitalReadmissionPredictor:
    """
    A comprehensive class for predicting hospital readmission risk.
    Implements the complete AI development workflow.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoder = LabelEncoder()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.feature_names = None
        
    def generate_mock_data(self, n_samples=1000):
        """
        Generate synthetic hospital data for demonstration purposes.
        In a real scenario, this would be replaced with actual EHR data.
        """
        print("Generating mock hospital data...")
        
        # Generate realistic patient data
        data = pd.DataFrame({
            'patient_id': range(1, n_samples + 1),
            'age': np.random.normal(65, 15, n_samples).astype(int),
            'gender': np.random.choice(['M', 'F'], n_samples, p=[0.45, 0.55]),
            'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], n_samples),
            'insurance_type': np.random.choice(['Private', 'Medicare', 'Medicaid', 'Uninsured'], n_samples),
            'num_prev_admissions': np.random.poisson(2, n_samples),
            'comorbidity_count': np.random.poisson(3, n_samples),
            'length_of_stay': np.random.exponential(5, n_samples).astype(int) + 1,
            'discharge_destination': np.random.choice(['Home', 'SNF', 'Rehab', 'Other'], n_samples),
            'admission_type': np.random.choice(['Emergency', 'Elective', 'Urgent'], n_samples),
            'diabetes': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'hypertension': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
            'heart_failure': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'chronic_kidney_disease': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'lab_glucose': np.random.normal(120, 30, n_samples),
            'lab_creatinine': np.random.normal(1.2, 0.5, n_samples),
            'lab_hemoglobin': np.random.normal(13, 2, n_samples)
        })
        
        # Ensure age is within reasonable bounds
        data['age'] = np.clip(data['age'], 18, 100)
        
        # Generate target variable (readmission within 30 days)
        # Create a more realistic model where readmission probability depends on features
        readmission_prob = (
            0.1 +  # Base probability
            0.02 * (data['age'] - 65) / 15 +  # Age effect
            0.05 * data['comorbidity_count'] / 3 +  # Comorbidity effect
            0.03 * data['num_prev_admissions'] / 2 +  # Previous admissions effect
            0.02 * (data['length_of_stay'] - 5) / 5 +  # Length of stay effect
            0.05 * data['diabetes'] +  # Diabetes effect
            0.03 * data['heart_failure'] +  # Heart failure effect
            0.04 * data['chronic_kidney_disease']  # CKD effect
        )
        
        # Add some randomness and ensure probabilities are between 0 and 1
        readmission_prob = np.clip(readmission_prob + np.random.normal(0, 0.05, n_samples), 0, 1)
        data['readmitted_30d'] = np.random.binomial(1, readmission_prob)
        
        # Introduce some missing values to simulate real-world data
        for col in ['lab_glucose', 'lab_creatinine', 'lab_hemoglobin']:
            missing_mask = np.random.choice([True, False], n_samples, p=[0.1, 0.9])
            data.loc[missing_mask, col] = np.nan
            
        print(f"Generated {n_samples} patient records")
        print(f"Readmission rate: {data['readmitted_30d'].mean():.2%}")
        
        return data
    
    def preprocess_data(self, data):
        """
        Comprehensive data preprocessing pipeline.
        Includes handling missing values, encoding categorical variables, and feature scaling.
        """
        print("Preprocessing data...")
        
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # 1. Handle missing values
        print("  - Handling missing values...")
        numerical_cols = ['lab_glucose', 'lab_creatinine', 'lab_hemoglobin']
        df[numerical_cols] = self.imputer.fit_transform(df[numerical_cols])
        
        # 2. Feature engineering
        print("  - Creating engineered features...")
        df['age_group'] = pd.cut(df['age'], bins=[0, 50, 65, 80, 100], labels=['Young', 'Middle', 'Senior', 'Elderly'])
        df['risk_score'] = (
            df['comorbidity_count'] * 0.3 +
            df['num_prev_admissions'] * 0.2 +
            df['diabetes'] * 0.15 +
            df['heart_failure'] * 0.15 +
            df['chronic_kidney_disease'] * 0.2
        )
        df['length_of_stay_category'] = pd.cut(
            df['length_of_stay'], 
            bins=[0, 3, 7, 14, 100], 
            labels=['Short', 'Medium', 'Long', 'Extended']
        )
        
        # 3. Encode categorical variables
        print("  - Encoding categorical variables...")
        categorical_cols = ['gender', 'race', 'insurance_type', 'discharge_destination', 
                          'admission_type', 'age_group', 'length_of_stay_category']
        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = self.label_encoder.fit_transform(df[col].astype(str))
        
        # 4. Select features for modeling
        feature_cols = [
            'age', 'gender', 'num_prev_admissions', 'comorbidity_count', 'length_of_stay',
            'diabetes', 'hypertension', 'heart_failure', 'chronic_kidney_disease',
            'lab_glucose', 'lab_creatinine', 'lab_hemoglobin', 'risk_score'
        ]
        
        # Add encoded categorical features
        for col in ['race', 'insurance_type', 'discharge_destination', 'admission_type']:
            if col in df.columns:
                feature_cols.append(col)
        
        self.feature_names = feature_cols
        X = df[feature_cols]
        y = df['readmitted_30d']
        
        # 5. Scale numerical features
        print("  - Scaling numerical features...")
        numerical_features = ['age', 'num_prev_admissions', 'comorbidity_count', 'length_of_stay',
                            'lab_glucose', 'lab_creatinine', 'lab_hemoglobin', 'risk_score']
        X[numerical_features] = self.scaler.fit_transform(X[numerical_features])
        
        print(f"  - Final feature set: {len(feature_cols)} features")
        return X, y
    
    def train_model(self, X, y):
        """
        Train the Random Forest model with cross-validation.
        """
        print("Training Random Forest model...")
        
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='f1')
        print(f"  - Cross-validation F1 scores: {cv_scores}")
        print(f"  - Mean CV F1 score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Train the final model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        return X_test, y_test, y_pred, y_pred_proba
    
    def evaluate_model(self, y_test, y_pred, y_pred_proba):
        """
        Comprehensive model evaluation with multiple metrics.
        """
        print("\nModel Evaluation:")
        print("=" * 50)
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")
        
        # 2. Classification Report
        print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        
        # 3. Key Metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
        
        # 4. Feature Importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 5 Most Important Features:")
        print(feature_importance.head())
        
        return cm, precision, recall, f1, feature_importance
    
    def create_visualizations(self, y_test, y_pred, y_pred_proba, feature_importance):
        """
        Create visualizations for model evaluation.
        """
        print("\nCreating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix Heatmap
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # 2. Feature Importance
        top_features = feature_importance.head(10)
        sns.barplot(data=top_features, x='importance', y='feature', ax=axes[0,1])
        axes[0,1].set_title('Top 10 Feature Importance')
        
        # 3. Prediction Probability Distribution
        axes[1,0].hist(y_pred_proba[y_test == 0], alpha=0.7, label='No Readmission', bins=20)
        axes[1,0].hist(y_pred_proba[y_test == 1], alpha=0.7, label='Readmission', bins=20)
        axes[1,0].set_title('Prediction Probability Distribution')
        axes[1,0].set_xlabel('Predicted Probability')
        axes[1,0].set_ylabel('Count')
        axes[1,0].legend()
        
        # 4. ROC Curve (simplified)
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        axes[1,1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1,1].set_xlim([0.0, 1.0])
        axes[1,1].set_ylim([0.0, 1.05])
        axes[1,1].set_xlabel('False Positive Rate')
        axes[1,1].set_ylabel('True Positive Rate')
        axes[1,1].set_title('ROC Curve')
        axes[1,1].legend(loc="lower right")
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as 'model_evaluation.png'")

def create_flask_api(predictor):
    """
    Create a Flask API for model deployment.
    """
    app = Flask(__name__)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({'status': 'healthy', 'message': 'Hospital Readmission Predictor is running'})
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """
        Predict readmission risk for a patient.
        
        Expected JSON input:
        {
            "age": 65,
            "gender": "M",
            "num_prev_admissions": 2,
            "comorbidity_count": 3,
            "length_of_stay": 7,
            "diabetes": 1,
            "hypertension": 1,
            "heart_failure": 0,
            "chronic_kidney_disease": 0,
            "lab_glucose": 120,
            "lab_creatinine": 1.2,
            "lab_hemoglobin": 13
        }
        """
        try:
            data = request.get_json()
            
            # Validate required fields
            required_fields = ['age', 'gender', 'num_prev_admissions', 'comorbidity_count', 
                             'length_of_stay', 'diabetes', 'hypertension', 'heart_failure', 
                             'chronic_kidney_disease', 'lab_glucose', 'lab_creatinine', 'lab_hemoglobin']
            
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'Missing required field: {field}'}), 400
            
            # Create feature vector
            features = pd.DataFrame([data])
            
            # Calculate risk score
            features['risk_score'] = (
                features['comorbidity_count'] * 0.3 +
                features['num_prev_admissions'] * 0.2 +
                features['diabetes'] * 0.15 +
                features['heart_failure'] * 0.15 +
                features['chronic_kidney_disease'] * 0.2
            )
            
            # Encode gender
            features['gender'] = features['gender'].map({'M': 0, 'F': 1})
            
            # Scale numerical features
            numerical_features = ['age', 'num_prev_admissions', 'comorbidity_count', 'length_of_stay',
                                'lab_glucose', 'lab_creatinine', 'lab_hemoglobin', 'risk_score']
            features[numerical_features] = predictor.scaler.transform(features[numerical_features])
            
            # Make prediction
            prediction = predictor.model.predict(features)[0]
            probability = predictor.model.predict_proba(features)[0, 1]
            
            # Determine risk level
            if probability < 0.3:
                risk_level = "Low"
            elif probability < 0.6:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            return jsonify({
                'readmission_risk': int(prediction),
                'probability': float(probability),
                'risk_level': risk_level,
                'message': f'Patient has {risk_level.lower()} risk of readmission'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/model_info', methods=['GET'])
    def model_info():
        """Get information about the trained model."""
        return jsonify({
            'model_type': 'Random Forest Classifier',
            'n_estimators': 100,
            'max_depth': 10,
            'features': predictor.feature_names,
            'feature_count': len(predictor.feature_names)
        })
    
    return app

def main():
    """
    Main function to run the complete AI workflow.
    """
    print("Hospital Readmission Risk Prediction System")
    print("=" * 50)
    
    # Initialize the predictor
    predictor = HospitalReadmissionPredictor()
    
    # 1. Generate data
    data = predictor.generate_mock_data(n_samples=1000)
    
    # 2. Preprocess data
    X, y = predictor.preprocess_data(data)
    
    # 3. Train model
    X_test, y_test, y_pred, y_pred_proba = predictor.train_model(X, y)
    
    # 4. Evaluate model
    cm, precision, recall, f1, feature_importance = predictor.evaluate_model(y_test, y_pred, y_pred_proba)
    
    # 5. Create visualizations
    predictor.create_visualizations(y_test, y_pred, y_pred_proba, feature_importance)
    
    # 6. Start Flask API
    print("\nStarting Flask API...")
    print("API endpoints:")
    print("  - GET  /health      : Health check")
    print("  - POST /predict     : Make predictions")
    print("  - GET  /model_info  : Model information")
    print("\nExample prediction request:")
    print("""
    curl -X POST http://localhost:5000/predict \\
      -H "Content-Type: application/json" \\
      -d '{
        "age": 65,
        "gender": "M",
        "num_prev_admissions": 2,
        "comorbidity_count": 3,
        "length_of_stay": 7,
        "diabetes": 1,
        "hypertension": 1,
        "heart_failure": 0,
        "chronic_kidney_disease": 0,
        "lab_glucose": 120,
        "lab_creatinine": 1.2,
        "lab_hemoglobin": 13
      }'
    """)
    
    app = create_flask_api(predictor)
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main() 