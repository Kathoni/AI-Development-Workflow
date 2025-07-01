# AI Development Workflow Assignment

## Project Overview

This repository contains the implementation and documentation for an AI system designed to predict hospital patient readmission risk within 30 days of discharge. The project demonstrates a complete AI development workflow, from problem definition to deployment, with a focus on healthcare applications.

**Course:** AI for Software Engineering  
**Duration:** 7 days  
**Total Points:** 100

## Problem Statement

**Objective:** Develop an AI model to predict the likelihood that a patient will be readmitted to the hospital within 30 days of discharge.

**Key Goals:**
- Accurately identify patients at high risk of readmission
- Enable proactive interventions to reduce readmission rates
- Support hospital resource planning and improve patient outcomes

## Project Structure

```
AI-Development-Workflow/
├── README.md                           # This file
├── AI Development Workflow Assignment.pdf  # Main assignment document
├── requirements.txt                    # Python dependencies
├── readmission_prediction.py          # Main implementation script
├── readmission_prediction.ipynb       # Jupyter notebook (report version)
└── docs/                              # Additional documentation
    └── workflow_diagram.png           # AI workflow visualization
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd AI-Development-Workflow
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

### Required Packages
- numpy
- pandas
- scikit-learn
- flask
- matplotlib (for visualizations)
- seaborn (for enhanced plots)

## Usage

### Running the Main Script
```bash
python readmission_prediction.py
```

This will:
1. Generate mock hospital data
2. Preprocess the data (handle missing values, encode categorical variables, scale features)
3. Train a Random Forest classifier
4. Evaluate the model performance
5. Start a Flask API server for making predictions

### Using the API
Once the Flask server is running, you can make predictions by sending POST requests to `/predict`:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
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
```

### Jupyter Notebook
Open `readmission_prediction.ipynb` in Jupyter Notebook or JupyterLab for an interactive exploration of the workflow.

## Implementation Details

### Data Preprocessing
- **Missing Value Handling:** Median imputation for numerical features
- **Categorical Encoding:** Gender encoded as binary (M=0, F=1)
- **Feature Scaling:** StandardScaler applied to numerical features
- **Feature Engineering:** Includes derived features like comorbidity count

### Model Selection
**Random Forest Classifier** was chosen for the following reasons:
- Handles tabular data effectively
- Robust to outliers and noise
- Provides feature importance rankings
- Good interpretability for healthcare applications
- No assumptions about data distribution

### Evaluation Metrics
- **Confusion Matrix:** Shows true positives, false positives, true negatives, false negatives
- **Precision:** Ratio of correctly predicted positive observations to total predicted positives
- **Recall:** Ratio of correctly predicted positive observations to all actual positives

## Ethical Considerations

### Privacy and Security
- All patient data should be anonymized and securely stored
- Compliance with healthcare regulations (e.g., HIPAA) is essential
- Access controls and encryption should be implemented

### Bias and Fairness
- Training data should be representative of the patient population
- Regular bias audits should be conducted
- Model performance should be evaluated across different demographic groups

## Deployment Considerations

### Technical Challenges
- **Scalability:** Model should handle real-time predictions efficiently
- **Integration:** Seamless integration with existing hospital systems
- **Monitoring:** Continuous monitoring for concept drift and model performance

### Healthcare Compliance
- **HIPAA Compliance:** All data handling must meet HIPAA requirements
- **Audit Trails:** Complete documentation of all processes and decisions
- **Explainability:** Model decisions must be interpretable by healthcare professionals

## Future Improvements

1. **Enhanced Data Collection:** Include more comprehensive patient data
2. **Feature Engineering:** Collaborate with domain experts for better feature design
3. **Model Validation:** Conduct more extensive cross-validation and testing
4. **Real-time Monitoring:** Implement automated monitoring for model performance
5. **A/B Testing:** Compare model performance against existing clinical protocols

## Contributing

This is an academic assignment. For questions or suggestions, please contact the project team.

## References

1. CRISP-DM: A Standard Process Model for Data Mining
2. HIPAA Journal. (2023). What is HIPAA Compliance?
3. Chouldechova, A., & Roth, A. (2020). A snapshot of the frontiers of fairness in machine learning
4. Scikit-learn documentation: https://scikit-learn.org/stable/
5. Flask documentation: https://flask.palletsprojects.com/

## License

This project is created for educational purposes as part of an academic assignment.

## Contact

For questions about this assignment, please refer to the course instructor or teaching assistant.

---

**Note:** This implementation uses synthetic data for demonstration purposes. In a real healthcare setting, all data handling must comply with relevant regulations and ethical guidelines.

