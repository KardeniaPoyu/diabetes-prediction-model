# Diabetes Prediction Model with Deep Learning

This project implements a deep learning model to predict diabetes using the Pima Indians Diabetes Dataset. It demonstrates skills in data preprocessing, neural network implementation with PyTorch, and model evaluation, suitable for AI and data science applications in graduate school admissions.

## Project Overview
- **Goal**: Predict whether a patient has diabetes based on medical features (e.g., Glucose, BMI).
- **Dataset**: Pima Indians Diabetes Dataset (768 samples, 8 features, binary classification).
- **Model**: 3-layer Multi-Layer Perceptron (MLP) using PyTorch.
- **Features**: Data cleaning, standardization, model training, evaluation with metrics (accuracy, ROC AUC), and visualizations (loss curve, confusion matrix, Precision-Recall curve).
- **Improvements**: Uses deep learning for non-linear pattern recognition, compared to previous logistic regression or random forest models.

## Installation
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd diabetes-prediction-model
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure Python 3.8+ and internet access (for dataset download).

## Usage
1. Run the main script:
   ```bash
   python diabetes_prediction_dl.py
   ```
2. Outputs:
   - Console: Accuracy (~0.80), ROC AUC (~0.78), classification report.
   - Images: `loss_curve.png`, `confusion_matrix_dl.png`, `pr_curve_dl.png` (saved locally).
   - Model: Saved as `diabetes_nn_model.pth`.

## Results
- **Accuracy**: ~0.80 on test set.
- **ROC AUC**: ~0.78, indicating good discrimination.
- **Classification Report**:
  ```
              precision    recall  f1-score   support
         0.0       0.83      0.87      0.85        99
         1.0       0.74      0.67      0.70        55
    accuracy                           0.80       154
   macro avg       0.79      0.77      0.78       154
  ```
- **Visualizations**:
  - **Training Loss Curve**: Shows model convergence over 100 epochs.
    ![Training Loss Curve](loss_curve.png)
  - **Confusion Matrix**: Highlights prediction performance (True Positives ~37, True Negatives ~86).
    ![Confusion Matrix](confusion_matrix_dl.png)
  - **Precision-Recall Curve**: Area ~0.75, suitable for imbalanced data.
    ![Precision-Recall Curve](pr_curve_dl.png)

## Dataset
- Source: [Pima Indians Diabetes Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
- Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.
- Target: Outcome (0 = No Diabetes, 1 = Diabetes).

## Future Improvements
- Add dropout layers to prevent overfitting.
- Implement k-fold cross-validation for robust evaluation.
- Extend to CNN if imaging data is integrated.
- Deploy as a web app using Streamlit.

## License
MIT License. Free to use and modify with attribution.

## Contact
For questions, contact me at yirongyiburong@gmail.com.
