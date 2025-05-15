# AI Heart Disease Prediction using Artificial Neural Network (ANN)

## Overview
This project implements an Artificial Neural Network (ANN) to predict the likelihood of a person having heart disease based on a dataset of health-related attributes. Built with Python, Keras, and TensorFlow, the model provides a binary classification output for heart disease presence.

## Features
- **ANN Model**: Multi-layer perceptron for binary classification.
- **Data Preprocessing**: Handles dataset splitting and feature scaling.
- **Performance Evaluation**: Uses confusion matrix and accuracy score.
- **Clear Code Structure**: Well-commented Python script for easy understanding.

## Requirements
- Python 3.x
- TensorFlow (>= 2.x)
- Keras (>= 2.x)
- pandas (>= 1.x)
- NumPy (>= 1.x)
- scikit-learn (>= 1.x)

## Project Structure

The project directory should contain the following files:

* `AI heart_disease_prediction.py`: The main Python script containing the data loading, preprocessing, model training, and prediction logic.
* `Heart disease dataset.csv`: The dataset used for training and prediction.
* `README.md`: This file, providing information about the project.

## 🧠 Problem Statement

Develop a binary classification ANN model that can predict the presence of heart disease using 13 input features derived from medical data.

---

## ✅ Proposed Solution

An Artificial Neural Network (ANN) model is used for binary classification with the following architecture:

- **Input Layer**: 13 neurons (representing 13 input features).
- **Hidden Layers**:
  - Layer 1: 8 neurons, ReLU activation
  - Layer 2: 14 neurons, ReLU activation
- **Output Layer**: 1 neuron, sigmoid activation (outputs probability of heart disease)

---

## ⚙️ Dataset Notes

- The dataset includes 13 health-related features such as age, sex, chest pain type, cholesterol level, fasting blood sugar, etc.
- This dataset is commonly used for heart disease prediction in machine learning research.

---
# Confusion Matrix Accuracy Calculator

## Overview

This simple Python script calculates the **accuracy** of a classification model using a confusion matrix.

### Accuracy Formula:

\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]

Where:

- `TP` = True Positives = `cm[1][1]`
- `TN` = True Negatives = `cm[0][0]`
- `FP` = False Positives = `cm[0][1]`
- `FN` = False Negatives = `cm[1][0]`

### Sample Code:

```python
accuracy = (cm[0][0] + cm[1][1]) / (cm[0][1] + cm[1][0] + cm[0][0] + cm[1][1])
print(accuracy * 100)
```
Example:
For a confusion matrix like:
cm = [[50, 10],
      [5, 85]]
Accuracy will be:
1.  **Numerator Calculation:**
    ```
    50 + 85 = 135
    ```

2.  **Denominator Calculation:**
    ```
    50 + 85 + 10 + 5 = 150
    ```

3.  **Fraction Formation:**
    The calculation can be represented as a fraction:
    ```
    135 / 150
    ```

4.  **Decimal Conversion:**
    Dividing the numerator by the denominator yields the decimal value:
    ```
    135 ÷ 150 = 0.90
    ```

5.  **Percentage Conversion:**
    Multiplying the decimal by 100 converts it to a percentage:
    ```
    0.90 * 100 = 90%
    ```

## Summary

The provided calculation demonstrates that the ratio of the sum of 50 and 85 to the sum of 50, 85, 10, and 5 is equal to 0.90, or 90%.

## Expected Result
We expect the accuracy to be approximately 85%, depending on the actual values in the confusion matrix.

## ⚠️ Notes & Warnings

- **Dataset Limitations**: Model performance may vary with larger or more diverse datasets.
- **Missing Values**: The columns `ca` and `thal` contain missing values. These are **not currently handled** in the code.
- **Validation**: Accuracy metrics are based on a **single train-test split**. Cross-validation is recommended for a more robust evaluation.
- **Medical Disclaimer**: This model is for **educational and research purposes only**. It should **not be used as a substitute for professional medical advice** or diagnosis.

---

## 🚀 Future Improvements

- Handle missing values in `ca` and `thal` columns.
- Implement k-fold cross-validation.
- Experiment with different model architectures and hyperparameters.
- Perform feature scaling and engineering for performance optimization.

---
