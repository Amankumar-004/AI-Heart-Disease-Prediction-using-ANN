# Heart Disease Prediction using Artificial Neural Network (ANN)

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

## Dependencies

This project likely relies on the following Python libraries:

* pandas: For data manipulation and analysis.
* scikit-learn (sklearn): For machine learning algorithms and tools.

You can install these dependencies using pip if you haven't already:

```bash
pip install pandas scikit-learn
# Heart Disease Prediction Using ANN

This project builds an Artificial Neural Network (ANN) to predict the likelihood of heart disease based on health metrics such as age, cholesterol, and blood pressure.

---

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
