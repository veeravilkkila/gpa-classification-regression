# gpa-classification-regression

A machine learning project that uses **Bayesian Neural Networks (BNN)** for predicting student GPAs. The project includes both **classification** (categorizing students based on GPA thresholds) and **regression** (predicting exact GPA values). 

## Data Source

The data used in this project is from the following source:

- **Rabie El Kharoua. (2024).** 📚 Students Performance Dataset 📚 [Data set]. Kaggle. [https://doi.org/10.34740/KAGGLE/DS/5195702](https://doi.org/10.34740/KAGGLE/DS/5195702)


## Notebooks in the Project

This repository contains three main Jupyter notebooks for preprocessing data, training models, and evaluating results.

### 1. **data_preprocessing.ipynb**
   This notebook handles the preprocessing of the raw data and prepares it for use in the BNN models.
   - **Data Loading**: Loads the raw dataset of students' information (e.g., study hours, absences, parents' education, and other relevant features).
   - **Feature Selection**: Chooses the features that will be used for training the models. This may involve selecting numeric features (e.g., study hours, previous grades), categorical features (e.g., parents' education, gender), and performing any necessary transformations.
   - **Data Transformation**: 
     - **Min-Max Scaling**: Scales continuous features to a range between 0 and 1 to ensure that they are all on a similar scale.
     - **One-Hot Encoding**: Encodes categorical features into binary vectors, allowing the neural network to handle them effectively.
   - **Saving Data**: The processed data, including feature selection and transformations, is saved as `.pkl` files, which will be used in subsequent steps to train and evaluate the models.

### 2. **Model_BNN_classifier.ipynb**
   This notebook focuses on training a **Bayesian Neural Network (BNN)** for the **classification** task. It performs the following steps:
   - **Data Loading**: Loads the preprocessed `.pkl` files.
   - **Data Splitting**: The data is split into training and testing sets (80/20) to evaluate the model's performance. 
   - **Model Construction**: Builds a BNN model suitable for classification.
   - **Training**: Trains the model on the classification task, predicting GPA categories (0-4).
   - **Evaluation**: Evaluates the model performance on the test data, reporting metrics like accuracy, confusion matrix, and classification report.

### 3. **Model_BNN_regression.ipynb**
   This notebook is for training a **Bayesian Neural Network (BNN)** for the **regression** task. It includes:
   - **Data Loading**: Loads the preprocessed `.pkl` data.
   - **Data Splitting**: The data is split into training and testing sets (80/20) to evaluate the model's performance. 
   - **Model Construction**: Builds a BNN model for regression.
   - **Training**: Trains the model to predict exact GPA values based on student features.
   - **Evaluation**: Evaluates the model's performance using regression metrics such as Mean Squared Error (MSE) and R² score, along with visualizations to better understand the model's predictions.

The project showcases the benefits of using a **Bayesian Neural Network** (BNN), which provides a measure of uncertainty in the predictions. This is particularly useful for regression and classification tasks where uncertainty quantification is important.

## Results

### Classification

The **Bayesian Neural Network (BNN)** model was trained to classify GPA categories (0-4). 

- **Confusion Matrix**: The confusion matrix indicates:
  - **Class 0**: 33.33% accuracy, with confusion across Classes 1, 3, and 4.
  - **Class 1**: High accuracy, but confusion with Classes 0, 2, and 4.
  - **Class 2**: Moderate performance, confusion mostly with Class 3.
  - **Class 3**: Strong accuracy, but confusion with Class 2.
  - **Class 4**: Highest accuracy, with minor confusion with Class 3.
  
  These misclassifications are due to the continuous nature of GPA categories that overlap in their features.

### Regression

The **Bayesian Neural Network (BNN)** model was trained to predict exact GPA values. 

- **Key Metrics**:
  - **Mean Absolute Error (MAE)**: `0.1590`
  - **R² Score**: `0.9537`
  
  These metrics suggest the model predicts GPA values with high accuracy. The **MAE** of 0.1590 means that, on average, the model's predictions are off by only about 0.16 GPA points. The **R² score** of 0.9537 indicates that 95.37% of the variance in GPA is explained by the model, reflecting a strong fit and predictive power.

---
