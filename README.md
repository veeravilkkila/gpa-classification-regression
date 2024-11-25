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
   - **Training**: Trains the model on the classification task, predicting GPA categories (1-4).
   - **Evaluation**: Evaluates the model performance on the test data, reporting metrics like accuracy, confusion matrix, and classification report.

### 3. **Model_BNN_regression.ipynb**
   This notebook is for training a **Bayesian Neural Network (BNN)** for the **regression** task. It includes:
   - **Data Loading**: Loads the preprocessed `.pkl` data.
   - **Data Splitting**: The data is split into training and testing sets (80/20) to evaluate the model's performance. 
   - **Model Construction**: Builds a BNN model for regression.
   - **Training**: Trains the model to predict exact GPA values based on student features.
   - **Evaluation**: Evaluates the model's performance using regression metrics such as Mean Squared Error (MSE) and R² score, along with visualizations to better understand the model's predictions.

The project showcases the benefits of using a **Bayesian Neural Network** (BNN), which provides a measure of uncertainty in the predictions. This is particularly useful for regression and classification tasks where uncertainty quantification is important.


