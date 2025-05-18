#!/usr/bin/env python
# coding: utf-8

# In[82]:


# Importing libraries 
import streamlit as st 
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR


# In[83]:


# Dataset Type detection
def detect_file_type(file):
    if file.name.endswith(".csv"):
        return 'csv'
    elif file.name.endswith(".json"):
        return "json"
    elif file.name.endswith(".xlsx"):
        return "excel"
    elif file.name.endswith((".docx", ".txt")):
        return "text"
    else:
        return "Unsupported file format!!!"


# In[84]:


def process_dataset(data, target_column):
    # Separating the dataset into X (features) and y (target)
    y = data[target_column]
    x = data.drop(columns=[target_column])
    
    # Identify column types
    numeric_columns = x.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = x.select_dtypes(include=['object']).columns

    if len(numeric_columns) == 0 and len(categorical_columns) == 0:
        st.error("No numeric or categorical columns found in the dataset.")
        return None, None

    # Create the ColumnTransformer for preprocessing
    transformer = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), numeric_columns if len(numeric_columns) > 0 else []),  # Impute numeric columns with mean
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_columns if len(categorical_columns) > 0 else [])  # OneHot encode categorical columns
        ],
        remainder='passthrough'  # Keep other columns as is
    )

    # Apply transformations to the features
    try:
        X_transformed = transformer.fit_transform(x)
    except ValueError as e:
        st.error(f"An error occurred during preprocessing: {str(e)}")
        return None, None

    return X_transformed, y


# In[85]:


# Determine the Problem Type (Classification or Regression)
def Problem_Type(y):
    if y.dtype == 'object' or len(y.unique()) <= 10:
        return 'classification'
    else:
        return 'regression'


# In[86]:


# Train and Evaluate Model for Classification
def Training_model_Classification(x, y):
    models = {
        "Random Forest Classifier": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
        "Support Vector Machine (SVM)": SVC()
    }
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    best_model = None
    best_accuracy = 0
    best_name = ""
    best_cm = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_name = name
            best_cm = confusion_matrix(y_test, y_pred)

    return best_name, best_accuracy, best_cm


# In[87]:


# Train and Evaluate Model for Regression
def Training_model_Regression(x, y):
    models = {
        "Random Forest Regressor": RandomForestRegressor(random_state=42),
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Support Vector Regressor (SVR)": SVR()
    }

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    best_model = None
    best_r2 = -np.inf
    best_name = ""
    best_y_pred = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2_score_value = model.score(X_test, y_test)

        if r2_score_value > best_r2:
            best_r2 = r2_score_value
            best_model = model
            best_name = name
            best_y_pred = y_pred

    return best_name, best_r2, best_y_pred, y_test


# In[88]:


# StreamLit App for creating upload button
st.title("Machine Learning Algorithm Recommendation")
st.write("Upload a dataset (CSV, JSON, EXCEL) to analyze and get insights.")


# In[89]:


# File Uploader function
uploaded_file = st.file_uploader("upload your dataset", type=["csv", "json", "xlsx"])
if uploaded_file is not None:
    file_type = detect_file_type(uploaded_file)
    if file_type == "Unsupported":
        st.error("Unsupported file type. Please Re-upload a CSV, JSON, or EXCEL file. ")
    else:
        if file_type == "csv":
            data= pd.read_csv(uploaded_file)
        elif file_type == "json":
            data= pd.read_json(uploaded_file)
        elif file_type == "excel":
            data = pd.read_excel(uploaded_file)

        st.write("Dataset Preview:")
        st.write(data.head())

        # Select the target column
        target_column = st.selectbox("Select the target column (label):", data.columns)


# In[91]:


# Process and Train
if st.button("Analyze Dataset"):
    if uploaded_file == None:
        st.warning("Upload the file before Analysing")
    else:
        x, y = process_dataset(data, target_column)

    # Determine the Problem
        Problem_Type = Problem_Type(y)

    if Problem_Type == 'classification':
        best_name, best_accuracy, best_cm = Training_model_Classification(x, y)
        st.write(f"Best Algorithm: {best_name}")
        st.write(f"Accuracy: {best_accuracy * 100:.2f}%")
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots()
        sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        st.write("* Best Algorithm suggested for Classification Problem from below:")
        st.write("1. Random Forest Classifier")
        st.write("2. Logistic Regression")
        st.write("3. Decision Tree Classifier")
        st.write("4. Support Vector Machine (SVM)")

        
    elif Problem_Type == 'regression':
        best_name, best_r2, best_y_pred, y_test = Training_model_Regression(x, y)
        st.write(f"Best Algorithm: {best_name}")
        st.write(f"R-squared Score: {best_r2 * 100:.2f}%")
            
        st.write("Residual Plot:")
        residuals = y_test - best_y_pred
        fig, ax = plt.subplots()
        ax.scatter(best_y_pred, residuals, color='Blue', alpha=0.5)
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_title("Residual Plot")
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals (Actual - Predicted)")
        st.pyplot(fig)


        
        st.write("* Best Algorithm suggested for Regression Problem from below:")
        st.write("1. Random Forest Regressor")
        st.write("2. Linear Regression")
        st.write("3. Decision Tree Regressor")
        st.write("4. Support Vector Regressor")


# In[ ]:




