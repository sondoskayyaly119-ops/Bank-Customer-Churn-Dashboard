# 🏦 Bank Customer Churn Prediction Dashboard

## 📌 Overview

This project focuses on predicting customer churn in a banking system using Machine Learning techniques.

The goal is to identify customers who are likely to leave the bank and provide insights that can help improve customer retention strategies.

The project includes data analysis, preprocessing, model training, evaluation, and deployment through an interactive Streamlit dashboard.


---

## 🎯 Problem Statement

Customer churn is a major challenge for banking institutions. Predicting which customers are likely to leave allows banks to take proactive actions and improve customer satisfaction.

This project builds a classification model to predict whether a customer will exit the bank or remain.


---

## 📂 Dataset

The project uses the **Bank Customer Churn Dataset (Churn_Modelling)**.

The dataset contains customer information such as:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card Status
- Active Membership
- Estimated Salary

The target variable is:

- `Exited`
    - 0 → Customer stays
    - 1 → Customer leaves


---

## 🔄 Project Workflow

The project follows these steps:

### 1. Data Exploration (EDA)

- Understanding customer characteristics
- Analyzing churn distribution
- Studying relationships between features and customer exit behavior


### 2. Data Preprocessing

Applied preprocessing techniques:

- Feature scaling using `StandardScaler`
- Categorical encoding using `OneHotEncoder`


### 3. Feature Engineering

Created an additional feature:

- `BalancePerProduct`

which represents the customer's balance relative to the number of products.


### 4. Model Training

Several machine learning algorithms were evaluated, including:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM


### 5. Hyperparameter Optimization

Hyperparameter tuning was performed to improve model performance.


### 6. Model Deployment

The final model was deployed using Streamlit to create an interactive prediction dashboard.


---

## 🤖 Final Model

The final selected model is:

**Random Forest Classifier**

The model was integrated with a complete pipeline containing:

- Data preprocessing
- Feature transformation
- Classification model


---

## 📊 Dashboard Features

The Streamlit dashboard provides:

### Customer Prediction

Users can enter customer information and receive:

- Churn prediction
- Churn probability


### Customer Overview

The dashboard displays:

- Total customers
- Churn rate
- Average customer age


---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Streamlit
- Joblib
- Matplotlib
- Seaborn


---

## 📁 Project Structure
Bank-dashbourd/
│
├── app.py
├── models/
│ └── bank_churn_model.pkl
│
├── data/
│ └── Churn_Modelling.csv
│
├── requirements.txt
│
└── README.md




---

## ▶️ How to Run the Project

Install required libraries:

```bash
pip install -r requirements.txt

streamlit run app.py

📌 Results

The developed model can successfully predict customer churn and provides an interactive interface for testing new customer cases.

The dashboard demonstrates the practical deployment of a machine learning model into a user-friendly application.

👩‍💻 Author

Sondos Kayyali