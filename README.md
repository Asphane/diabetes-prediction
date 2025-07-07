# 🩺 Diabetes Prediction App

This Streamlit web application predicts the probability of a person having diabetes based on key medical features such as glucose level, BMI, insulin levels, and more. It uses a machine learning model trained and validated using multiple approaches, with final selection based on performance.

---

## 🚀 Key Highlights

- Accepts real-time input of medical data  
- Uses a trained AdaBoostClassifier pipeline  
- Automatically handles imputation and scaling  
- Returns **risk probability in percentage (%)**  
- Built with Streamlit for easy web deployment  

---

## 🧠 Model Selection Process

The following models were evaluated using `RandomizedSearchCV`:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- AdaBoost  
- XGBoost  

After hyperparameter tuning and evaluation on the test set, **AdaBoost** was selected based on its superior balance between performance and interpretability.

---

## ✅ Final Model: AdaBoostClassifier

- **Base Estimator:** DecisionTreeClassifier (max_depth=2)  
- **Learning Rate:** 0.2  
- **n_estimators:** 50  
- **Algorithm:** SAMME  

### 📈 Final Test Set Performance

| Metric      | Score   |
|-------------|---------|
| Accuracy    | 0.7013  |
| Precision   | 0.5600  |
| Recall      | 0.7636  |
| F1 Score    | 0.6462  |
| ROC AUC     | 0.8378  |

---

## ⚙️ Preprocessing Pipeline

The model was trained within a `sklearn.pipeline.Pipeline`:

1. **Imputer:** `SimpleImputer(strategy='mean')`  
2. **Scaler:** `StandardScaler()`  
3. **Classifier:** `AdaBoostClassifier`  

The entire pipeline is saved as `diabetes_model_pipeline.pkl`, ensuring consistent prediction during deployment.

---

## 📁 Project Structure
```
📦 diabetes-prediction-app/
├── app.py # Streamlit frontend
├── diabetes_model_pipeline.pkl # Trained ML pipeline
├── features_used.pkl # Feature list used for training
├── requirements.txt # Dependencies for deployment
├── main.ipynb # Model training and selection notebook
└── README.md # This file
```

---

## 📊 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/diabetes-prediction-app.git
cd diabetes-prediction-app

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py

---
🌐 Deploy on Streamlit Cloud
Push this folder to a public GitHub repository

Go to https://share.streamlit.io

Click New App

Connect your GitHub account and select the repo

Set app.py as the main file

Click Deploy

A public app URL will be generated automatically!

✨ Credits
Trained using Pima Indians Diabetes Dataset

Model selection, tuning & training by Bisakh Patra

Built with ❤️ using Python, Scikit-learn, and Streamlit

---
