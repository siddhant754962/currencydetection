


# 💵 Banknote Authentication using SVM

A **Machine Learning project** to classify **genuine vs. counterfeit banknotes** using the **UCI Banknote Authentication Dataset**.  
This project uses **Support Vector Machines (SVM)** with both **Linear** and **RBF kernels** and applies **hyperparameter tuning (GridSearchCV)** to achieve near-perfect accuracy.

---

## 📌 Project Overview

- Goal: Detect counterfeit banknotes using extracted numerical features from banknote images.  
- Algorithm: **Support Vector Machine (SVM)**  
- Approach:
  - Data Exploration (EDA)
  - Feature Scaling (StandardScaler)
  - Model Training (Linear & RBF kernels)
  - Hyperparameter Tuning (GridSearchCV)
  - Evaluation (Accuracy, Precision, Recall, F1-score, ROC Curve)
- Result: Achieved **~99%+ accuracy** with RBF Kernel SVM.

---

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository – Banknote Authentication](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)  
- **Size:** 1372 rows × 5 columns  
- **Features:**
  1. **Variance** of wavelet transformed image  
  2. **Skewness** of wavelet transformed image  
  3. **Curtosis** of wavelet transformed image  
  4. **Entropy** of image  
- **Target Variable:**
  - `0` → Genuine banknote  
  - `1` → Counterfeit banknote  

✅ Dataset is clean (no missing values).  
✅ Balanced distribution of genuine vs. fake notes.  

---

## ⚙️ Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Checked dataset shape, statistics, and class balance.  
- Visualized distributions of each feature using histograms and KDE plots.  
- Boxplots were used to identify outliers.  
- Correlation heatmap showed low-to-moderate feature correlations.  
- Pairplots indicated clear class separation for some features (variance & skewness).  

### 2. Data Preprocessing
- **Train-Test Split** (80% training, 20% testing) with `stratify=y` to maintain class balance.  
- **StandardScaler** applied to normalize features → ensures fair contribution of each feature.  

### 3. Model Training
- **Linear Kernel SVM** → tested for linearly separable patterns.  
- **RBF Kernel SVM** → captured nonlinear decision boundaries, better suited for dataset.  

### 4. Hyperparameter Tuning
- Used **GridSearchCV** with 5-fold cross-validation.  
- Parameters searched:
  - `C`: [0.1, 1, 10, 100]  
  - `gamma`: [0.01, 0.1, 1, "scale"]  
  - `kernel`: ["rbf"]  
- Best parameters found → `{'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}`  

### 5. Model Evaluation
- **Metrics:**
  - Accuracy  
  - Confusion Matrix  
  - Precision, Recall, F1-score  
  - ROC Curve and AUC  
- **Results:**
  - Linear Kernel → Accuracy ~97.5%  
  - RBF Kernel → Accuracy ~99–100%  
  - ROC AUC → ~0.99  

### 6. Model Saving
- Final best model and scaler saved with **Joblib**:  
  - `svm_model.pkl`  
  - `scaler.pkl`  

---

## 🚀 Results Summary

### 🔹 Linear Kernel SVM
- Accuracy: ~97.5%  
- Few misclassifications.  

### 🔹 RBF Kernel SVM
- Accuracy: **99–100%**  
- Perfect classification in most runs.  

### 🔹 Best Model (After GridSearchCV)
- Parameters: `{'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}`  
- Accuracy: **99%+**  
- ROC AUC: **0.99+**  

✅ Strong separation between genuine and counterfeit notes.  
✅ Model is robust, highly reliable, and CV-ready.  

---

## 📈 Visualizations

- Class distribution bar chart  
- Feature distribution histograms & KDE plots  
- Pairplots by class  
- Correlation heatmap  
- Boxplots for outlier detection  
- ROC Curve with AUC  

---

## 🛠️ Technologies Used

- **Python 3.9+**  
- **Libraries:**
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `joblib`  

---

## ▶️ How to Run the Project

1. **Clone the repository**
```bash
git clone <your-repo-link>
cd banknote-authentication
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run training script**

```bash
python main.py
```

4. **Model & Scaler will be saved as:**

* `svm_model.pkl`
* `scaler.pkl`

5. **Use trained model for new predictions**

```python
import joblib

# Load model & scaler
svm_model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Example prediction
X_new = [[3.5, 6.1, -1.2, 0.5]]  # variance, skewness, curtosis, entropy
X_new_scaled = scaler.transform(X_new)
prediction = svm_model.predict(X_new_scaled)

print("Prediction:", "Genuine" if prediction[0]==0 else "Fake")
```

---



**Banknote Authentication using SVM**

* Implemented a Support Vector Machine classifier to distinguish between genuine and counterfeit banknotes using the UCI dataset.
* Performed Exploratory Data Analysis (EDA) and feature scaling for preprocessing.
* Compared Linear and RBF kernels, applied hyperparameter tuning with GridSearchCV.
* Achieved **99% accuracy and 0.99 ROC-AUC**.
* Demonstrated strong skills in fraud detection, ML pipelines, and model evaluation.






