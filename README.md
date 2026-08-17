# 🌾 Farmer Profit Prediction Using Machine Learning

A machine learning regression project that predicts **farmer profit in INR** using agricultural, environmental, crop, irrigation, and market-related factors.

## 📌 Project Overview

Agricultural profitability depends on several factors such as rainfall, temperature, humidity, crop type, soil type, irrigation method, fertilizer usage, cultivated area, crop yield, and market price.

This project applies machine learning regression techniques to predict **`Farmer_Profit_INR`** from these features.

The dataset contains **7,920 observations and 17 columns** before preprocessing.

---

## 🎯 Objectives

* Predict farmer profit using machine learning.
* Analyze the relationship between agricultural factors and profitability.
* Handle numerical and categorical variables using appropriate preprocessing.
* Compare multiple regression models.
* Evaluate model performance using **R², MAE, and RMSE**.
* Use cross-validation to assess model generalization.

---

## 📊 Dataset

The dataset contains information related to agricultural production from **2014 to 2024**.

### Features

| Feature                    | Description                        |
| -------------------------- | ---------------------------------- |
| `State`                    | State where the crop is cultivated |
| `District`                 | District of cultivation            |
| `Year`                     | Year of cultivation                |
| `Season`                   | Agricultural season                |
| `Crop_Type`                | Type of crop                       |
| `Rainfall_mm`              | Rainfall in millimeters            |
| `Avg_Temperature_C`        | Average temperature                |
| `Humidity_percent`         | Humidity percentage                |
| `Soil_Type`                | Type of soil                       |
| `Irrigation_Type`          | Irrigation method                  |
| `Fertilizer_Used_kg`       | Fertilizer quantity used           |
| `Pesticide_Used_kg`        | Pesticide quantity used            |
| `Cultivated_Area_hectares` | Cultivated area                    |
| `Yield_ton_per_hectare`    | Crop yield                         |
| `Market_Price_per_ton`     | Market price per ton               |
| `Farmer_Profit_INR`        | **Target variable**                |

The original notebook also removes `Record_ID` because it is an identifier rather than a predictive feature.

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* SciPy
* Scikit-learn
* Jupyter Notebook

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Train-Test Split
   ↓
Numerical & Categorical Preprocessing
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Cross-Validation
   ↓
Model Comparison
```

---

## 🧹 Data Preprocessing

The project performs the following preprocessing steps:

### Numerical Features

* Missing-value imputation using median
* Feature scaling using `StandardScaler`

### Categorical Features

* Missing-value imputation using the most frequent value
* One-hot encoding using `OneHotEncoder`
* `handle_unknown="ignore"` is used for unseen categories

The preprocessing is implemented using a Scikit-learn `ColumnTransformer` and `Pipeline`.

---

## ⚙️ Feature Engineering

Additional features were created to capture relationships between agricultural variables:

* `Fertilizer_per_Hectare`
* `Pesticide_per_Hectare`
* `Rain_Temp_Index`

These features are included in the numerical preprocessing pipeline.

---

## 🤖 Machine Learning Models

The project compares multiple regression algorithms:

### 1. Linear Regression

A baseline regression model used to establish a performance benchmark.

### 2. Ridge Regression

Ridge regression is implemented with L2 regularization.

```python
Ridge(alpha=1.0)
```

### 3. Lasso Regression

Lasso regression is implemented with L1 regularization.

```python
Lasso(alpha=0.1)
```

### 4. Random Forest Regression

A non-linear ensemble model that combines multiple decision trees.

```python
RandomForestRegressor(random_state=42)
```

## The notebook uses Scikit-learn pipelines to combine preprocessing and model training.

## 📏 Model Evaluation

Models are evaluated using:

### R² Score

Measures how much variance in farmer profit is explained by the model.

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted profit.

### Root Mean Squared Error (RMSE)

Measures prediction error while giving greater weight to larger errors.

```python
def evaluate(y_true, y_pred):
    print("R2 :", r2_score(y_true, y_pred))
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
```

These evaluation metrics are implemented directly in the notebook.

---

## 📈 Linear Regression Results

For the Linear Regression model, the notebook reports:

| Dataset  |         R² |           MAE |          RMSE |
| -------- | ---------: | ------------: | ------------: |
| Training |     0.7851 |     45,846.31 |     58,487.59 |
| Testing  | **0.7847** | **46,071.73** | **59,786.72** |

The model also achieved a mean **5-fold cross-validation R² of approximately 0.7812**.

---

## 📁 Project Structure

```text
Farmer-Profit-Prediction/
│
├── my_project.ipynb
├── regresion ML project.csv
├── README.md
│
└── images/
    └── model_results.png
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Farmer-Profit-Prediction.git
cd Farmer-Profit-Prediction
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open

```text
my_project.ipynb
```

### 5. Run all cells

Make sure the dataset is located in the same directory as the notebook:

```text
regresion ML project.csv
```

---

## 🔮 Future Improvements

* Hyperparameter tuning using GridSearchCV
* XGBoost / Gradient Boosting comparison
* Feature importance analysis
* SHAP-based model explainability
* Deployment using Streamlit or Flask
* Development of an interactive farmer-profit prediction dashboard
* Testing on external agricultural datasets

---

## 👨‍💻 Author

**Your Name**

If you found this project useful, consider ⭐ starring the repository!
