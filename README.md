# Customer Churn Prediction

## Project Overview

Customer churn prediction is a classification problem where the goal is to identify customers who are likely to discontinue a service.

In this project, I analyzed a telecom customer dataset and built machine learning models to predict whether a customer is likely to churn based on customer demographics, tenure, services, contract type, and billing information.

The project follows an end-to-end Data Science workflow:

**Data Collection → Data Cleaning → EDA → Feature Preprocessing → Model Training → Evaluation → Model Interpretation → Insights**

---

## Problem Statement

Telecom companies can lose customers when users discontinue their services.

The objective of this project is to:

* Understand customer churn patterns.
* Identify factors associated with customer churn.
* Build machine learning models to predict churn.
* Evaluate model performance using multiple classification metrics.
* Interpret model results and derive useful business insights.

---

## Dataset

The project uses the **IBM Telco Customer Churn dataset**, obtained through Kaggle.

* **Rows:** 7,043
* **Original columns:** 21
* **Features used before preprocessing:** 19
* **Processed model features:** 45
* **Target variable:** `Churn`
* **Target classes:** `Yes`, `No`

The dataset contains information about:

* Customer demographics
* Tenure
* Phone and internet services
* Online services
* Contract type
* Payment method
* Monthly charges
* Total charges

---

## Data Cleaning

The following preprocessing steps were performed:

* Checked the dataset structure and data types.
* Checked for duplicate customer IDs.
* Converted `TotalCharges` from object to numeric.
* Identified 11 missing values in `TotalCharges`.
* Filled these missing values with `0`, as the corresponding customers had zero tenure.
* Removed `customerID` because it is an identifier rather than a predictive feature.

---

## Exploratory Data Analysis

Several customer attributes were analyzed to understand churn patterns.

### Overall Churn

Approximately **26.5%** of customers in the dataset had churned.

### Contract Type

Observed churn differed substantially across contract types:

| Contract       | Churn Rate |
| -------------- | ---------: |
| Month-to-month |     42.71% |
| One year       |     11.27% |
| Two year       |      2.83% |

### Tenure

Customers with shorter tenure showed considerably higher observed churn.

Average tenure:

| Customer Group | Average Tenure |
| -------------- | -------------: |
| Stayed         |   37.57 months |
| Churned        |   17.98 months |

### Internet Service

Observed churn rates were:

| Internet Service    | Churn Rate |
| ------------------- | ---------: |
| DSL                 |     18.96% |
| Fiber optic         |     41.89% |
| No internet service |      7.40% |

### Monthly Charges

Average monthly charges were:

| Customer Group | Average Monthly Charges |
| -------------- | ----------------------: |
| Stayed         |                   61.27 |
| Churned        |                   74.44 |

These are observed relationships in the dataset and should not be interpreted as causal effects.

---

## Feature Preprocessing

The dataset contained both numerical and categorical variables.

### Numerical Features

Numerical features were standardized using `StandardScaler`.

### Categorical Features

Categorical variables were converted into numerical representations using `OneHotEncoder`.

`handle_unknown="ignore"` was used so that previously unseen categories would not cause transformation errors.

After preprocessing:

* Original input features: **19**
* Processed model features: **45**

The preprocessing pipeline was fitted only on the training data and then applied to the test data to avoid data leakage.

---

## Machine Learning Models

Two classification models were trained and compared.

### 1. Logistic Regression

Logistic Regression was used as the primary baseline classification model.

**Test Accuracy:** 80.55%

| Class    | Precision | Recall | F1-score |
| -------- | --------: | -----: | -------: |
| No Churn |      0.85 |   0.89 |     0.87 |
| Churn    |      0.66 |   0.56 |     0.60 |

### 2. Random Forest

Random Forest was trained to compare its performance with Logistic Regression.

**Test Accuracy:** 78.35%

| Class    | Precision | Recall | F1-score |
| -------- | --------: | -----: | -------: |
| No Churn |      0.83 |   0.89 |     0.86 |
| Churn    |      0.62 |   0.48 |     0.54 |

For this particular train/test split and configuration, Logistic Regression produced higher test accuracy and higher churn recall than Random Forest.

---

## Model Improvement Experiments

After establishing the Logistic Regression baseline, additional experiments were performed to understand how different approaches affect churn prediction.

### Class Weighting

A class-weighted Logistic Regression model was tested to give more importance to the minority `Churn = Yes` class.

| Metric         | Baseline Logistic Regression | Class-Weighted Logistic Regression |
| -------------- | ---------------------------: | ---------------------------------: |
| Accuracy       |                       80.55% |                             73.81% |
| Churn Recall   |                          56% |                                78% |
| Churn F1-score |                         0.60 |                               0.61 |

The experiment showed that increasing attention to the minority class improved churn recall but reduced overall accuracy and churn precision.

### Classification Threshold

The prediction threshold of the original Logistic Regression model was changed from the default `0.50` to `0.40`.

Results:

* **Accuracy:** 77.71%
* **Churn Precision:** 57%
* **Churn Recall:** 67%
* **Churn F1-score:** 0.61

Lowering the threshold increased the model's ability to identify churners, but also resulted in more false positives.

### Final Model Selection

The original Logistic Regression model with the default `0.50` threshold was retained as the final model for this project.

The additional experiments were kept as comparison and learning exercises because they demonstrated the trade-offs between accuracy, precision, and recall rather than producing an improvement across all evaluation metrics.

---

## Model Evaluation

A confusion matrix was used to examine the Logistic Regression predictions.

```text
                 Predicted
                 No     Yes

Actual No       926    109
Actual Yes      165    209
```

The model correctly identified **209 out of 374 actual churn customers** in the test set.

Because the dataset contains more non-churn customers than churn customers, accuracy alone was not used to evaluate the model. Precision, recall, F1-score, and the confusion matrix were also considered.

---

## Model Interpretation

The Logistic Regression coefficients were examined to understand which processed features provided stronger signals to the model.

Some of the strongest coefficient magnitudes included:

* `tenure`
* `Contract_Two year`
* `InternetService_DSL`
* `InternetService_Fiber optic`
* `MonthlyCharges`
* `Contract_Month-to-month`
* `TotalCharges`

The model particularly showed strong signals associated with **customer tenure and contract type**.

Coefficient interpretation describes patterns learned by the model and should not be interpreted as proof that a particular factor causes churn.

---

## Key Insights

Based on the analysis:

1. Customers with shorter tenure showed higher observed churn.
2. Month-to-month contract customers had substantially higher observed churn than customers with longer-term contracts.
3. Fiber optic customers had the highest observed churn rate among the internet-service groups.
4. Churned customers had higher average monthly charges than customers who stayed.
5. Tenure and contract-related variables were among the strongest signals in the Logistic Regression model.
6. The baseline model achieved **80.55% test accuracy** and identified **56% of actual churners** at the default classification threshold.
7. Model improvement experiments demonstrated the trade-off between identifying more churners and maintaining precision and overall accuracy.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook
* KaggleHub

---

## Project Structure

```text
customer-churn-prediction/
│
├── customer_churn_analysis.ipynb
├── README.md
└── .gitignore
```

---

## Future Improvements

Possible extensions to this project include:

* Hyperparameter tuning
* Cross-validation
* Testing additional classification algorithms
* More systematic threshold optimization
* ROC-AUC and Precision-Recall analysis
* Feature engineering
* Deployment as a simple prediction application

---

## Author

**Mahima Agarwal**

This project was developed as a hands-on Data Science project to practice the complete machine learning workflow from data collection and exploration to model interpretation and business insights.
