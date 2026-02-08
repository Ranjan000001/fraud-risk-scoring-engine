# 🚨 Fraud Risk Scoring Engine
## 📘 Project Overview
This project focuses on building an end-to-end Fraud Risk Scoring Engine using machine learning to identify suspicious financial transactions. The system analyzes transaction behavior such as amount, device used, payment method, and location to assign fraud risk and help financial institutions minimize losses.

## 📂 Dataset Information
- **Source**: Simulated / Kaggle-style transaction dataset
- **Rows**: ~50,000 transactions
- **Target** Variable: Fraudulent (0 = Legitimate, 1 = Fraud)
- 
## 🧠 Problem Statement
Financial fraud causes significant revenue loss for banks and digital payment platforms. The challenge is to predict whether a transaction is fraudulent based on historical transaction data while handling class imbalance and minimizing incorrect predictions.

## 🛠️ Tools & Technologies Used

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Jupyter Notebook

## Work Flow
### Import Libraries
```jupyter 
# libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
```
### Load File and Info
```jupyter
# loading and info, head
fraud_data = pd.read_csv('/kaggle/input/fraud-detection-dataset-csv/Fraud Detection Dataset.csv')

fraud_data.info()
fraud_data.tail()
```
### Null and Duplicates
```jupyter
# null and duplicated
fraud_data.isnull().sum()
fraud_data.duplicated().sum()
```
### Statics 
```jupyter
# mathematics states
fraud_data.describe()
```
## EDA
```jupyter
# Fraud distribution and Features wise Fraud Patterns
print(fraud_data['Fraudulent'].value_counts())
for col in fraud_data.columns.difference(['Transaction_ID', 'User_ID', 'Fraudulent']):
    print(pd.crosstab(fraud_data[col],fraud_data['Fraudulent'], normalize='columns')*100)
```
### Insides 
```jupyter 
1) Device: 34.18% chance fraud is done by mobile, and 66% mobile or desktop and unknown device is the lowest
2) Amount: The highest fraud amount is 49,997.80 (32 cases), while the lowest is 14.31 with a single occurrence.
3) Location: Chicago has the highest fraud count (334), followed by Boston and Los Angeles (310 each), with Seattle showing the lowest.
4) Transaction Type: Online purchases are the most fraud-prone, whereas ATM withdrawals have the lowest fraud occurrence.
5) Previous Fraud: Users with a prior fraud history are significantly more likely to commit fraud, accounting for 519 cases.
6) Account Age: Accounts under 100 days old contribute to 83% of total fraud, highlighting higher risk in new accounts.
7) Frequency: 72% of fraud occurs within fewer than 10 transactions in 24 hours, with noticeable spikes at 6 and 12 transactions.
8) Payment Method: UPI has the highest fraud cases (612), followed by credit cards and net banking (574), while invalid methods are least risky.
10) Time: Fraud activity is higher around hours 15, 20, and 22, indicating peak-risk time windows.
```
### summary in short
```jupyter 
Fraudsters are repeated offenders operating through newly created accounts, primarily using mobile and desktop to make online purchases via UPI around 3 PM, frequently executing high-value transactions (49,997.8) with burst activity of about 6 transactions within the last 24 hours.
```

### Visualization
```jupyter
# graphical representation numerical 
fraud = fraud_data[fraud_data['Fraudulent']==1]
for col in fraud.select_dtypes(include=np.number).columns.difference(['Fraudulent','User_ID', 'Previous_Fraudulent_Transactions']):
    sns.violinplot(data=fraud, y=col)
    plt.title(f'graphical representation of Fraudulent(1) vs {col}')
    plt.ylabel(col)
    plt.show()
```
```jupyter
# graphical representation of categorical 
fraud = fraud_data[fraud_data['Fraudulent']==1]
colum = fraud.select_dtypes(include='object').columns.tolist() +['Previous_Fraudulent_Transactions']
for col in set(colum)-{'Transaction_ID'}:
    sns.countplot(data=fraud, x=col)
    plt.title(f'graphical representation of Fraudulent(1) vs {col}')
    plt.xlabel(col)
    plt.ylabel('Fraudulent')
    plt.show()
```
```jupyter
# feature bucket
time_bucket = pd.cut(fraud_data['Time_of_Transaction'], bins=[0,10,14,19,24], labels=['morning','evening','afternoon','night'])
age_bucket = pd.cut(fraud_data['Account_Age'], bins=[0,10,30,100,240], labels=['new','young','mid','old'])
amount_bucket = pd.cut(fraud_data['Transaction_Amount'], bins=[0,80,900,1500,5000,100000], labels=['very small','small','moderate','high','very high'])
trans_bucket = pd.qcut(fraud_data['Number_of_Transactions_Last_24H'], q=4, labels=['single','low','mid','high'])

pd.crosstab([fraud_data['Location'], fraud_data['Transaction_Type']],fraud_data['Fraudulent'],normalize='index')*100
pd.crosstab([ fraud_data['Previous_Fraudulent_Transactions'], age_bucket],fraud_data['Fraudulent'],normalize='index')*100
pd.crosstab([ fraud_data['Previous_Fraudulent_Transactions'], amount_bucket],fraud_data['Fraudulent'],normalize='index')*100
pd.crosstab([ time_bucket, trans_bucket],fraud_data['Fraudulent'],normalize='index')*100
pd.crosstab([fraud_data['Device_Used'], fraud_data['Payment_Method']], fraud_data['Fraudulent'],normalize='index')*100
```
## Insides 
```jupyter
- top 2 are online purchases and bank transfer chicago and los angeles in main highly fraud and seattle both least, POS payment have the lowest fraud rate.
- by overall number upi(top on tablet) is bigger, but top 2 positions debit card in higher fraud mobile and desktop and in unknown device 2nd lowest debit card
- new account (0 to 9 days) are top on not offenders, one and two time offenders, but 3 to 4 time offenders have mid(30 to 99) and old(+100) are top respectively
- 1 and 4 time offenders do very small(0 to 80) amount, but 0,2, and 3 time offender do very high (5000 to 100000) amount.
- morning, afternoon, and night have low(4 to 7) transaction, and evening have mid(7 to 11) transaction on last 24h.
```
### Model Preparing
```jupyter
# split the data
X = fraud_data[fraud_data.columns.difference(['Fraudulent', 'User_ID', 'Transaction_ID'])]
y = fraud_data['Fraudulent']
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42,stratify=y)
```
### Filling Null
```jupyter
# fill numerical values
num_col=['Transaction_Amount','Time_of_Transaction']
for col in num_col:
    X_train[col]=X_train[col].fillna(X_train[col].median())
    X_test[col]=X_test[col].fillna(X_train[col].median())

X_train.info()
```
```jupyter
# fill categorical values
cat_col = ['Device_Used', 'Location','Payment_Method']
for col in cat_col:
    X_train[col]=X_train[col].fillna(X_train[col].mode()[0])
    X_test[col]=X_test[col].fillna(X_train[col].mode()[0])

X_train.info()
```

### Feature engineering
```jupyter
#  Train feature engineering
X_train['time_bucket'] = pd.cut(X_train['Time_of_Transaction'], bins=[0,10,14,19,24], labels=['morning','evening','afternoon','night'])
X_train['Age_bucket'] = pd.cut(X_train['Account_Age'], bins=[0,10,30,100,240], labels=['new','young','mid','old'])
X_train['Amount_bucket'] = pd.cut(X_train['Transaction_Amount'], bins=[0,80,900,1500,5000,100000],labels=['very small', 'small', 'moderate', 'high', 'very high'])
X_train['Trans_bucket']= pd.cut(X_train['Number_of_Transactions_Last_24H'],bins=[0,4,7,11,14],labels=['single','low', 'mid', 'high'])

#  Train feature engineering
X_train['time_bucket'] = pd.cut(X_train['Time_of_Transaction'], bins=[0,10,14,19,24], labels=['morning','evening','afternoon','night'])
X_train['Age_bucket'] = pd.cut(X_train['Account_Age'], bins=[0,10,30,100,240], labels=['new','young','mid','old'])
X_train['Amount_bucket'] = pd.cut(X_train['Transaction_Amount'], bins=[0,80,900,1500,5000,100000],labels=['very small', 'small', 'moderate', 'high', 'very high'])
X_train['Trans_bucket']= pd.cut(X_train['Number_of_Transactions_Last_24H'],bins=[0,4,7,11,14],labels=['single','low', 'mid', 'high'])
```
Features Definitions & Value Ranges
# time_bucket – Time of Transaction (Hours)
## Bucket Label	Range
- morning	0 – 10
- evening	10 – 14
- afternoon	14 – 19
- night	19 – 24
# age_bucket – Account Age (Days)
## Bucket Label	Range
- new	0 – 10 days
- young	10 – 30 days
- mid	30 – 100 days
- old	100 – 240 days
# amount_bucket – Transaction Amount
## Bucket Label	Range
- very small	0 – 80
- small	80 – 900
- moderate	900 – 1500
- high	1500 – 5000
- very high	5000 – 100000
# trans_bucket – Transactions in Last 24 Hours
## Bucket Label	Description
- single	Lowest 25% of transaction counts
- low	25% – 50%
- mid	50% – 75%
- high	Highest 25%
### Encoding
```jupyter
# Encoding
X_train_p = X_train.drop(columns=['Age_bucket', 'time_bucket', 'Trans_bucket', 'Amount_backet'])
X_test_p = X_test.drop(columns=['Age_bucket', 'time_bucket', 'Trans_bucket', 'Amount_backet'])
X_train_p.info()
X_test_p.info()
col = X_train_p.select_dtypes(include='object').columns
num_col = X_train.select_dtypes(include=np.number).columns
preprocessing = ColumnTransformer(transformers=[('num', StandardScaler(), num_col), ('cat', OneHotEncoder(handle_unknown='ignore',sparse_output=False), col)])
X_train_pre = preprocessing.fit_transform(X_train_p)
X_test_pre = preprocessing.transform(X_test_p)
```
## Logistic Regression 
```jupyter
lr = LogisticRegression(class_weight='balanced', random_state=42)
lr.fit(X_train_pre,y_train)
score=lr.score(X_test_pre,y_test)
y_pre_lr=lr.predict(X_test_pre)
y_preo_lr = lr.predict_proba(X_test_pre)[:, 1]
y_pred_lr = (y_preo_lr>0.5).astype(int)
cm = confusion_matrix(y_test,y_pred_lr)
plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt='d',cmap='coolwarm')
plt.title('Confusion Matrix (LogisticRegression)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print(f'acuracy: {score}')
```
```jupyter
from sklearn.metrics import roc_auc_score
y_preo = lr.predict_proba(X_test_pre)[:, 1]
print(f'Roc Auc Score:{roc_auc_score(y_test, y_preo)}')
print(classification_report(y_test, y_pred_lr))
```
### LR conclusion
Logistic Regression with class balancing improves fraud sensitivity, its ROC-AUC (~0.49) indicates weak separation between fraud and non-fraud classes.At the tuned threshold, the model captures approximately 45% of fraud cases, but this comes with a high number of false alarms, leading to a drop in overall accuracy. Try Random Forest Model
## Random Forest Classifier
```jupyter
rf = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=20, class_weight='balanced',random_state=42, n_jobs=-1)
rf.fit(X_train_pre,y_train)
y_pre = rf.predict(X_test_pre)
y_preo = rf.predict_proba(X_test_pre)[:, 1]
y_pred_rf = (y_preo>0.5).astype(int)
score=rf.score(X_test_pre,y_test)
# confusion matrix
cm = confusion_matrix(y_test,y_pred_rf)
plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix(Random Forest)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print(f'accuracy: {score}')
```

```jupyter
y_preo = rf.predict_proba(X_test_pre)[:, 1]
print(f'Roc Auc Score:{roc_auc_score(y_test, y_preo)}')
print(classification_report(y_test, y_pred_rf))
```
### RF conclusion
The Random Forest model achieves high overall accuracy due to class imbalance, but at the default threshold(0.5) it detects very few fraud cases(7%). That's why see XGBoost
## XG Boost
```jupyter
xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,subsample=0.8, colsample_bytree=0.8, scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(), objective='binary:logistic', random_state=42, eval_metric='aucpr')
xgb.fit(X_train_pre, y_train)
y_preo_xgb = xgb.predict_proba(X_test_pre)[:, 1]
y_pred_xgb = (y_preo_xgb>0.4).astype(int)
print(classification_report(y_test, y_pred_xgb))
```
### XG conclusion
The XGBoost model with a tuned decision threshold of 0.4 captures approximately 58% of fraud cases while reducing false alarms compared to lower thresholds, making it a practical trade-off for real-world fraud detection where recall is prioritized over raw accuracy 

# Project Summery 
The fraud  risk scoring engine to detect suspicious financial transactions using machine learning. Through extensive EDA, key fraud patterns were identified: fraud is concentrated in New Account,Repeat Offenders, Online Transaction, UPI Payment, High Volume Payment, and Specific Time Window. Feature bucketing captured non-linear risk behavior across time, account age, transaction frequency, and amount. Logistic Regression and Random Forest showed limitations under class imbalance, while XG Boosting achive the best trade of and deteing 58% approx fraud cases with reduced false alarms. The final system demonstrates how data-driven risk modeling can enhance fraud prevention strategies in Digital payment.

# Business inputs to reduce fraud cases
- Transactions  between $5000 to $45000 should required OTP Veriffication along with Phone call confirmation.
- Transactions above $45000  should trigger enhanced profile verification, such as customer identity details, verification questions, and reason of the transaction.
- Implement high-risk monitoring for transactions occurring after 3:00 PM, especially for repeat-offender accounts where the transaction amount above $45000.
- Send sms to customers awareness about fraud and how new types of fraud happens.
-  Enforce Stricter and multiple  KYC for new accounts and users with a history of repeated fraudulent activity



## Data Source
link: https://www.kaggle.com/datasets/ranjitmandal/fraud-detection-dataset-csv
