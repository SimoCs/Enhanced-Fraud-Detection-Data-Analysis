# Enhanced Fraud Detection Data Analysis

Fraud detection is critical to financial and online services, ensuring the security and integrity of transactions and user accounts. With the rapid increase in digital transactions and the emergence of sophisticated fraudulent schemes, businesses must adopt advanced fraud detection mechanisms to protect their assets and customers. Data analysis and machine learning have become indispensable tools in this endeavor, as they enable the identification of unusual patterns and behaviors that may indicate fraudulent activities.

This project aims to develop a robust and efficient fraud detection system using data analysis and machine learning techniques. By analyzing a dataset of historical transactions, we will build a model that can effectively identify potential fraud cases. The insights gained from this project can be applied across various industries, including banking, e-commerce, and insurance, to enhance their security measures and minimize the impact of fraud on their operations.

Throughout this project, we will perform the following tasks:

- Import necessary libraries for data manipulation, analysis, and modeling.

- Acquire and explore a dataset of transaction records to understand its structure and characteristics.

- Clean and preprocess the data to prepare it for machine learning.

- Engineer new features and select the most relevant ones for modeling.

- Split the data into training and testing subsets to evaluate model performance.

- Select, implement, and compare various machine learning algorithms for fraud detection.

- Train the chosen models and optimize their hyperparameters to enhance their performance.

- Evaluate the models using appropriate performance metrics and interpret the results.

By the end of this project, we expect to have a comprehensive understanding of the data and its underlying patterns, and a reliable machine-learning model that can effectively detect potential fraud cases.

## Requirements

We must import several essential libraries that provide data manipulation, visualization, and machine learning tools to carry out this fraud detection data analysis project. Below is a brief overview of the libraries used in this project and their purposes:

### Data Analysis

- **Pandas**: A powerful data manipulation and analysis library that provides flexible data structures like DataFrames for handling tabular data.
- **NumPy**: A Python numerical computing library that supports arrays, matrices, and mathematical functions.

### Data Visualization

- **Matplotlib**: A widely-used library for creating static, animated, and interactive visualizations in Python.
- **Seaborn**: A statistical data visualization library based on matplotlib that simplifies the creation of informative and attractive visualizations.

### Machine Learning

- **Scikit-learn**: A popular library for machine learning in Python, providing tools for data preprocessing, model training, evaluation, and more.
  - **SimpleImputer**: A class for handling missing data by replacing missing values with specified strategies such as mean, median, or mode.
  - **train_test_split**: A function for splitting data into training and testing subsets.
  - **StandardScaler**: A class for standardizing numerical features by transforming them to have zero mean and unit variance.
- **Imbalanced-learn**: A library for handling imbalanced datasets in machine learning, offering various resampling techniques.
  - **SMOTE**: A class implementing the Synthetic Minority Over-sampling Technique for addressing class imbalance issues.
- **RandomForestClassifier**: An ensemble learning method from Scikit-learn that constructs multiple decision trees for classification tasks.
- **LogisticRegression**: A linear model for binary classification from Scikit-learn, based on the logistic function.
- **XGBoost**: An optimized distributed gradient boosting library that implements machine learning algorithms for regression, classification, and ranking tasks.
  - **xgb**: The XGBoost module in Python.
- **roc_auc_score**: A function from Scikit-learn that computes the Area Under the Receiver Operating Characteristic (ROC) curve, a performance metric for binary classification problems.
- **accuracy_score**: A function from Scikit-learn that calculates the accuracy of a classification model.
- **classification_report**: A function from Scikit-learn that builds a text report showing the main classification metrics.

### Warnings

- **warnings**: A Python module for issuing warning messages that custom handlers can filter, ignore, or capture.
  - `warnings.filterwarnings('ignore')`: A method to suppress all warning messages.

## Data Acquisition and Exploration

This project will use the IEEE-CIS Fraud Detection dataset to build a fraud detection model. This dataset contains two separate data files for transactional and identity information, which must be combined for analysis. Below, we describe the loading process and explore the dataset to understand its structure and characteristics.

### Loading the Data

First, we load the transactional and identity data for the training set, and then merge them into a single DataFrame:

```python
# Loading train_transaction data
train_transaction = pd.read_csv('train_transaction.csv')
train_transaction.shape

# Loading train_identity data
train_identity = pd.read_csv('train_identity.csv')
train_identity.shape

train_df = pd.merge(train_transaction, train_identity, how='left')
train_df.shape
```

After merging, we delete the original DataFrames to free up memory:

```python
del train_transaction, train_identity
```

We then store the length of the training DataFrame for future use:

```python
len_train_df = len(train_df)
```

Next, we perform the same process for the test set:

```python
# Loading test data
test_transaction = pd.read_csv('test_transaction.csv')
test_transaction.shape

test_identity = pd.read_csv('test_identity.csv')
test_identity.shape

test_df = pd.merge(test_transaction, test_identity, how='left')
```

To ensure consistency between the training and test sets, we rename the test DataFrame columns to match those of the training DataFrame, excluding the target variable 'isFraud':

```python
test_df.columns = train_df.drop('isFraud', axis=1).columns
test_df.shape
```

Finally, we delete the original test DataFrames to free up memory:

```python
del test_transaction, test_identity
```

### Data Exploration

We can explore the dataset with the data loaded and merged to gain insights into its structure, variables, and data types. Here are some initial steps to take:

- Examine the first few rows of the training and test DataFrames using the `head()` method.

- Use the `info()` method to obtain an overview of the variables, their data types, and the number of non-null values.

- Investigate the summary statistics for numerical variables using the `describe()` method.

- Identify the proportion of missing values in each column and decide how to handle them.
- Visualize the distribution of the target variable 'isFraud' to assess the class imbalance.
- Analyze the relationships between variables using correlation analysis, scatterplots, and other visualization techniques.

By thoroughly exploring the dataset, we can gain a deeper understanding of the data and identify any potential issues that may need to be addressed during the data cleaning and preprocessing stages.

## Data Cleaning and Preprocessing

In this section, we will clean and preprocess the data to address issues such as duplicates, missing values, and class imbalance and prepare the data for machine learning.

### Duplicates and Class Imbalance Check

First, we check for duplicates in the training data and visualize the class imbalance using a pie chart:

```python
# Duplicates check in train data
train_df.duplicated().sum()

# Class imbalance check
plt.pie(train_df.isFraud.value_counts(), labels=['Not Fraud', 'Fraud'], autopct='%0.1f%%')
plt.axis('equal')
plt.show()
```

![download (1)](C:\Users\vip phone\Desktop\UpWork\Analytical Neuron\download (1).png)

### Timestamp Comparison

We compare the transaction timestamps for the train and test datasets to ensure they do not overlap:

```python
# Timestamp of train and test data
plt.figure(figsize=(8, 4))
plt.hist(train_df['TransactionDT'], label='Train')
plt.hist(test_df['TransactionDT'], label='Test')
plt.ylabel('Count')
plt.title('Transaction Timestamp')
plt.legend()
plt.tight_layout()
plt.show()
```

![download](C:\Users\vip phone\Desktop\UpWork\Analytical Neuron\download.png)

### Handling Missing Values

We concatenate the train and test datasets (excluding the target variable 'isFraud' and 'TransactionID'), and check for missing values:

```python
# Missing values check
combined_df = pd.concat([train_df.drop(columns=['isFraud', 'TransactionID']), test_df.drop(columns='TransactionID')])
print(combined_df.shape)

# Dependent variable
y = train_df['isFraud']
print(y.shape)
```

Next, we drop columns with more than 20% missing values:

```python
# Dropping columns with more than 20% missing values 
mv = combined_df.isnull().sum()/len(combined_df)
combined_mv_df = combined_df.drop(columns=mv[mv>0.2].index)
del combined_df, train_df, test_df
print(combined_mv_df.shape)
```

We then separate the numerical and categorical data:

```python
# Filtering numerical data
num_mv_df = combined_mv_df.select_dtypes(include=np.number)
num_mv_df.shape

# Filtering categorical data
cat_mv_df = combined_mv_df.select_dtypes(exclude=np.number)
cat_mv_df.shape
```

For numerical columns, we fill in the missing values with the median:

```python
# Filling missing values by median for numerical columns 
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
num_df = pd.DataFrame(imp_median.fit_transform(num_mv_df), columns=num_mv_df.columns)
num_df.shape
```

For categorical columns, we fill in the missing values with the most frequent value:

```python
# Filling missing values by the most frequent value for categorical columns
imp_max = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
cat_df = pd.DataFrame(imp_max.fit_transform(cat_mv_df), columns=cat_mv_df.columns)
cat_df.shape
```

We then concatenate the cleaned numerical and categorical data:

```python
# Concatinating numerical and categorical data
combined_df_cleaned = pd.concat([num_df, cat_df], axis=1)

# Verifying missing values
print(f'Total missing values: {combined_df_cleaned.isnull().sum().sum()}')
combined_df_cleaned.shape
```

### One-hot Encoding

We apply one-hot encoding to the categorical columns:

```python
# One-hot encoding
combined_df_encoded = pd.get_dummies(combined_df_cleaned, drop_first=True)
print(combined_df_encoded.shape)

del combined_df_cleaned
```

We then separate the train and test data:

```python
# Separating train and test data
X = combined_df_encoded.iloc[:len_train_df]
```

```python
print(X.shape)
test = combined_df_encoded.iloc[len_train_df:]
print(test.shape)

del combined_df_encoded
```

### Time-based Train-Validation Split

We perform a time-based train-validation split, with 20% of the data in the validation set:

```python
# Time-based train validation splitting with 20% data in validation set

train = pd.concat([X, y], axis=1)
train.sort_values('TransactionDT', inplace=True)
X = train.drop(['isFraud'], axis=1)
y = train['isFraud']
splitting_index = int(0.8*len(X))
X_train = X.iloc[:splitting_index].values
X_val = X.iloc[splitting_index:].values
y_train = y.iloc[:splitting_index].values
y_val = y.iloc[splitting_index:].values
test = test.values
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

del y, train
```

### Standardization

We standardize the data by transforming it to have zero mean and unit variance:

```python
# Standardization

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_scaled = scaler.transform(test)

del X_train, X_val, test
```

### Addressing Class Imbalance with SMOTE

We check for class imbalance in the training data and apply SMOTE to address it by oversampling:

```python
# Class imbalance check
pd.value_counts(y_train)

# Applying SMOTE to deal with the class imbalance by oversampling
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print(X_train_smote.shape, y_train_smote.shape)

del X_train_scaled, y_train
pd.value_counts(y_train_smote)
```

We can build and evaluate various models to detect fraudulent transactions with the data cleaned, preprocessed, and prepared for machine learning.

## Machine Learning Model Selection and Implementation

This section will implement three machine-learning models: Random Forest, Logistic Regression, and XGBoost. We will evaluate their performance on the validation set and compare their results to select the best model for fraud detection.

### Random Forest Classifier

We start by training a Random Forest Classifier with specified hyperparameters:

```python
# Random Forest Classifier
rfc = RandomForestClassifier(criterion='entropy', max_features='sqrt', max_samples=0.5, min_samples_split=80)
rfc.fit(X_train_smote, y_train_smote)
y_predproba = rfc.predict_proba(X_val_scaled)

print(f'Validation AUC={roc_auc_score(y_val, y_predproba[:, 1])}')

y_pred = rfc.predict(X_val_scaled)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))
```

### Logistic Regression Classifier

Next, we train a Logistic Regression Classifier:

```python
# Logistic Regression Classifier
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train_smote, y_train_smote)

y_predproba = logistic_regression_model.predict_proba(X_val_scaled)
print(f'Validation AUC={roc_auc_score(y_val, y_predproba[:, 1])}')

y_pred = logistic_regression_model.predict(X_val_scaled)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))
```

### XGBoost Classifier

Lastly, we train an XGBoost Classifier:

```python
# XGBoost Classifier
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train_smote, y_train_smote)

y_predproba = xgb_model.predict_proba(X_val_scaled)
print(f'Validation AUC={roc_auc_score(y_val, y_predproba[:, 1])}')

y_pred = xgb_model.predict(X_val_scaled)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))
```

After implementing and evaluating the three models, we can compare their performance on the validation set. We can select the best model for fraud detection in the given dataset based on metrics such as AUC, accuracy, and classification report.

## Interpretation

Based on the evaluation metrics of the three machine learning models, we can interpret their performance as follows:

### Random Forest Classifier

- Validation AUC: 0.8866
- Accuracy: 0.9665
- Classification Report:
  - Precision for non-fraud: 0.98
  - Recall for non-fraud: 0.99
  - F1-score for non-fraud: 0.98
  - Precision for fraud: 0.52
  - Recall for fraud: 0.43
  - F1-score for fraud: 0.47

### Logistic Regression Classifier

- Validation AUC: 0.8128
- Accuracy: 0.8316
- Classification Report:
  - Precision for non-fraud: 0.99
  - Recall for non-fraud: 0.84
  - F1-score for non-fraud: 0.91
  - Precision for fraud: 0.09
  - Recall for fraud: 0.61
  - F1-score for fraud: 0.15

### XGBoost Classifier

- Validation AUC: 0.8875
- Accuracy: 0.9838
- Classification Report:
  - Precision for non-fraud: 0.98
  - Recall for non-fraud: 1.00
  - F1-score for non-fraud: 0.99
  - Precision for fraud: 0.89
  - Recall for fraud: 0.38
  - F1-score for fraud: 0.54

Based on the evaluation metrics, the XGBoost Classifier has the highest AUC (**0.8875**) and accuracy (**0.9838**) and the best performance in precision and F1-score for fraud detection. Although the recall for fraud detection is higher in the Logistic Regression Classifier, the overall performance of the XGBoost Classifier is better than the other two models.

In conclusion, the XGBoost Classifier is the best model for fraud detection in the given dataset, considering its performance on various metrics.
