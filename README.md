# Heart Disease Predictor

This project implements a machine learning model to predict whether a person has heart disease based on various medical features. The dataset used in this project contains attributes such as age, sex, cholesterol levels, blood pressure, etc., to classify the target variable (`target`) into two classes: heart disease or no heart disease.

## Table of Contents

- [Libraries and Tools](#libraries-and-tools)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
  - [Handling Missing Values](#handling-missing-values)
  - [Handling Duplicates](#handling-duplicates)
  - [Categorical and Continuous Variables](#categorical-and-continuous-variables)
  - [Encoding Categorical Data](#encoding-categorical-data)
  - [Feature Scaling](#feature-scaling)
- [Model Training](#model-training)
  - [Logistic Regression](#logistic-regression)
  - [Support Vector Machine](#support-vector-machine)
  - [K-Nearest Neighbors](#k-nearest-neighbors)
  - [Decision Tree Classifier](#decision-tree-classifier)
  - [Random Forest Classifier](#random-forest-classifier)
  - [Gradient Boosting Classifier](#gradient-boosting-classifier)
- [Model Evaluation](#model-evaluation)
- [Prediction on New Data](#prediction-on-new-data)
- [GUI Implementation](#gui-implementation)
- [Saving and Loading the Model](#saving-and-loading-the-model)

---

## Libraries and Tools

The following libraries were used in this project:

- **pandas**: For data manipulation and analysis.
- **matplotlib**: For data visualization.
- **scikit-learn**: For machine learning models and preprocessing.
- **joblib**: For saving and loading the trained model.
- **tkinter**: For the GUI implementation.

---

## Dataset

The dataset used is the **Heart Disease UCI dataset**, which contains various medical parameters to predict whether a person has heart disease. You can download the dataset from [Kaggle's heart disease dataset](https://www.kaggle.com/).

---

## Data Preprocessing

### Handling Missing Values

The dataset is checked for missing values, and it is found that there are no missing values in any columns:

```python
data.isnull().sum()
```

### Handling Duplicates

Duplicate rows are removed from the dataset to ensure the quality of the data:

```python
data_dup = data.duplicated().any()
data = data.drop_duplicates()
```

### Categorical and Continuous Variables

We classify variables into categorical and continuous types based on the number of unique values they have:

```python
cate_val = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
cont_val = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
```

### Encoding Categorical Data

Categorical features are encoded using **one-hot encoding** to convert them into numeric values:

```python
data = pd.get_dummies(data, columns=cate_val, drop_first=True)
```

### Feature Scaling

Continuous features are scaled using the **StandardScaler**:

```python
from sklearn.preprocessing import StandardScaler
st = StandardScaler()
data[cont_val] = st.fit_transform(data[cont_val])
```

---

## Model Training

Several machine learning algorithms were used to predict the target variable:

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train, y_train)
```

### Support Vector Machine (SVM)

```python
from sklearn import svm
svm = svm.SVC()
svm.fit(X_train, y_train)
```

### K-Nearest Neighbors (KNN)

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
```

### Decision Tree Classifier

```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
```

### Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
```

### Gradient Boosting Classifier

```python
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
```

---

## Model Evaluation

The models are evaluated based on their accuracy:

```python
final_data = pd.DataFrame({'Models': ['LR', 'SVM', 'KNN', 'DT', 'RF', 'GB'],
                          'ACC': [accuracy_score(y_test, y_pred1) * 100,
                                  accuracy_score(y_test, y_pred2) * 100,
                                  accuracy_score(y_test, y_pred3) * 100,
                                  accuracy_score(y_test, y_pred4) * 100,
                                  accuracy_score(y_test, y_pred5) * 100,
                                  accuracy_score(y_test, y_pred6) * 100]})
```

---

## Prediction on New Data

A trained **Random Forest model** is used to make predictions on new data:

```python
new_data = pd.DataFrame({
    'age': 52,
    'sex': 1,
    'cp': 0,
    'trestbps': 125,
    'chol': 212,
    'fbs': 0,
    'restecg': 1,
    'thalach': 168,
    'exang': 0,
    'oldpeak': 1.0,
    'slope': 2,
    'ca': 2,
    'thal': 3
}, index=[0])

p = rf.predict(new_data)
```

---

## GUI Implementation

A **GUI (Graphical User Interface)** is implemented using **Tkinter** to predict heart disease based on user inputs. The system takes input values for different features and returns whether a person has heart disease or not.

```python
master = Tk()
master.title("Heart Disease Prediction System")

# Code for GUI creation and interaction
```

---

## Saving and Loading the Model

After training the model, it is saved to disk using **joblib** for future predictions:

```python
import joblib
joblib.dump(rf, 'model_joblib_heart')
```

To load the saved model for making predictions on new data:

```python
model = joblib.load('model_joblib_heart')
model.predict(new_data)
```

---

## Conclusion

This project demonstrates the use of various machine learning algorithms to predict heart disease based on medical parameters. A user-friendly GUI is also created for easy interaction with the model. The **Random Forest** model achieved the best performance with an accuracy of **81.97%**.

---

## Requirements

To run this project, make sure to install the necessary libraries:

```bash
pip install pandas scikit-learn matplotlib seaborn joblib tkinter
```

---
