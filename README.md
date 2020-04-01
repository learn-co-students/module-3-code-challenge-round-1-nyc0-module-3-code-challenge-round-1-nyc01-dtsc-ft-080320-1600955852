
# Module 3 Code Challenge

## Overview

This assessment is designed to test your understanding of the Module 3 materials. It covers:

* Calculus, Cost Function, and Gradient Descent
* Logistic Regression
* Decision Trees
* Ensemble Methods 

_Read the instructions carefully._ You will be asked both to write code and respond to a few short answer questions.

### Note on the short answer questions

For the short answer questions, _please use your own words._ The expectation is that you have **not** copied and pasted from an external source, even if you consult another source to help craft your response. While the short answer questions are not necessarily being assessed on grammatical correctness or sentence structure, do your best to communicate yourself clearly.

---
## Part 1: Calculus, Cost Function, and Gradient Descent [Suggested Time: 25 min]
---

![best fit line](visuals/best_fit_line.png)

The best fit line that goes through the scatterplot up above can be generalized in the following equation: $$y = mx + b$$

Of all the possible lines, we can prove why that particular line was chosen using the plot down below:

![](visuals/cost_curve.png)

where RSS is defined as the residual sum of squares:

$$ 
\begin{align}
RSS &= \sum_{i=1}^n(actual - expected)^2 \\
&= \sum_{i=1}^n(y_i - \hat{y})^2 \\
&= \sum_{i=1}^n(y_i - (mx_i + b))^2
\end{align}
$$ 

### 1.1) What is a more generalized name for the RSS curve above? How is it related to machine learning models?


```python
"""
Your written answer here
"""
```

### 1.2) Would you rather choose a $m$ value of 0.08 or 0.05 from the RSS curve up above?   What is the relation between the position on the cost curve, the error, and the slope of the line?


```python
"""
Your written answer here
"""
```

![](visuals/gd.png)

### 1.3) Using the gradient descent visual from above, explain why the distance between each step is getting smaller as more steps occur with gradient descent.


```python
"""
Your written answer here
"""
```

### 1.4) What is the purpose of a learning rate in gradient descent? Explain how a very small and a very large learning rate would affect the gradient descent.


```python
"""
Your written answer here
"""
```

---
## Part 2: Logistic Regression [Suggested Time: 25 min]
---

![cnf matrix](visuals/cnf_matrix.png)

### 2.1) Using the confusion matrix above, calculate precision, recall, and F-1 score.

Show your work, not just your final numeric answer


```python
# Your code here to calculate precision
```


```python
# Your code here to calculate recall
```


```python
# Your code here to calculate F-1 score
```

<img src = "visuals/many_roc.png" width = "700">

### 2.2) Pick the best ROC curve from the above graph and explain your choice. 

Note: each ROC curve represents one model, each labeled with the feature(s) inside each model.


```python
"""
Your written answer here
"""
```

The following cell includes code to train and evaluate a model


```python
# Run this cell without changes

# Include relevant imports
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score

network_df = pickle.load(open('write_data/sample_network_data.pkl', 'rb'))

# partion features and target 
X = network_df.drop('Purchased', axis=1)
y = network_df['Purchased']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2019)

# scale features
scale = StandardScaler()
scale.fit(X_train)
X_train = scale.transform(X_train)
X_test = scale.transform(X_test)

# build classifier
model = LogisticRegression(C=1e5, solver='lbfgs')
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)

# get the accuracy score
print(f'The original classifier has an accuracy score of {round(accuracy_score(y_test, y_test_pred), 3)}.')

# get the area under the curve from an ROC curve
y_score = model.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, y_score)
auc = round(roc_auc_score(y_test, y_score), 3)
print(f'The original classifier has an area under the ROC curve of {auc}.')
```

### 2.3) The model above has an accuracy score that might be too good to believe. Using `y.value_counts()`, explain how `y` is affecting the accuracy score.


```python
# Run this cell without changes

y.value_counts()
```


```python
"""
Your written answer here
"""
```

### 2.4) What methods would you use to address the issues in Question 2.3? 


```python
"""
Your written answer here
"""
```

---
## Part 3: Decision Trees [Suggested Time: 15 min]
---

### Concepts 
You're given a dataset of **30** elements, 15 of which belong to a positive class (denoted by *`+`* ) and 15 of which do not (denoted by `-`). These elements are described by two attributes, A and B, that can each have either one of two values, true or false. 

The diagrams below show the result of splitting the dataset by attribute: the diagram on the left hand side shows that if we split by attribute A there are 13 items of the positive class and 2 of the negative class in one branch and 2 of the positive and 13 of the negative in the other branch. The right hand side shows that if we split the data by attribute B there are 8 items of the positive class and 7 of the negative class in one branch and 7 of the positive and 8 of the negative in the other branch.

<img src="visuals/decision_stump.png">

### 3.1) Which one of the two attributes resulted in the best split of the original data? How do you select the best attribute to split a tree at each node? 

It may be helpful to discuss splitting criteria.


```python
"""
Your written answer here
"""
```

### Decision Trees for Regression 

In this section, you will use decision trees to fit a regression model to the Combined Cycle Power Plant dataset. 

This dataset is from the UCI ML Dataset Repository, and has been included in the `raw_data` folder of this repository as an Excel `.xlsx` file, `'Folds5x2_pp.xlsx'`. 

The features of this dataset consist of hourly average ambient variables taken from various sensors located around a power plant that record the ambient variables every second.  
- Temperature (AT) 
- Ambient Pressure (AP) 
- Relative Humidity (RH)
- Exhaust Vacuum (V) 

The target to predict is the net hourly electrical energy output (PE). 

The features and target variables are not normalized.

In the cells below, we import `pandas` and `numpy` for you, and we load the data into a pandas DataFrame. We also include code to inspect the first five rows and get the shape of the DataFrame.


```python
# Run this cell without changes

import pandas as pd 
import numpy as np 

# Load the data
filename = 'raw_data/Folds5x2_pp.xlsx'
df = pd.read_excel(filename)
```


```python
# Run this cell without changes
# Inspect the first five rows of the DataFrame
df.head()
```


```python
# Run this cell without changes
# Get the shape of the DataFrame 
df.shape
```

Before fitting any models, you need to create training and test splits for the data.

Below, we split the data into features and target (`'PE'`) for you. 


```python
# Run this cell without changes

X = df.drop(columns=['PE'], axis=1)
y = df['PE']
```

### 3.2) Split the data into training and test sets. Create training and test sets with `test_size=0.5` and `random_state=1`.


```python
# Replace None with appropriate code  

X_train, X_test, y_train, y_test = None
```

### 3.3) Fit a vanilla decision tree regression model with scikit-learn to the training data. Set `random_state=1` for reproducibility. Evaluate the model on the test data.

For the rest of this section feel free to refer to the scikit-learn documentation on [decision tree regressors](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html). 


```python
# Your code here 
```

### 3.4) Obtain the mean squared error, mean absolute error, and coefficient of determination (r2 score) of the predictions on the test set. 

You can use the `sklearn.metrics` module.


```python
# Your code imports here


# Replace None with appropriate code 

print('Mean Squared Error:', None)
print('Mean Absolute Error:', None)
print('R-squared:', None)
```

Hint: MSE should be about 22.21 

### Hyperparameter Tuning of Decision Trees for Regression

### 3.5) Add hyperparameters to a new decision tree and fit it to our training data. Evaluate the model with the test data.


```python
# Your code here 
```

### 3.6) Obtain the mean squared error, mean absolute error, and coefficient of determination (r2 score) of the predictions on the test set. Did this improve your previous model? (It's ok if it didn't)


```python
# Your code here
```


```python
"""
Your written answer here
"""
```

---
## Part 4: Ensemble Methods [Suggested Time: 10 min]
---

### Random Forests and Hyperparameter Tuning using GridSearchCV

In this section, you will perform hyperparameter tuning for a Random Forest classifier using GridSearchCV. You will use scikit-learn's wine dataset to classify wines into one of three different classes. 

After finding the best estimator, you will interpret the best model's feature importances. 

In the cells below, we have loaded the relevant imports and the wine data for you. 


```python
# Run this cell without changes

# Relevant imports 
from sklearn.datasets import load_wine

# Load the data 
wine = load_wine()
X, y = load_wine(return_X_y=True)
X = pd.DataFrame(X, columns=wine.feature_names)
y = pd.Series(y)
y.name = 'target'
df = pd.concat([X, y.to_frame()], axis=1)
```

In the cells below, we inspect the first five rows of the DataFrame and compute the DataFrame's shape.


```python
# Run this cell without changes
# Inspect the first five rows of the DataFrame
df.head()
```


```python
# Run this cell without changes
# Get the shape of the DataFrame 
df.shape
```

We also get descriptive statistics for the dataset features, and obtain the distribution of classes in the dataset. 


```python
# Run this cell without changes
# Get descriptive statistics for the features
X.describe()
```


```python
# Run this cell without changes
# Obtain distribution of classes
y.value_counts().sort_index()
```

You will now perform hyperparameter tuning for a Random Forest classifier.

In the cell below, we include the relevant imports for you.


```python
# Run this cell without changes

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
```

### 4.1) Create an instance of a Random Forest classifier estimator. Call the instance `rfc`. 

Make sure to set `random_state=42` for reproducibility. 


```python
# Replace None with appropriate code
rfc = None
```

### 4.2) Construct a `param_grid` dictionary to pass to `GridSearchCV` when instantiating the object. 

Choose at least three hyperparameters to tune, and at least three values for each.


```python
# Replace None with appropriate code 
param_grid = None
```

Now that you have created the `param_grid` dictionary of hyperparameters, let's continue performing hyperparameter optimization of a Random Forest Classifier. 

### 4.3) Create an instance of an `GridSearchCV` object and fit it to the data. Call the instance `cv_rfc`. 

- Use the random forest classification estimator you instantiated above, the parameter grid dictionary you constructed, and make sure to perform 5-fold cross validation. 
- The fitting process should take 10-15 seconds to complete. 


```python
# Replace None with appropriate code 
cv_rfc = None 

cv_rfc.fit(None, None)
```
