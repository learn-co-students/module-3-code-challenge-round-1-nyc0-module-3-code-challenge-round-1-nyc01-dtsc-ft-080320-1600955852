
# Module 3 Code Challenge

## Overview

This assessment is designed to test your understanding of Module 3 material. It covers:

* Gradient Descent
* Logistic Regression
* Classification Metrics
* Decision Trees

_Read the instructions carefully._ You will be asked both to write code and respond to a few short answer questions.

### Note on the short answer questions

For the short answer questions, _please use your own words._ The expectation is that you have **not** copied and pasted from an external source, even if you consult another source to help craft your response. While the short answer questions are not necessarily being assessed on grammatical correctness or sentence structure, do your best to communicate yourself clearly.

---
## Part 1: Gradient Descent [Suggested Time: 20 min]
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

### 1.1) What is a more generalized name for the RSS curve above? How could a machine learning model use this curve?


```python
"""
Your written answer here
"""
```

### 1.2) Would you rather choose a $m$ value of 0.08 or 0.05 from the RSS curve up above? Explain your reasoning.


```python
"""
Your written answer here
"""
```

![](visuals/gd.png)

### 1.3) Using the gradient descent visual from above, explain why the distance between estimates in each step is getting smaller as more steps occur with gradient descent.


```python
"""
Your written answer here
"""
```

### 1.4) What does the learning rate do in the gradient descent algorithm? Explain how a very small and a very large learning rate would affect the gradient descent.


```python
"""
Your written answer here
"""
```

---
## Part 2: Logistic Regression [Suggested Time: 15 min]
---

### 2.1) Why is logistic regression typically better than linear regression for modeling a binary target/outcome?


```python
"""
Your written answer here
"""
```

### 2.2) Compare logistic regression to another classification model of your choice (e.g. KNN, Decision Tree, etc.). What is one advantage and one disadvantage it has when compared with the other model?


```python
"""
Your written answer here
"""
```

---
## Part 3: Classification Metrics [Suggested Time: 20 min]
---

![cnf matrix](visuals/cnf_matrix.png)

### 3.1) Using the confusion matrix above, calculate precision, recall, and F-1 score.

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

### 3.2) Which ROC curve from the above graph is the best? Explain your reasoning. 

Note: each ROC curve represents one model, each labeled with the feature(s) inside each model.


```python
"""
Your written answer here
"""
```

### Logistic Regression Example

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
print(f'The classifier has an accuracy score of {round(accuracy_score(y_test, y_test_pred), 3)}.')
```

### 3.3) Explain how the distribution of `y` shown below could explain the very high accuracy score.


```python
# Run this cell without changes

y.value_counts()
```


```python
"""
Your written answer here
"""
```

### 3.4) What is one method you could use to improve your model to address the issue discovered in Question 3.3?


```python
"""
Your written answer here
"""
```

---
## Part 4: Decision Trees [Suggested Time: 20 min]
---

### Concepts 
You're given a dataset of **30** elements, 15 of which belong to a positive class (denoted by `+` ) and 15 of which do not (denoted by `-`). These elements are described by two attributes, A and B, that can each have either one of two values, true or false. 

The diagrams below show the result of splitting the dataset by attribute: the diagram on the left hand side shows that if we split by attribute A there are 13 items of the positive class and 2 of the negative class in one branch and 2 of the positive and 13 of the negative in the other branch. The right hand side shows that if we split the data by attribute B there are 8 items of the positive class and 7 of the negative class in one branch and 7 of the positive and 8 of the negative in the other branch.

<img src="visuals/decision_stump.png">

### 4.1) Which one of the two attributes resulted in the best split of the original data? How do you select the best attribute to split a tree at each node? 

It may be helpful to discuss splitting criteria.


```python
"""
Your written answer here
"""
```

### Decision Tree Example

In this section, you will use decision trees to fit a classification model to the wine dataset. The data is the results of a chemical analysis of wines grown in the same region in Italy by three different cultivators. There are thirteen different measurements taken for different constituents found in the three types of wine.


```python
# Run this cell without changes

# Relevant imports 
import pandas as pd 
import numpy as np 
from sklearn.datasets import load_wine

# Load the data 
wine = load_wine()
X, y = load_wine(return_X_y=True)
X = pd.DataFrame(X, columns=wine.feature_names)
y = pd.Series(y)
y.name = 'target'
df = pd.concat([X, y.to_frame()], axis=1)
df.head()
```


```python
# Run this cell without changes
# Get the shape of the DataFrame 
df.shape
```


```python
# Run this cell without changes
# Get the distribution of the target variable 
y.value_counts()
```

### 4.2) Split the data into training and test sets. Create training and test sets with `test_size=0.5` and `random_state=1`.


```python
# Replace None with appropriate code  

X_train, X_test, y_train, y_test = None
```

### 4.3) Fit a decision tree model with scikit-learn to the training data. Use parameter defaults and `random_state=1` for this model. Then use the fitted classifier to generate predictions for the test data.

You can use the Scikit-learn DecisionTreeClassifier (docs [here](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html))


```python
# Your code here 
```

### 4.4) Obtain the accuracy score of the predictions on the test set. 

You can use the `sklearn.metrics` module.


```python
# Your code imports here

# Replace None with appropriate code 

print('Accuracy Score:', None)
```

### 4.5) Produce a confusion matrix for the predictions on the test set. 

You can use the `sklearn.metrics` module.


```python
# Your code imports here

# Your code here 
```

### 4.6) Based on the accuracy score and confusion matrix, does the model seem to be performing well or to have substantial performance issues? Explain your answer.


```python
"""
Your written answer here
"""
```
