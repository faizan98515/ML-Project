import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('adult_income_dataset.csv')

# Handling missing values by imputing with the mode
for column in ['workclass', 'occupation', 'native-country']:
    dataset[column].fillna(dataset[column].mode()[0], inplace=True)

#Splitting the data into features and target variable
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


#Encoding the Categorical variables 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,9] = le.fit_transform(X[:,9])
# Set sparse_output=False and handle_unknown='ignore'
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), [1,3,5,6,7,8,13])], remainder='passthrough')  
X = np.array(ct.fit_transform(X))

# Clean the target variable
dataset['income'] = dataset['income'].str.strip().str.replace('.', '', regex=False)

# Encode the cleaned target variable
le_target = LabelEncoder()
y = le_target.fit_transform(dataset['income'])

#Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying Grid Search to optimize SVM parameters
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [3, 4, 5]  # Relevant only for polynomial kernel
}

# Initialize SVM Classifier
svm = SVC(random_state=0)

# Perform Grid Search
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters from Grid Search
print("Best Grid Search Parameters:", grid_search.best_params_)

# Applying Random Search to optimize SVM parameters
from sklearn.model_selection import RandomizedSearchCV

# Define parameter distribution
param_dist = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [3, 4, 5]  # Relevant only for polynomial kernel
}

# Perform Randomized Search
random_search = RandomizedSearchCV(estimator=svm, param_distributions=param_dist, scoring='accuracy', cv=5, n_iter=20, verbose=1, n_jobs=-1, random_state=0)
random_search.fit(X_train, y_train)

# Best parameters from Random Search
print("Best Random Search Parameters:", random_search.best_params_)

# Evaluate the best model from Grid Search
best_model_grid = grid_search.best_estimator_
y_pred_grid = best_model_grid.predict(X_test)

# Evaluate the best model from Random Search
best_model_random = random_search.best_estimator_
y_pred_random = best_model_random.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Predictions on the test set
y_pred = best_model_random.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# F1-Score
f1 = f1_score(y_test, y_pred)
print("F1-Score:", f1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Detailed Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Bar chart for evaluaiton metrics 
import matplotlib.pyplot as plt

# Metrics and their values
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]

# Plot the evaluation metrics
plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=['skyblue', 'orange', 'green', 'red'])
plt.title('Evaluation Metrics for the Best Model', fontsize=16)
plt.ylabel('Scores', fontsize=12)
plt.xlabel('Metrics', fontsize=12)
plt.ylim(0, 1)
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
plt.show()

#Confusion metrics
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le_target.classes_, yticklabels=le_target.classes_)
plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.show()