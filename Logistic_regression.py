#SP23-BAI-033
#SP23-BAI-035
#SP23-BAI-057


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


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Parameter grid for Grid Search
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],  # Only use solvers compatible with 'l1' and 'l2'
    'solver': ['liblinear', 'saga']  # Compatible solvers
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=LogisticRegression(random_state=0, max_iter=500), 
                           param_grid=param_grid, 
                           scoring='accuracy', 
                           cv=5, 
                           verbose=1)

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters and accuracy
print("Best Parameters (Grid Search):", grid_search.best_params_)
print("Best Accuracy (Grid Search):", grid_search.best_score_)

# Parameter grid for Randomized Search
param_dist = {
    'C': np.logspace(-3, 3, 100),  # Sample C values in a wide range
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'solver': ['liblinear', 'saga'],
    'l1_ratio': np.linspace(0, 1, 10)  # Only applicable if penalty='elasticnet'
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=LogisticRegression(random_state=0, max_iter=500), 
                                   param_distributions=param_dist, 
                                   n_iter=50,  # Number of random combinations
                                   scoring='accuracy', 
                                   cv=5, 
                                   verbose=1, 
                                   random_state=42)

# Fit the model
random_search.fit(X_train, y_train)

# Best parameters and accuracy
print("Best Parameters (Randomized Search):", random_search.best_params_)
print("Best Accuracy (Randomized Search):", random_search.best_score_)


# Evaluate the best model from Grid Search
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Test Set Accuracy (Grid Search):", accuracy_score(y_test, y_pred))

# Evaluate the best model from Randomized Search
best_model_random = random_search.best_estimator_
y_pred_random = best_model_random.predict(X_test)
print("Test Set Accuracy (Randomized Search):", accuracy_score(y_test, y_pred_random))


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







