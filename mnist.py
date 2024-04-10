import numpy as np
import tkinter as tk
from tkinter import messagebox
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Load the MNIST Handwritten Digits Dataset
digits = load_digits()
X = digits.data
y = digits.target

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train and evaluate model
def train_and_evaluate(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

# Function to perform hyperparameter optimization
def optimize_parameters(model, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_
    
    return best_params, best_estimator

# Function to display results
def display_results(results):
    result_str = "Model\tAccuracy\tPrecision\tRecall\tF1 Score\n"
    for result in results:
        result_str += f"{result['Model']}\t{result['Accuracy']:.4f}\t{result['Precision']:.4f}\t{result['Recall']:.4f}\t{result['F1 Score']:.4f}\n"
    messagebox.showinfo("Results", result_str)

# Function to classify digits using different models
def classify_digits():
    results = []
    
    # Logistic Regression
    logistic_regression = LogisticRegression(max_iter=10000)
    results.append(train_and_evaluate(logistic_regression, "Logistic Regression"))
    
    # Decision Trees
    decision_tree = DecisionTreeClassifier()
    results.append(train_and_evaluate(decision_tree, "Decision Trees"))
    
    # k-Nearest Neighbors
    knn = KNeighborsClassifier()
    results.append(train_and_evaluate(knn, "k-Nearest Neighbors"))
    
    # Support Vector Machines
    svm = SVC()
    results.append(train_and_evaluate(svm, "Support Vector Machines"))
    
    display_results(results)

# Function to optimize hyperparameters
def optimize_hyperparameters():
    results = []
    
    # Logistic Regression
    param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
    best_params_lr, _ = optimize_parameters(LogisticRegression(max_iter=10000), param_grid_lr)
    results.append({'Model': "Logistic Regression", 'Best Parameters': best_params_lr})
    
    # Decision Trees
    param_grid_dt = {'max_depth': [None, 5, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
    best_params_dt, _ = optimize_parameters(DecisionTreeClassifier(), param_grid_dt)
    results.append({'Model': "Decision Trees", 'Best Parameters': best_params_dt})
    
    # k-Nearest Neighbors
    param_grid_knn = {'n_neighbors': [3, 5, 10, 20], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}
    best_params_knn, _ = optimize_parameters(KNeighborsClassifier(), param_grid_knn)
    results.append({'Model': "k-Nearest Neighbors", 'Best Parameters': best_params_knn})
    
    # Support Vector Machines
    param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    best_params_svm, _ = optimize_parameters(SVC(), param_grid_svm)
    results.append({'Model': "Support Vector Machines", 'Best Parameters': best_params_svm})
    
    messagebox.showinfo("Best Hyperparameters", '\n'.join([f"{result['Model']}: {result['Best Parameters']}" for result in results]))

# Create GUI window
window = tk.Tk()
window.title("MNIST Handwritten Digits Classifier")

# Create buttons to classify digits and optimize hyperparameters
classify_button = tk.Button(window, text="Classify Digits", command=classify_digits)
classify_button.pack()

optimize_button = tk.Button(window, text="Optimize Hyperparameters", command=optimize_hyperparameters)
optimize_button.pack()

# Run the GUI
window.mainloop()
