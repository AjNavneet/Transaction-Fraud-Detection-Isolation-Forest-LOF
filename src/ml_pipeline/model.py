# Importing required libraries
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from numpy import quantile, where, random
import pickle

# Function to train the Isolation Forest (IF) model
def train_IF(data):
    # Create an Isolation Forest model with specified hyperparameters
    clf = IsolationForest(n_estimators=500, max_samples=len(data), contamination=0.0018)
    clf.fit(data)
    return clf

# Function to predict scores using the trained model
def predict_scores(model, data):
    # Use the model to predict decision scores for the given data
    scores_prediction = model.decision_function(data)
    return scores_prediction

# Function to train the Local Outlier Factor (LOF) model
def train_lof(data):
    # Create a LOF model with specified hyperparameters
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.0018)
    lof.fit_predict(data)
    return lof

# Function to calculate anomaly scores based on the model
def anomaly_scores(model):
    # Get the negative outlier factor scores from the LOF model
    anomaly_scores = model.negative_outlier_factor_
    return anomaly_scores

# Function to save the trained model to a specified path
def save_model(model, framework, model_path):
    if framework == "IF":
        # Save the Isolation Forest model to a pickle file
        model_path += '/IF_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        # Save the Local Outlier Factor model to a pickle file
        model_path += '/LOF_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    print('Model saved at:', model_path)
    return model
