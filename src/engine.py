# Importing required packages
import numpy as np
import pandas as pd
import pickle

# Importing the Loader for reading YAML
from yaml import CLoader as Loader, load

# Import custom utility functions and classes
from ml_pipeline.utils import read_data_csv, read_config, find_contamination
from ml_pipeline.preprocessing import handle_null_values
from ml_pipeline.model import train_IF, train_lof, predict_scores, anomaly_scores, save_model

# Reading the configuration file
config = read_config("modular_code/input/config.yaml")

# Reading the data from CSV
transaction_data = read_data_csv(config['data_path'])

# Handling missing values in the dataset
transaction_data = handle_null_values(transaction_data)

# Calculate the contamination score for anomaly detection
contamination_score = find_contamination('Class', transaction_data)

# Separating features (X) and target variable (y)
X = transaction_data.drop('Class', axis=1)
y = transaction_data['Class']

# Training the Isolation Forest (IF) model
clf = train_IF(X)
print("Isolation Forest model trained successfully")

# Predicting isolation forest scores
scores_prediction = predict_scores(clf, X)

# Adding the scores as a new column in the dataset
transaction_data['scores'] = scores_prediction

# Saving the Isolation Forest model
save_model(clf, "IF", config['model_path'])

# Training the Local Outlier Factor (LOF) model
lof = train_lof(X)
print("LOF model trained successfully")

# Finding anomaly scores using the LOF model
anomaly_scores = anomaly_scores(lof)
print("Anomaly scores for the LOF model:", anomaly_scores)

# Saving the LOF model
save_model(lof, "LOF", config['model_path'])
