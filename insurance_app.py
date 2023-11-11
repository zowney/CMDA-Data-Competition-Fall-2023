# Handles warnings thrown by sklearn
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Imports

from generalMethods import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from flask import Flask, render_template, request

dataset = pd.read_csv('raw.csv')

# Separate features (X) and labels (y)
X = dataset[['age','number_of_vehicles_involved','witnesses', 'bodily_injuries','injury_claim', 'property_claim','vehicle_claim', 'incident_severity']]
y = dataset['fraud_reported']

# Identify categorical columns for encoding
categorical_columns = ['incident_severity']

# One-hot encode categorical columns
encoder = OneHotEncoder(drop='first', sparse=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_columns]), columns=encoder.get_feature_names_out(categorical_columns))
X = pd.concat([X, X_encoded], axis=1)
X = X.drop(categorical_columns, axis=1)

# Handling Imbalanced Data
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Split the resampled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.5, random_state=42)

# Hyperparameter Tuning 
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Train the model with the best hyperparameters
best_clf = grid_search.best_estimator_
best_clf.fit(X_train, y_train)


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def calculate_sum():
    chanceOfFraud = None  # Initialize the total to None

    if request.method == 'POST':
        try:
            AGE = int(request.form['Age'])
            NVI = int(request.form['Number Of Vehicles Involved'])
            WIT = int(request.form['Witnesses'])
            BDI = int(request.form['Bodily Injuries'])
            IJC = int(request.form['Injury Claim'])
            PRC = int(request.form['Property Claim'])
            VHC = int(request.form['Vehicle Claim'])
            INS = str(request.form['Incident Severity'])
            INS = INS.upper()
            if(INS == 'TOTAL'):
                INS = 'Total Loss'
            elif(INS == 'MAJOR'):
                INS = 'Major Damage'
            elif(INS == 'MINOR'):
                INS = 'Minor Damage'
            elif(INS == 'TRIVIAL'):
                INS = 'Trivial Damage'
            incident_severity_encoded = encoder.transform([[INS]])
            features  = [AGE, NVI, WIT, BDI, IJC, PRC, VHC] + list(incident_severity_encoded[0])
            
            # Converts the encoded incident severity into a 2D Array for processing.
            features_flat = np.array(features).reshape(1, -1)
            features_flat = features_flat.flatten()

            fraudChance = best_clf.predict(features_flat.reshape(1, -1))
            chanceOfFraud = fraudChance[0]
            
        except ValueError:
            return "Invalid input. Please Make Sure you are Inputting Proper Values. Go back one Webpage!"

    return render_template('index.html', total=chanceOfFraud)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')