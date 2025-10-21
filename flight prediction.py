
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer



# Load dataset
df = pd.read_csv(r"C:\Users\DELL\Downloads\Flight_Price.csv")

def preprocess_data(df):
    # Convert Date_of_Journey to datetime
    df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], errors='coerce')
    df['Journey_Day'] = df['Date_of_Journey'].dt.day
    df['Journey_Month'] = df['Date_of_Journey'].dt.month

    # Extract hours and minutes from Dep_Time and Arrival_Time
    df['Dep_hour'] = pd.to_datetime(df['Dep_Time']).dt.hour
    df['Dep_min'] = pd.to_datetime(df['Dep_Time']).dt.minute
    df['Arrival_hour'] = pd.to_datetime(df['Arrival_Time']).dt.hour
    df['Arrival_min'] = pd.to_datetime(df['Arrival_Time']).dt.minute

    # Convert Duration into minutes
    def duration_to_minutes(duration):
    # Remove spaces for consistency
     duration = duration.replace(" ", "")
    
     hours = 0
     minutes = 0

    # Extract hours if present
     if 'h' in duration:
        hours_part = duration.split('h')[0]
        hours = int(hours_part)
        duration = duration.split('h')[1]  # Keep the remaining part
    
     # Extract minutes if present
     if 'm' in duration:
        minutes_part = duration.split('m')[0]
        if minutes_part:  # Only convert if not empty
            minutes = int(minutes_part)

     duration = hours * 60 + minutes
     return duration
  
    df['Duration'] = (duration_to_minutes)(df['Duration'])
    # Handle Total_Stops
    df['Total_Stops'] = df['Total_Stops'].replace({'non-stop': 0,
                                                   '1 stop': 1,
                                                   '2 stops': 2,
                                                   '3 stops': 3,
                                                   '4 stops': 4})
   

    # Drop irrelevant columns
    df.drop(['Route', 'Dep_Time', 'Arrival_Time', 'Date_of_Journey'], axis=1, inplace=True)

    return df

def build_preprocessor(categorical_cols, numerical_cols):
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    return preprocessor


# Load dataset
df = pd.read_csv(r"C:\Users\DELL\Downloads\Flight_Price.csv")

# Clean and feature-engineer
df = preprocess_data(df)

# Define features and target
X = df.drop('Price', axis=1)
y = df['Price']

categorical_cols = [c for c in X.columns if X[c].dtype == 'object']
numerical_cols = [c for c in X.columns if X[c].dtype != 'object']

# Build preprocessor
preprocessor = build_preprocessor(categorical_cols, numerical_cols)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model pipeline
from sklearn.pipeline import Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
])

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save model
joblib.dump(model, 'model.pkl')
print(" Model saved successfully!")

import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.title("Flight Price Prediction App")

# Load model
model = joblib.load('model.pkl')

# Inputs
airline = st.selectbox("Airline", ["IndiGo", "Air India", "Jet Airways", "SpiceJet", "Vistara"])
source = st.selectbox("Source", ["Delhi", "Kolkata", "Mumbai", "Chennai", "Banglore"])
destination = st.selectbox("Destination", ["Cochin", "Banglore", "Delhi", "Hyderabad", "Kolkata"])
stops = st.selectbox("Total Stops", [0, 1, 2, 3, 4])
dep_time = st.time_input("Departure Time")
arr_time = st.time_input("Arrival Time")
journey_date = st.date_input("Journey Date")

# Convert inputs into dataframe
def create_input():
    duration = abs((datetime.combine(datetime.today(), arr_time) - datetime.combine(datetime.today(), dep_time)).seconds // 60)
    data = {
        'Airline': [airline],
        'Source': [source],
        'Destination': [destination],
        'Total_Stops': [stops],
        'Duration': [duration],
        'Dep_Time': [dep_time.strftime('%H:%M')],
        'Arrival_Time': [arr_time.strftime('%H:%M')],
        'Date_of_Journey': [journey_date.strftime('%d/%m/%Y')]
    }
    return pd.DataFrame(data)

# Predict
if st.button("Predict Price"):
    input_df = create_input()
    price = model.predict(input_df)[0]
    st.success(f"Estimated Flight Price: ₹{int(price):,}")