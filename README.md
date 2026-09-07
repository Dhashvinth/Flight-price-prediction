# Flight Price Prediction

An end-to-end machine learning project that predicts flight ticket
prices from journey and flight details. The project compares multiple
regression approaches, tunes an XGBoost model, saves the trained
pipeline, and serves predictions through a Streamlit web application.

## Project Overview

Flight prices can vary based on airline, source, destination, journey
date, departure and arrival time, duration, number of stops, and
additional flight information.

The project workflow includes:

-   Data inspection and missing-value handling
-   Price outlier analysis
-   Date, time, duration, and stop feature engineering
-   Categorical encoding and numerical preprocessing
-   Multiple Linear Regression as a baseline
-   Random Forest Regression
-   XGBoost Regression
-   Train/test evaluation and 5-fold cross-validation
-   XGBoost hyperparameter tuning with `RandomizedSearchCV`
-   Model serialization with Joblib
-   Interactive prediction using Streamlit

## Dataset

The dataset contains 10,683 flight records before cleaning. It includes
Airline, Date of Journey, Source, Destination, Route, Departure Time,
Arrival Time, Duration, Total Stops, Additional Information, and Price.

One incomplete record is removed during training.

## Data Preprocessing

The preprocessing logic is kept in `preprocess.py` so the same
transformations are used during training and prediction.

Key transformations:

-   Extract journey day, month, and day of week
-   Extract departure hour and minute
-   Extract arrival hour and minute
-   Convert flight duration into total minutes
-   Convert total stops into numerical values
-   Drop original route/date/time columns after feature engineering
-   One-hot encode categorical features
-   Median-impute numerical features
-   Most-frequent-impute categorical features
-   Standard-scale numerical features

## Outlier Analysis

Flight-price outliers are identified using the IQR method. High-price
records are inspected rather than automatically deleted because several
represent legitimate premium or business-class fares. The final model
therefore retains these observations.

## Models Compared

1.  Multiple Linear Regression
2.  Random Forest Regressor
3.  XGBoost Regressor

Multiple Linear Regression provides a baseline. Random Forest and
XGBoost are used to capture nonlinear relationships and interactions.

## Model Evaluation

Models are evaluated using MAE, RMSE, R², training-versus-testing R²,
and 5-fold cross-validation.

Final XGBoost results:

  Metric              Result
  -------------- -----------
  Training R²         0.9694
  Testing R²          0.9209
  Testing MAE        ₹727.72
  Testing RMSE     ₹1,306.22

Training and testing scores are both reported so model generalization
can be assessed rather than relying only on training performance.

## Hyperparameter Tuning

`RandomizedSearchCV` tunes important XGBoost parameters:

-   `n_estimators`
-   `learning_rate`
-   `max_depth`
-   `min_child_weight`
-   `subsample`

These parameters help balance predictive performance and model
complexity.

## Streamlit Application

The Streamlit interface accepts airline, source, destination, journey
date, departure and arrival time, number of stops, and additional flight
information. It applies the same preprocessing used during training and
returns the predicted flight price.

## Project Structure

``` text
Flight-Price-Prediction/
├── app.py
├── preprocess.py
├── train_model.py
├── Flight_Price.csv
├── model.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

``` bash
git clone <your-repository-url>
cd Flight-Price-Prediction
pip install -r requirements.txt
```

## Run the Application

Because `model.pkl` is included, the app can be started directly:

``` bash
streamlit run app.py
```

## Retrain the Model

``` bash
python train_model.py
```

This compares the regression models, tunes XGBoost, evaluates the final
model, and saves a new `model.pkl`.

## Technologies Used

Python, Pandas, NumPy, scikit-learn, XGBoost, Streamlit, and Joblib.

## Key Learning Outcomes

This project demonstrates end-to-end regression development, feature
engineering for date/time data, categorical and numerical preprocessing,
baseline-versus-ensemble comparison, generalization checks,
hyperparameter tuning, consistent training/inference preprocessing, and
Streamlit deployment.
