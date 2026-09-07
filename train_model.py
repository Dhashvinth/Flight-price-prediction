import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from preprocess import preprocess_data


# ======================================================
# LOAD DATA
# ======================================================

df = pd.read_csv("Flight_Price.csv")


print("\nDataset Shape:")
print(df.shape)


print("\nFirst 5 Rows:")
print(df.head())


# ======================================================
# CHECK MISSING VALUES
# ======================================================

print("\nMissing Values:")

print(df.isnull().sum())


# Only one row has missing values in our dataset.
# So we remove the incomplete row.

df = df.dropna()


print("\nShape After Removing Missing Values:")

print(df.shape)


# ======================================================
# OUTLIER ANALYSIS
# ======================================================

Q1 = df["Price"].quantile(0.25)

Q3 = df["Price"].quantile(0.75)

IQR = Q3 - Q1


lower_limit = Q1 - 1.5 * IQR

upper_limit = Q3 + 1.5 * IQR


print("\nQ1:", Q1)

print("Q3:", Q3)

print("IQR:", IQR)

print("Lower Limit:", lower_limit)

print("Upper Limit:", upper_limit)


outliers = df[

    (df["Price"] < lower_limit)

    |

    (df["Price"] > upper_limit)

]


print("\nNumber of Price Outliers:")

print(len(outliers))


print("\nTop Expensive Outliers:")

print(

    outliers[

        [
            "Airline",
            "Additional_Info",
            "Price"
        ]

    ]

    .sort_values(

        "Price",

        ascending=False

    )

    .head(10)

)


# We are not removing the outliers.
# Some of the high-price observations are valid
# Business Class or premium flights.


# ======================================================
# PREPROCESS DATA
# ======================================================

df = preprocess_data(df)


print("\nData After Preprocessing:")

print(df.head())


# ======================================================
# CHECK DATA TYPES
# ======================================================

print("\nData Types After Preprocessing:")

print(df.dtypes)


# ======================================================
# CHECK OBJECT COLUMNS
# ======================================================

object_cols = df.select_dtypes(

    include=["object"]

).columns


print("\nObject Columns:")


for column in object_cols:

    print("\nColumn:", column)

    print(

        df[column].unique()[:20]

    )


# ======================================================
# SPLIT X AND y
# ======================================================

X = df.drop(

    "Price",

    axis=1

)


y = df["Price"]


# ======================================================
# IDENTIFY CATEGORICAL AND NUMERICAL COLUMNS
# ======================================================

categorical_cols = X.select_dtypes(

    include=["object"]

).columns.tolist()


numerical_cols = X.select_dtypes(

    exclude=["object"]

).columns.tolist()


print("\nCategorical Columns:")

print(categorical_cols)


print("\nNumerical Columns:")

print(numerical_cols)


# ======================================================
# NUMERICAL PIPELINE
# ======================================================

numeric_pipeline = Pipeline([

    (

        "imputer",

        SimpleImputer(

            strategy="median"

        )

    ),

    (

        "scaler",

        StandardScaler()

    )

])


# ======================================================
# CATEGORICAL PIPELINE
# ======================================================

categorical_pipeline = Pipeline([

    (

        "imputer",

        SimpleImputer(

            strategy="most_frequent"

        )

    ),

    (

        "onehot",

        OneHotEncoder(

            handle_unknown="ignore"

        )

    )

])


# ======================================================
# COLUMN TRANSFORMER
# ======================================================

preprocessor = ColumnTransformer(

    transformers=[

        (

            "num",

            numeric_pipeline,

            numerical_cols

        ),

        (

            "cat",

            categorical_pipeline,

            categorical_cols

        )

    ]

)


# ======================================================
# TRAIN TEST SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)


print("\nTraining Rows:")

print(len(X_train))


print("\nTesting Rows:")

print(len(X_test))


# ======================================================
# FUNCTION TO EVALUATE MODEL
# ======================================================

def evaluate_model(model_name, model):

    print("\n======================================")

    print(model_name)

    print("======================================")


    # Train model

    model.fit(

        X_train,

        y_train

    )


    # Prediction on training data

    train_prediction = model.predict(

        X_train

    )


    # Prediction on testing data

    test_prediction = model.predict(

        X_test

    )


    # ======================================================
    # TRAINING METRICS
    # ======================================================

    train_mae = mean_absolute_error(

        y_train,

        train_prediction

    )


    train_rmse = np.sqrt(

        mean_squared_error(

            y_train,

            train_prediction

        )

    )


    train_r2 = r2_score(

        y_train,

        train_prediction

    )


    # ======================================================
    # TESTING METRICS
    # ======================================================

    test_mae = mean_absolute_error(

        y_test,

        test_prediction

    )


    test_rmse = np.sqrt(

        mean_squared_error(

            y_test,

            test_prediction

        )

    )


    test_r2 = r2_score(

        y_test,

        test_prediction

    )


    print("\nTraining Results")

    print("MAE :", train_mae)

    print("RMSE:", train_rmse)

    print("R2  :", train_r2)


    print("\nTesting Results")

    print("MAE :", test_mae)

    print("RMSE:", test_rmse)

    print("R2  :", test_r2)


    # ======================================================
    # OVERFITTING CHECK
    # ======================================================

    r2_difference = train_r2 - test_r2


    print("\nTrain-Test R2 Difference:")

    print(r2_difference)


    # ======================================================
    # CROSS VALIDATION
    # ======================================================

    cv_scores = cross_val_score(

        model,

        X_train,

        y_train,

        cv=5,

        scoring="r2"

    )


    print("\nCross Validation R2 Scores:")

    print(cv_scores)


    print("\nAverage Cross Validation R2:")

    print(cv_scores.mean())


    return model


# ======================================================
# MULTIPLE LINEAR REGRESSION
# ======================================================

linear_model = Pipeline([

    (

        "preprocessor",

        preprocessor

    ),

    (

        "model",

        LinearRegression()

    )

])


evaluate_model(

    "MULTIPLE LINEAR REGRESSION",

    linear_model

)


# ======================================================
# RANDOM FOREST
# ======================================================

random_forest_model = Pipeline([

    (

        "preprocessor",

        preprocessor

    ),

    (

        "model",

        RandomForestRegressor(

            n_estimators=200,

            random_state=42

        )

    )

])


evaluate_model(

    "RANDOM FOREST",

    random_forest_model

)


# ======================================================
# XGBOOST - INITIAL MODEL
# ======================================================

xgboost_model = Pipeline([

    (

        "preprocessor",

        preprocessor

    ),

    (

        "model",

        XGBRegressor(

            n_estimators=300,

            learning_rate=0.05,

            max_depth=6,

            random_state=42

        )

    )

])


evaluate_model(

    "XGBOOST - BEFORE TUNING",

    xgboost_model

)


# ======================================================
# XGBOOST HYPERPARAMETER TUNING
# ======================================================

print("\n======================================")

print("XGBOOST HYPERPARAMETER TUNING")

print("======================================")


xgb_tuning_model = Pipeline([

    (

        "preprocessor",

        preprocessor

    ),

    (

        "model",

        XGBRegressor(

            random_state=42

        )

    )

])


# ======================================================
# PARAMETERS TO CONTROL OVERFITTING
# ======================================================

parameters = {

    "model__n_estimators": [

        200,

        300,

        400

    ],

    "model__learning_rate": [

        0.03,

        0.05,

        0.1

    ],

    "model__max_depth": [

        3,

        4,

        5

    ],

    "model__min_child_weight": [

        1,

        3,

        5

    ],

    "model__subsample": [

        0.8,

        0.9,

        1.0

    ]

}


# RandomizedSearchCV will not try every possible
# combination.
#
# It will try only 20 combinations.

random_search = RandomizedSearchCV(

    xgb_tuning_model,

    parameters,

    n_iter=20,

    cv=5,

    scoring="neg_root_mean_squared_error",

    random_state=42,

    n_jobs=-1

)


random_search.fit(

    X_train,

    y_train

)


print("\nBest Parameters:")

print(

    random_search.best_params_

)


# ======================================================
# FINAL MODEL
# ======================================================

final_model = random_search.best_estimator_


# ======================================================
# FINAL MODEL PREDICTIONS
# ======================================================

train_prediction = final_model.predict(

    X_train

)


test_prediction = final_model.predict(

    X_test

)


# ======================================================
# FINAL TRAINING METRICS
# ======================================================

final_train_mae = mean_absolute_error(

    y_train,

    train_prediction

)


final_train_rmse = np.sqrt(

    mean_squared_error(

        y_train,

        train_prediction

    )

)


final_train_r2 = r2_score(

    y_train,

    train_prediction

)


# ======================================================
# FINAL TESTING METRICS
# ======================================================

final_test_mae = mean_absolute_error(

    y_test,

    test_prediction

)


final_test_rmse = np.sqrt(

    mean_squared_error(

        y_test,

        test_prediction

    )

)


final_test_r2 = r2_score(

    y_test,

    test_prediction

)


# ======================================================
# FINAL MODEL RESULTS
# ======================================================

print("\n======================================")

print("FINAL XGBOOST MODEL")

print("======================================")


print("\nTraining MAE:")

print(final_train_mae)


print("\nTraining RMSE:")

print(final_train_rmse)


print("\nTraining R2:")

print(final_train_r2)


print("\nTesting MAE:")

print(final_test_mae)


print("\nTesting RMSE:")

print(final_test_rmse)


print("\nTesting R2:")

print(final_test_r2)


print("\nTrain-Test R2 Difference:")

print(

    final_train_r2

    -

    final_test_r2

)


# ======================================================
# FINAL CROSS VALIDATION
# ======================================================

final_cv_scores = cross_val_score(

    final_model,

    X_train,

    y_train,

    cv=5,

    scoring="r2"

)


print("\nFinal Model Cross Validation R2 Scores:")

print(final_cv_scores)


print("\nFinal Average Cross Validation R2:")

print(

    final_cv_scores.mean()

)


# ======================================================
# SAVE FINAL MODEL
# ======================================================

model_path = Path(__file__).parent / "model.pkl"

joblib.dump(

    final_model,

    model_path

)


print("\nModel Saved Successfully!")

print("Model saved at:", model_path)