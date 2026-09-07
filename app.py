import streamlit as st

import pandas as pd

import joblib

from preprocess import preprocess_data



# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(

    page_title="Flight Price Prediction",

    page_icon="✈️",

    layout="wide"

)



# ======================================================
# LOAD MODEL
# ======================================================
from pathlib import Path


@st.cache_resource
def load_model():

    model_path = Path(__file__).parent / "model.pkl"

    model = joblib.load(model_path)

    return model


model = load_model()





# ======================================================
# TITLE
# ======================================================

st.title(
    "✈️ Flight Price Prediction"
)


st.write(
    "Predict the estimated airfare using Machine Learning."
)


st.divider()



# ======================================================
# VALUES FOR DROPDOWNS
# ======================================================

AIRLINES = [

    "Air India",

    "Air Asia",

    "GoAir",

    "IndiGo",

    "Jet Airways",

    "Jet Airways Business",

    "Multiple carriers",

    "Multiple carriers Premium economy",

    "SpiceJet",

    "Trujet",

    "Vistara",

    "Vistara Premium economy"

]



CITIES = [

    "Banglore",

    "Chennai",

    "Delhi",

    "Kolkata",

    "Mumbai",

    "Cochin",

    "Hyderabad",

    "New Delhi"

]



STOPS = [

    "non-stop",

    "1 stop",

    "2 stops",

    "3 stops",

    "4 stops"

]



ADDITIONAL_INFO = [

    "No info",

    "In-flight meal not included",

    "No check-in baggage included",

    "1 Long layover",

    "Change airports",

    "Business class"

]



# ======================================================
# INPUT SECTION
# ======================================================

col1, col2 = st.columns(2)



# ======================================================
# LEFT SIDE
# ======================================================

with col1:


    airline = st.selectbox(

        "Airline",

        AIRLINES

    )


    source = st.selectbox(

        "Source",

        CITIES

    )


    journey_date = st.date_input(

        "Date of Journey"

    )


    departure_time = st.time_input(

        "Departure Time"

    )


    total_stops = st.selectbox(

        "Total Stops",

        STOPS

    )



# ======================================================
# RIGHT SIDE
# ======================================================

with col2:


    destination = st.selectbox(

        "Destination",

        CITIES

    )


    arrival_time = st.time_input(

        "Arrival Time"

    )


    additional_info = st.selectbox(

        "Additional Information",

        ADDITIONAL_INFO

    )



# ======================================================
# VALIDATION
# ======================================================

if source == destination:

    st.warning(
        "Source and Destination cannot be the same."
    )



# ======================================================
# PREDICT BUTTON
# ======================================================

if st.button(

    "Predict Flight Price",

    use_container_width=True

):


    if source == destination:

        st.error(
            "Please choose a different destination."
        )


    else:


        # ======================================================
        # FORMAT DATE
        # ======================================================

        journey_date_string = (
            journey_date.strftime(
                "%d/%m/%Y"
            )
        )



        # ======================================================
        # FORMAT TIMES
        # ======================================================

        departure_string = (
            departure_time.strftime(
                "%H:%M"
            )
        )


        arrival_string = (
            arrival_time.strftime(
                "%H:%M"
            )
        )



        # ======================================================
        # CALCULATE DURATION
        # ======================================================

        departure_minutes = (

            departure_time.hour * 60

            + departure_time.minute

        )


        arrival_minutes = (

            arrival_time.hour * 60

            + arrival_time.minute

        )



        # If arrival is earlier than departure,
        # assume arrival is on the next day.

        if arrival_minutes < departure_minutes:

            arrival_minutes = (

                arrival_minutes

                + 24 * 60

            )



        duration_minutes = (

            arrival_minutes

            - departure_minutes

        )



        # ======================================================
        # CONVERT DURATION TO HOURS AND MINUTES
        # ======================================================

        hours = (
            duration_minutes // 60
        )


        minutes = (
            duration_minutes % 60
        )



        if hours > 0 and minutes > 0:

            duration_string = (

                f"{hours}h {minutes}m"

            )


        elif hours > 0:

            duration_string = (

                f"{hours}h"

            )


        else:

            duration_string = (

                f"{minutes}m"

            )



        # ======================================================
        # CREATE DATAFRAME FOR PREDICTION
        # ======================================================

        input_data = pd.DataFrame({

            "Airline": [
                airline
            ],

            "Date_of_Journey": [
                journey_date_string
            ],

            "Source": [
                source
            ],

            "Destination": [
                destination
            ],

            "Route": [
                "Unknown"
            ],

            "Dep_Time": [
                departure_string
            ],

            "Arrival_Time": [
                arrival_string
            ],

            "Duration": [
                duration_string
            ],

            "Total_Stops": [
                total_stops
            ],

            "Additional_Info": [
                additional_info
            ]

        })



        # ======================================================
        # PREPROCESS INPUT
        # ======================================================

        processed_data = preprocess_data(
            input_data
        )



        # ======================================================
        # PREDICT
        # ======================================================

        prediction = model.predict(
            processed_data
        )


        predicted_price = (
            prediction[0]
        )



        # ======================================================
        # RESULT
        # ======================================================

        st.success(

            f"Estimated Flight Price: ₹{predicted_price:,.0f}"

        )



        # ======================================================
        # FLIGHT SUMMARY
        # ======================================================

        st.subheader(
            "Flight Summary"
        )


        summary1, summary2 = st.columns(2)



        with summary1:

            st.write(
                "**Airline:**",
                airline
            )

            st.write(
                "**Source:**",
                source
            )

            st.write(
                "**Destination:**",
                destination
            )

            st.write(
                "**Journey Date:**",
                journey_date.strftime(
                    "%d %B %Y"
                )
            )



        with summary2:

            st.write(
                "**Departure Time:**",
                departure_string
            )

            st.write(
                "**Arrival Time:**",
                arrival_string
            )

            st.write(
                "**Total Stops:**",
                total_stops
            )

            st.write(
                "**Duration:**",
                duration_string
            )