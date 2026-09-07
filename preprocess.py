import pandas as pd


# ======================================================
# CONVERT DURATION INTO MINUTES
# ======================================================

def convert_duration(duration):

    duration = str(duration)

    hours = 0
    minutes = 0


    if "h" in duration:

        hours = int(
            duration.split("h")[0]
        )


    if "m" in duration:

        if "h" in duration:

            temp = duration.split("h")[1]

            temp = temp.replace(" ", "")

            temp = temp.replace("m", "")


            if temp != "":

                minutes = int(temp)

        else:

            minutes = int(
                duration.replace("m", "")
            )


    total_minutes = hours * 60 + minutes

    return total_minutes



# ======================================================
# PREPROCESS DATA
# ======================================================

def preprocess_data(df):

    df = df.copy()


    # ======================================================
    # DATE OF JOURNEY
    # ======================================================

    df["Date_of_Journey"] = pd.to_datetime(

        df["Date_of_Journey"],

        format="%d/%m/%Y",

        errors="coerce"

    )


    df["Journey_Day"] = (
        df["Date_of_Journey"].dt.day
    )


    df["Journey_Month"] = (
        df["Date_of_Journey"].dt.month
    )


    df["Journey_DayOfWeek"] = (
        df["Date_of_Journey"].dt.dayofweek
    )


    # ======================================================
    # DEPARTURE TIME
    # ======================================================

    dep = pd.to_datetime(

        df["Dep_Time"],

        format="%H:%M",

        errors="coerce"

    )


    df["Dep_hour"] = dep.dt.hour

    df["Dep_min"] = dep.dt.minute


    # ======================================================
    # ARRIVAL TIME
    # ======================================================

    df["Arrival_Time"] = (
        df["Arrival_Time"]
        .astype(str)
        .str.split()
        .str[0]
    )


    arr = pd.to_datetime(

        df["Arrival_Time"],

        format="%H:%M",

        errors="coerce"

    )


    df["Arrival_hour"] = arr.dt.hour

    df["Arrival_min"] = arr.dt.minute


    # ======================================================
    # DURATION
    # ======================================================

    df["Duration"] = (
        df["Duration"]
        .apply(convert_duration)
    )


    # ======================================================
    # TOTAL STOPS
    # ======================================================

    df["Total_Stops"] = df["Total_Stops"].replace({

        "non-stop": 0,

        "1 stop": 1,

        "2 stops": 2,

        "3 stops": 3,

        "4 stops": 4

    })


    # ======================================================
    # DROP COLUMNS
    # ======================================================

    df.drop(

        [
            "Route",
            "Date_of_Journey",
            "Dep_Time",
            "Arrival_Time"
        ],

        axis=1,

        errors="ignore",

        inplace=True

    )


    return df