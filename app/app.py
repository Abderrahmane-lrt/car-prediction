import streamlit as st
import pandas as pd
import joblib


# Configuration
st.set_page_config(
    page_title="Driver Station",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Driver Station")
st.write("Prédiction du prix d'une voiture")


# Charger le modèle
model = joblib.load("model/random-forest-model.pkl")


# Formulaire
st.header("Informations de la voiture")

year = st.number_input(
    "Année",
    min_value=1990,
    max_value=2026,
    value=2018
)

km_driven = st.number_input(
    "Kilométrage",
    min_value=0,
    value=50000
)

owner = st.selectbox(
    "Propriétaire",
    ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"]
)

fuel = st.selectbox(
    "Carburant",
    ["CNG", "Diesel", "Electric", "LPG", "Petrol"]
)

seller_type = st.selectbox(
    "Type de vendeur",
    ["Dealer", "Individual", "Trustmark Dealer"]
)

transmission = st.selectbox(
    "Transmission",
    ["Automatic", "Manual"]
)


# Conversion des choix en features
data = pd.DataFrame([{
    "year": year,
    "km_driven": km_driven,
    "owner": {
        "First Owner": 1,
        "Second Owner": 2,
        "Third Owner": 3,
        "Fourth & Above Owner": 4,
        "Test Drive Car": 0
    }[owner],

    "fuel_CNG": int(fuel == "CNG"),
    "fuel_Diesel": int(fuel == "Diesel"),
    "fuel_Electric": int(fuel == "Electric"),
    "fuel_LPG": int(fuel == "LPG"),
    "fuel_Petrol": int(fuel == "Petrol"),

    "seller_type_Dealer": int(seller_type == "Dealer"),
    "seller_type_Individual": int(seller_type == "Individual"),
    "seller_type_Trustmark Dealer": int(seller_type == "Trustmark Dealer"),

    "transmission_Automatic": int(transmission == "Automatic"),
    "transmission_Manual": int(transmission == "Manual")
}])


# Prédiction
if st.button("Prédire le prix"):

    prediction = model.predict(data)

    st.success(
        f"💰 Prix estimé : {prediction[0]:,.0f} DH"
    )
