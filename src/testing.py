import joblib
import pandas as pd

model = joblib.load('model/random-forest-model.pkl')

data = pd.DataFrame([{
    'year': 2018,
    'km_driven': 50000,
    'owner': 1,
    'fuel_CNG': 0,
    'fuel_Diesel': 1,
    'fuel_Electric': 0,
    'fuel_LPG': 0,
    'fuel_Petrol': 0,
    'seller_type_Dealer': 1,
    'seller_type_Individual': 0,
    'seller_type_Trustmark Dealer': 0,
    'transmission_Automatic': 1,
    'transmission_Manual': 0
}])

prediction = model.predict(data)

print("Prix prédit :", prediction[0], "DH")