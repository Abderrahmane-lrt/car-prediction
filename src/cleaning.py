from .explore import df 
import pandas as pd


"""
Fill Missing Values :

Categorical Columns => using mode
Numerical Columns => using median

"""

char= [
    'fuel',
    'seller_type',
    'transmission',
    'owner'
]

df[char] = df[char].fillna(df[char].mode()[0])

num = [
    'year',
    'selling_price',
    'km_driven'
]

df[num] = df[num].fillna(df[num].median())


# Drop Duplicates

df = df.drop_duplicates()




def outlier(c):
    Q1 = df[c].quantile(0.25)
    Q3 = df[c].quantile(0.75)

    iqr = Q3-Q1

    borninf = Q1 - 1.5*iqr
    bornsup = Q3 + 1.5*iqr

    outliers = df[
        (df[c] < borninf) |
        (df[c] > bornsup)
    ]
    return outliers


"""
Encoding Categorical Columns using One-hot Encodding 
"""

df = pd.get_dummies(df,columns=['fuel','seller_type','transmission'],dtype=int)


# Use Ordinary for owner 
owner = {
    'First Owner':1,
    'Second Owner':2,
    'Third Owner':3,
    'Fourth & Above Owner':4,
    'Test Drive Car':0
}

df['owner']=df['owner'].map(owner)

df.to_csv('data/processed/cleaned_cars.csv' ,index=False)

x = df.drop(columns=['selling_price', 'name'],axis=1)
y = df['selling_price']
