from .explore import df 
import pandas as pd

char= [
    'fuel',
    'seller_type',
    'transmission',
    'owner'
]

for c in char :
    df[c]=df[c].fillna(df[c].mode()[0])

num = [
    'year',
    'selling_price',
    'km_driven'
]

for n in num:
    df[n]=df[n].fillna(df[n].median())



df = df.drop_duplicates()


# print('year = ', (df["year"]<1900).sum())
# print('km = ', (df['km_driven']<0).sum())
# print('prix = ',(df['selling_price']<0).sum())

# for c in char :
#     print(df[c].value_counts())



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

# print("Prix négatif :", (df["selling_price"] < 0).sum())
# print("Km négatif :", (df["km_driven"] < 0).sum())
# print("Année négative :", (df["year"] < 0).sum())

# print(df[["year", "selling_price", "km_driven"]].describe())

# print(
#     df[df["selling_price"] == df["selling_price"].max()]
#     [["name", "year", "selling_price", "km_driven"]]
# )

# print(
#     df[df["km_driven"] == df["km_driven"].max()]
#     [["name", "year", "selling_price", "km_driven"]]
# )

# print(
#     df[
#         ["name", "year", "selling_price", "km_driven"]
#     ].sort_values("km_driven", ascending=False).head(10)
# )

# print("année min :", df["year"].min())
# print("année max :", df["year"].max())

df = pd.get_dummies(df,columns=['fuel','seller_type','transmission'],dtype=int)


owner={
    'First Owner':1,
    'Second Owner':2,
    'Third Owner':3,
    'Fourth & Above Owner':4,
    'Test Drive Car':0
}

df['owner']=df['owner'].map(owner)

df.to_csv('data/processed/cleaned_cars.csv' ,index=False)
# print(df['owner'].isna().sum())

x = df.drop('selling_price',axis=1)
x= x.drop('name',axis=1)
y=df['selling_price']
