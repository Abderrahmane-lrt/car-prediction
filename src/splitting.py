from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

CLEANED_CSV = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "cleaned_cars.csv"
)

TEST_SIZE = 0.2
SEED = 10

df = pd.read_csv(CLEANED_CSV)

X = df.drop("selling_price", axis=1)
X = X.drop("name", axis=1)
y = df["selling_price"]

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=SEED
)
