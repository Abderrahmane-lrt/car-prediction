from .cleaning import x,y
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import pandas as pd

x_train, x_test, y_train, y_test= train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=10
)

params_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_rf = GridSearchCV(
    RandomForestRegressor(random_state=10),
    params_rf,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

grid_rf.fit(x_train,y_train)


best_rf =grid_rf.best_estimator_




scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


model = LinearRegression()

model.fit(x_train_scaled,y_train)

y_pred = model.predict(x_test_scaled)

mae = mean_absolute_error(y_test,y_pred)
rmse = mean_squared_error(y_test,y_pred) **0.5
r2 = r2_score(y_test,y_pred)

print("MAE :", mae)
print("RMSE :", rmse)
print("R² :", r2)


model_rf= RandomForestRegressor(
    n_estimators=100,
    random_state=10
)

model_rf.fit(x_train,y_train)

y_pred_rf= model_rf.predict(x_test)


mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf) ** 0.5
r2_rf = r2_score(y_test, y_pred_rf)

# print("MAE :", mae_rf)
# print("RMSE :", rmse_rf)
# print("R² :", r2_rf)

model_xgb = XGBRegressor(
    n_estimators=100,
    random_state=10
)

model_xgb.fit(x_train,y_train)

y_pred_xgb=model_xgb.predict(x_test)

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = mean_squared_error(y_test, y_pred_xgb) ** 0.5
r2_xgb = r2_score(y_test, y_pred_xgb)

# print("MAE :", mae_xgb)
# print("RMSE :", rmse_xgb)
# print("R² :", r2_xgb)


model_svr=SVR(
    kernel="rbf",
    C=100,
    epsilon=0.1
)

model_svr.fit(x_train_scaled,y_train)

y_pred_svr=model_svr.predict(x_test_scaled)

mae_svr = mean_absolute_error(y_test, y_pred_svr)
rmse_svr = mean_squared_error(y_test, y_pred_svr) ** 0.5
r2_svr = r2_score(y_test, y_pred_svr)

print("SVR")
print("MAE :", mae_svr)
print("RMSE :", rmse_svr)
print("R² :", r2_svr)