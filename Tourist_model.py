import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib  

df = pd.read_csv(r"D:\Sparsh\ML_Projects\Tourist_Safety_Prediction\Dataset\tourist_safety_dataset.csv")


X = df[["time_in_red_zone_min", "time_near_red_zone_min", 
        "red_zone_passes", "last_update_gap_min", "deviation_km"]]
y = df["safety_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


monotone_constraints = (1, 1, 1, 1, 1)  

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    monotone_constraints=monotone_constraints,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

importances = pd.Series(model.feature_importances_, index=X.columns)
print("\n Feature Importances (Monotonic):")
print(importances.sort_values(ascending=False))


joblib.dump(model,'Tourist_safety_model.joblib')
print("Model is saved in joblib file")


