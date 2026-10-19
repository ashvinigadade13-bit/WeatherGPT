
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

DATA_FILE = "weather_data_real.csv"
MODEL_FILE = "weather_model_real.pkl"

FEATURES = ["temperature", "humidity", "wind_speed"]
TARGET = "next_temperature"

# Load real historical weather data
df = pd.read_csv(DATA_FILE)

# Support a few common alternative column names
aliases = {
    "temperature_2m_mean": "temperature",
    "relative_humidity_2m_mean": "humidity",
    "wind_speed_10m_mean": "wind_speed",
    "next_day_temperature": "next_temperature",
    "next_day_temp": "next_temperature",
}

df = df.rename(columns={
    old: new for old, new in aliases.items()
    if old in df.columns and new not in df.columns
})

print("Dataset columns:", df.columns.tolist())

required = FEATURES + [TARGET]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(
        f"Missing columns: {missing}. "
        "Check the column names in weather_data_real.csv."
    )

# Keep only required numeric data and remove missing values
df = df[required].apply(pd.to_numeric, errors="coerce")
df = df.dropna().reset_index(drop=True)

if len(df) < 30:
    raise ValueError(
        f"Only {len(df)} valid rows found. "
        "Download more historical weather data first."
    )

# Keep the chronological order: earlier data for training,
# later data for testing.
split = int(len(df) * 0.8)

train = df.iloc[:split]
test = df.iloc[split:]

X_train = train[FEATURES]
y_train = train[TARGET]
X_test = test[FEATURES]
y_test = test[TARGET]

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate only on data not used for training
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print("\nWeatherGPT model training complete!")
print("Training rows:", len(train))
print("Testing rows:", len(test))
print(f"Test MAE: {mae:.2f} °C")

# Train final model on all available historical rows
final_model = LinearRegression()
final_model.fit(df[FEATURES], df[TARGET])

joblib.dump(final_model, MODEL_FILE)
print(f"Saved model to {MODEL_FILE}")

print("\nExample prediction:")
example = df[FEATURES].iloc[[-1]]
estimate = final_model.predict(example)[0]
print(f"Estimated next-day temperature: {estimate:.1f} °C")
print("This is an experimental estimate, not an official forecast.")
