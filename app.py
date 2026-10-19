
import os
import math
import joblib
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Allow local testing and deployed frontend
allowed_origin = os.environ.get("FRONTEND_URL", "*")
CORS(app, resources={r"/predict": {"origins": allowed_origin}})

# Trained machine learning model
MODEL_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "weather_model_real.pkl"
)

FEATURES = ["temperature", "humidity", "wind_speed"]

try:
    model = joblib.load(MODEL_FILE)
    print("WeatherGPT model loaded successfully!")
except Exception as e:
    app.logger.exception("Could not load weather model")
    raise RuntimeError(
        f"Unable to load {MODEL_FILE}. "
        "Check that weather_model_real.pkl exists in the project."
    ) from e


# Home route
@app.route("/")
def home():
    return "WeatherGPT API is running!"


# Health check route
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "service": "WeatherGPT",
        "status": "ok"
    })


# AI temperature prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Please send valid JSON data."
            }), 400

        # Check required fields
        for field in FEATURES:
            if field not in data:
                return jsonify({
                    "error": f"Missing required field: {field}"
                }), 400

        # Convert input values to numbers
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        wind_speed = float(data["wind_speed"])

        # Reject NaN and infinite values
        values = [temperature, humidity, wind_speed]

        if not all(math.isfinite(value) for value in values):
            return jsonify({
                "error": "All values must be finite numbers."
            }), 400

        # Validate humidity and wind speed
        if not 0 <= humidity <= 100:
            return jsonify({
                "error": "Humidity must be between 0 and 100."
            }), 400

        if wind_speed < 0:
            return jsonify({
                "error": "Wind speed cannot be negative."
            }), 400

        # Prepare model input
        sample = pd.DataFrame(
            [{
                "temperature": temperature,
                "humidity": humidity,
                "wind_speed": wind_speed
            }],
            columns=FEATURES
        )

        # Generate prediction
        prediction = float(model.predict(sample)[0])

        if not math.isfinite(prediction):
            raise ValueError("Model returned an invalid prediction.")

        return jsonify({
            "predicted_next_temperature": round(prediction, 1),
            "unit": "°C",
            "note": (
                "Experimental ML estimate, "
                "not an official weather forecast."
            )
        })

    except (KeyError, TypeError, ValueError, OverflowError):
        return jsonify({
            "error": (
                "Provide valid temperature, humidity "
                "and wind_speed values."
            )
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed on the server."
        }), 500


# Run locally or on Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
