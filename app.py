
import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Local testing आणि deployed website साठी CORS
# Production मध्ये तुझ्या website चा origin environment
# variable मधून सेट करता येईल.
allowed_origin = os.environ.get("FRONTEND_URL", "*")
CORS(app, resources={r"/predict": {"origins": allowed_origin}})

MODEL_FILE = "weather_model_real.pkl"
FEATURES = ["temperature", "humidity", "wind_speed"]

try:
    model = joblib.load(MODEL_FILE)
    print("WeatherGPT model loaded successfully!")
except Exception as e:
    print("Model loading error:", e)
    raise

@app.route("/")
def home():
    return "WeatherGPT API is running!"

@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "WeatherGPT"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({"error": "Please send JSON data."}), 400

        sample = pd.DataFrame([{
            "temperature": float(data["temperature"]),
            "humidity": float(data["humidity"]),
            "wind_speed": float(data["wind_speed"])
        }], columns=FEATURES)

        if not all(sample.iloc[0].map(lambda x: pd.notna(x))):
            return jsonify({"error": "Values must be finite numbers."}), 400

        if not 0 <= sample.iloc[0]["humidity"] <= 100:
            return jsonify({"error": "Humidity must be between 0 and 100."}), 400

        if sample.iloc[0]["wind_speed"] < 0:
            return jsonify({"error": "Wind speed cannot be negative."}), 400

        prediction = float(model.predict(sample)[0])

        return jsonify({
            "predicted_next_temperature": round(prediction, 1),
            "unit": "°C",
            "note": "Experimental ML estimate, not an official weather forecast."
        })

    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Provide valid temperature, humidity and wind_speed."
        }), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed on the server."}), 500

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
