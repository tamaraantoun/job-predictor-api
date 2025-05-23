from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

model = joblib.load('job_seeking_model.pkl')
gender_enc = joblib.load('gender_encoder.pkl')
size_enc = joblib.load('size_encoder.pkl')

print("GENDER ENCODER:", gender_enc.classes_)
print("SIZE ENCODER:", size_enc.classes_)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    print("📥 INPUT:", data)

    try:
        gender = data["gender"].strip().title()
        size = data["company_size"].strip().upper()
        hours = float(data["training_hours"])

        gender_val = gender_enc.transform([gender])[0]
        size_val = size_enc.transform([size])[0]

        print("✅ ENCODED:", gender_val, hours, size_val)

        pred = model.predict([[gender_val, hours, size_val]])[0]
        print("✅ PREDICTION:", pred)

        return jsonify({"seeking": bool(pred)})
    except Exception as e:
        import traceback
        print("❌ ERROR:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400



@app.route("/", methods=["GET"])
def home():
    return "✅ Job Prediction API is running!"

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
