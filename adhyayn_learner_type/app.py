from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and label encoder
model = joblib.load("learner_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        
        # Extract gender (first feature)
        gender = int(data["gender"])  # 0 for Male, 1 for Female

        # Extract question responses
        questions = [int(data[f"q{i+1}"]) for i in range(len(data) - 1)]  # Exclude gender from count

        # Ensure feature count is correct
        features = np.array([gender] + questions).reshape(1, -1)  # Gender as first feature

        # Predict
        prediction = model.predict(features)
        learner_type = label_encoder.inverse_transform(prediction)[0]

        return jsonify({"learner_type": learner_type})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
