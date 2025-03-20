from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import webbrowser
import threading

# Load dataset (simulated for this example)
data = {
    "Vibration_Level": np.random.uniform(0.1, 5.0, 500),  # Min: 0.1 mm/s, Max: 5.0 mm/s
    "Temperature": np.random.uniform(30, 120, 500),       # Min: 30°C, Max: 120°C
    "Axle_Load": np.random.uniform(10, 25, 500),         # Min: 10 tons, Max: 25 tons
    "Brake_Pressure": np.random.uniform(4.0, 8.0, 500),  # Min: 4.0 bar, Max: 8.0 bar
    "Speed": np.random.uniform(30, 130, 500),            # Min: 30 km/h, Max: 130 km/h (Updated)
    "Voltage": np.random.uniform(600, 800, 500),         # Min: 600 V, Max: 800 V
    "Current": np.random.uniform(50, 150, 500),          # Min: 50 A, Max: 150 A
    "Acoustic_Noise": np.random.uniform(40, 100, 500),   # Min: 40 dB, Max: 100 dB
}

df = pd.DataFrame(data)

# Define THI calculation
THI = 100 - (
    (df["Vibration_Level"] > 3.0).astype(int) * 15 +
    (df["Temperature"] > 90).astype(int) * 20 +
    (df["Axle_Load"] > 20).astype(int) * 10 +
    (df["Brake_Pressure"] < 5.0).astype(int) * 15 +
    (df["Speed"] > 100).astype(int) * 10 +
    (df["Voltage"] < 650).astype(int) * 10 +
    (df["Current"] > 120).astype(int) * 10 +
    (df["Acoustic_Noise"] > 80).astype(int) * 10
)

THI = THI.clip(lower=0)
df["THI_Score"] = THI

# Prepare training data
X = df.drop("THI_Score", axis=1)
y = df["THI_Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using KNN
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "thi_model.pkl")

# Define suggestion logic
def get_suggestion(thi_score):
    if thi_score >= 90:
        return "No action required"
    elif thi_score >= 70:
        return "Routine maintenance suggested"
    elif thi_score >= 50:
        return "Preemptive repair needed"
    elif thi_score >= 30:
        return "Urgent maintenance required"
    else:
        return "Immediate shutdown & servicing"

# Start Flask app
app = Flask(__name__)

# HTML content embedded in Python
HTML_CONTENT = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Train Health Index (THI) Predictor</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
            color: #ffffff;
            text-align: center;
            padding: 20px;
        }
        h1 {
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 3px 3px 6px #000000;
            animation: fadeIn 2s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        .train-container {
            background: rgba(0, 0, 0, 0.8);
            padding: 30px;
            border-radius: 20px;
            display: inline-block;
            width: 90%;
            max-width: 700px;
            margin: 20px auto;
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.6);
            border: 2px solid #ffcc00;
        }
        input[type=number], button {
            width: 80%;
            padding: 15px;
            margin: 12px 0;
            border-radius: 15px;
            border: none;
            font-size: 1.2em;
            box-shadow: inset 2px 2px 5px rgba(0, 0, 0, 0.3);
        }
        button {
            background-color: #ffcc00;
            color: #000;
            cursor: pointer;
            transition: background 0.3s, transform 0.3s;
        }
        button:hover {
            background-color: #ff9900;
            transform: scale(1.05);
        }
        .result {
            margin-top: 20px;
            font-size: 1.7em;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
        }
        .footer {
            margin-top: 40px;
            font-size: 1em;
            color: #ddd;
        }
    </style>
</head>
<body>
    <h1>🚂 Train Health Index (THI) Predictor 🚂</h1>
    <div class="train-container">
        <form id="thiForm">
            <label>Vibration Level (0.1-5.0 mm/s): <input type="number" step="0.1" min="0.1" max="5.0" name="vibration" required></label><br>
            <label>Temperature (30-120°C): <input type="number" min="30" max="120" name="temperature" required></label><br>
            <label>Axle Load (10-25 tons): <input type="number" step="0.1" min="10" max="25" name="axle_load" required></label><br>
            <label>Brake Pressure (4.0-8.0 bar): <input type="number" step="0.1" min="4.0" max="8.0" name="brake_pressure" required></label><br>
            <label>Speed (30-130 km/h): <input type="number" min="30" max="130" name="speed" required></label><br>
            <label>Voltage (600-800 V): <input type="number" min="600" max="800" name="voltage" required></label><br>
            <label>Current (50-150 A): <input type="number" min="50" max="150" name="current" required></label><br>
            <label>Acoustic Noise (40-100 dB): <input type="number" min="40" max="100" name="noise" required></label><br>
            <button type="button" onclick="predictTHI()">Predict THI</button>
        </form>
        <div id="result" class="result"></div>
    </div>

    <div class="footer">🌟 For a smooth and secure journey, trust  healthy train! 🌟
    </div>

    <script>
        function predictTHI() {
            const form = document.getElementById('thiForm');
            const data = new FormData(form);
            
            fetch('/predict', {
                method: 'POST',
                body: JSON.stringify(Object.fromEntries(data)),
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = `THI Score: <strong>${data.thi}</strong><br>Suggestion: <strong>${data.suggestion}</strong>`;
                resultDiv.style.backgroundColor = getColorForTHI(data.thi);
                alert("FOR SAFE AND SECURE JOURNEY, CHOOSE HEALTHY TRAIN!");
            })
            .catch(error => console.error('Error:', error));
        }

        function getColorForTHI(thi) {
            if (thi >= 90) return '#4CAF50'; // Excellent - Green
            if (thi >= 70) return '#8BC34A'; // Good - Light Green
            if (thi >= 50) return '#FFC107'; // Moderate - Amber
            if (thi >= 30) return '#FF5722'; // Poor - Orange
            return '#F44336'; // Critical - Red
        }
    </script>
</body>
</html>
'''

# Serve the HTML content
@app.route('/')
def home():
    return render_template_string(HTML_CONTENT)

# API Endpoint for Prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array([
        float(data['vibration']),
        float(data['temperature']),
        float(data['axle_load']),
        float(data['brake_pressure']),
        float(data['speed']),
        float(data['voltage']),
        float(data['current']),
        float(data['noise'])
    ]).reshape(1, -1)

    model = joblib.load("thi_model.pkl")
    thi_score = model.predict(input_data)[0]
    suggestion = get_suggestion(thi_score)

    return jsonify({'thi': round(thi_score, 2), 'suggestion': suggestion})

# Function to open the web browser automatically
def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000')

if __name__ == '__main__':
    threading.Timer(1.25, open_browser).start()  # Delay to ensure the server starts first
    app.run(host='0.0.0.0', port=5000, debug=True)
