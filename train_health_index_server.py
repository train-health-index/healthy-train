from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import webbrowser
import threading

# Load trained model
model = joblib.load("thi_model.pkl")

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

# Simple HTML for testing
HTML_CONTENT = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Train Health Index (THI) Predictor</title>
</head>
<body>
    <h1>🚆 Train Health Index (THI) Predictor 🚆</h1>
    <form id="thiForm">
        <label>Vibration Level: <input type="number" name="vibration" required></label><br>
        <label>Temperature: <input type="number" name="temperature" required></label><br>
        <label>Axle Load: <input type="number" name="axle_load" required></label><br>
        <label>Brake Pressure: <input type="number" name="brake_pressure" required></label><br>
        <label>Speed: <input type="number" name="speed" required></label><br>
        <label>Voltage: <input type="number" name="voltage" required></label><br>
        <label>Current: <input type="number" name="current" required></label><br>
        <label>Acoustic Noise: <input type="number" name="noise" required></label><br>
        <button type="button" onclick="predictTHI()">Predict THI</button>
    </form>
    <div id="result"></div>
    <script>
        function predictTHI() {
            const form = document.getElementById('thiForm');
            const data = new FormData(form);
            fetch('/predict', {
                method: 'POST',
                body: JSON.stringify(Object.fromEntries(data)),
                headers: { 'Content-Type': 'application/json' }
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('result').innerHTML = `THI Score: ${data.thi} <br> Suggestion: ${data.suggestion}`;
            })
            .catch(error => console.error('Error:', error));
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_CONTENT)

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

    thi_score = model.predict(input_data)[0]
    suggestion = get_suggestion(thi_score)

    return jsonify({'thi': round(thi_score, 2), 'suggestion': suggestion})

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000')

if __name__ == '__main__':
    threading.Timer(1.25, open_browser).start()
    app.run(host='0.0.0.0', port=5000, debug=True)
