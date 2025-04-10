import os
import time
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import cv2
import mediapipe as mp
from playsound import playsound

# Correct usage of __name__
app = Flask(__name__)

# Load the trained model
model_dict = pickle.load(open('mod.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Define the path to the audio directory
AUDIO_DIR = os.path.join(os.path.dirname(__file__), 'static', 'audio')

# Global variables for cooldown
last_prediction = None
last_play_time = 0
COOLDOWN_TIME = 2  # seconds

def process_frame(image):
    global last_prediction, last_play_time

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        hands_data = []
        for hand_landmarks in results.multi_hand_landmarks:
            x_, y_ = [], []
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            hand_data = []
            for landmark in hand_landmarks.landmark:
                hand_data.append(landmark.x - min(x_))
                hand_data.append(landmark.y - min(y_))

            hands_data.append(hand_data)

        if len(hands_data) == 1:
            hands_data.append([0] * 42)  # Pad if only one hand

        data_aux = hands_data[0] + hands_data[1]

        if len(data_aux) == 84:
            prediction = model.predict([np.asarray(data_aux)])[0]

            current_time = time.time()
            if prediction != last_prediction or (current_time - last_play_time) > COOLDOWN_TIME:
                last_prediction = prediction
                last_play_time = current_time

                audio_file = os.path.join(AUDIO_DIR, f"{prediction}.wav")
                if os.path.exists(audio_file):
                    try:
                        playsound(audio_file)
                    except Exception as e:
                        print(f"Error playing sound: {e}")

            return prediction

    return "No hand detected"

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/model')
def model_page():
    return render_template('model.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['frame']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        prediction = process_frame(image)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

# Start the server
if __name__ == '__main__':
    if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # default to 5000 locally
    app.run(host='0.0.0.0', port=port)
