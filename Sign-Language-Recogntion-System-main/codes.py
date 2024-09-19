import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: ' '}  # Add space character to labels_dict

word = []  # To store the detected word
sentence = []  # To store the formed sentence
last_capture_time = time.time()  # To track the last time a frame was captured
capture_requested = False  # Flag to indicate whether a capture is requested

def capture_letter():
    global capture_requested
    capture_requested = True

def remove_last_letter():
    if word:
        del word[-1]  # Remove the last detected letter

while True:
    ret, frame = cap.read()

    if time.time() - last_capture_time >= 2:
        data_aux = []
        x_ = []
        y_ = []

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  
                    hand_landmarks,  
                    mp_hands.HAND_CONNECTIONS,  
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            if capture_requested:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                if predicted_character == ' ':  # If space character is detected, add space between letters
                    word.append(" ")  # Add space between letters
                else:
                    word.append(predicted_character)
                capture_requested = False

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

            last_capture_time = time.time()  

        cv2.putText(frame, "".join(word), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "".join(sentence), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    else:
        cv2.putText(frame, "".join(word), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "".join(sentence), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('c'):  
        capture_letter()
    elif key == ord(' '):  # If space key is pressed, add a space between letters
        word.append(" ")
    elif key == 8:  # If backspace key is pressed, remove the last detected letter
        remove_last_letter()
    elif key == 27:  # If Esc key is pressed, exit the loop
        break

cap.release()
cv2.destroyAllWindows()









<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign Language Recognition</title>
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        /* Custom CSS */
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 20px;
            color: #007bff; /* Blue color */
            transition: color 0.3s ease; /* Transition effect */
        }
        h1:hover {
            color: #0056b3; /* Darker blue color on hover */
        }
        .content {
            text-align: justify;
            font-size: 1.1rem;
            margin-bottom: 30px;
        }
        .img-container {
            text-align: center;
            margin-bottom: 30px;
        }
        img {
            max-width: 100%; /* Ensures the image is responsive */
            height: auto; /* Prevents image distortion */
            max-height: 300px; /* Limiting the maximum height */
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Welcome to Sign Language Recognition</h1>
        <div class="content">
            Sign language, a visual language using manual, facial, and body movements, is crucial for deaf communication worldwide. 
        </div>
        <div class="img-container">
            <img id="asl-image" src="#" alt="Sign Language Image" style="display: none;">
        </div>
        <!-- <div class="text-center">
            <button id="start-btn" class="btn btn-primary btn-lg">Start</button>
        </div> -->

        <form action="/start" method="post">
            <!-- <input type="submit" value="Start"> -->
            <button type="submit" class="btn btn-primary btn-lg" value="start">Start</button>
        </form>
    </div>

    <!-- Bootstrap JS (Optional) -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // JavaScript for loading the image
        document.addEventListener('DOMContentLoaded', function() {
            var image = document.getElementById('asl-image');
            image.onload = function() {
                image.style.display = 'block';
            };
            image.src = 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/American_Sign_Language_ASL.svg/1920px-American_Sign_Language_ASL.svg.png'; // Placeholder image URL
        });

        // JavaScript for handling the start button click event (Add your logic here)
        document.getElementById('start-btn').addEventListener('click', function() {
            // Add your logic for what happens when the Start button is clicked
            // For example, redirect to another page or start a process
            console.log('Start button clicked');
        });
    </script>
</body>
</html>
