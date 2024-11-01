from flask import Flask, render_template, Response
import cv2
import numpy as np
import random
import time

app = Flask(__name__)

# Load character image
character_img = cv2.imread('static/images/character.jpg', cv2.IMREAD_UNCHANGED)
if character_img is None:
    raise Exception("Character image not found!")

# Resize character image
character_img = cv2.resize(character_img, (150, 150))

def create_particle(frame, x, y):
    # Create colorful circle particles
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    radius = random.randint(5, 15)
    cv2.circle(frame, (x, y), radius, color, -1)
    return frame

def add_birthday_text(frame, x, y):
    # Add birthday text with different colors and sizes
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, 'Happy Birthday!', (x-50, y-50), 
                font, 1, (255, 0, 255), 2)
    return frame

def overlay_character(frame, face_x, face_y, face_w, face_h):
    # Calculate position for character (behind the face)
    char_x = face_x + face_w//2 - character_img.shape[1]//2
    char_y = face_y - character_img.shape[0]//2
    
    # Ensure character stays within frame bounds
    char_x = max(0, min(char_x, frame.shape[1] - character_img.shape[1]))
    char_y = max(0, min(char_y, frame.shape[0] - character_img.shape[0]))
    
    # Create ROI and overlay character
    roi = frame[char_y:char_y + character_img.shape[0], 
                char_x:char_x + character_img.shape[1]]
    
    # Blend character with background
    result = cv2.addWeighted(roi, 0.7, character_img, 0.3, 0)
    frame[char_y:char_y + character_img.shape[0], 
          char_x:char_x + character_img.shape[1]] = result
    
    return frame

def gen_frames():
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize particle system
    particles = []
    last_detection_time = 0
    particle_interval = 0.1  # seconds between particle bursts
    
    while True:
        success, frame = camera.read()
        if not success:
            break
            
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        current_time = time.time()
        
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Add character behind face
            frame = overlay_character(frame, x, y, w, h)
            
            # Add birthday text
            frame = add_birthday_text(frame, x, y)
            
            # Add particles at intervals
            if current_time - last_detection_time > particle_interval:
                # Add multiple particles around the face
                for _ in range(5):
                    particle_x = x + random.randint(0, w)
                    particle_y = y + random.randint(0, h)
                    frame = create_particle(frame, particle_x, particle_y)
                last_detection_time = current_time
        
        # Convert frame to jpg
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)