from flask import Flask, render_template, Response
import cv2
import numpy as np
import random
import time
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load character image with error handling
try:
    character_img_path = os.path.join('static', 'images', 'character.jpg')
    character_img = cv2.imread(character_img_path, cv2.IMREAD_UNCHANGED)
    if character_img is None:
        raise Exception(f"Character image not found at {character_img_path}")
    character_img = cv2.resize(character_img, (150, 150))
except Exception as e:
    logger.error(f"Error loading character image: {str(e)}")
    character_img = np.ones((150, 150, 3), dtype=np.uint8) * 255

# Camera configuration
CAMERA_CONFIG = {
    'resolution': (640, 480),
    'fps': 30,
    'retry_attempts': 3,
    'retry_delay': 1
}

def create_particle(frame, x, y):
    try:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        radius = random.randint(5, 15)
        cv2.circle(frame, (x, y), radius, color, -1)
        return frame
    except Exception as e:
        logger.error(f"Error creating particle: {str(e)}")
        return frame

def add_birthday_text(frame, x, y):
    try:
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, 'Happy Birthday!', (x-50, y-50), 
                    font, 1, (255, 0, 255), 2)
        return frame
    except Exception as e:
        logger.error(f"Error adding birthday text: {str(e)}")
        return frame

def overlay_character(frame, face_x, face_y, face_w, face_h):
    try:
        char_x = face_x + face_w//2 - character_img.shape[1]//2
        char_y = face_y - character_img.shape[0]//2
        
        char_x = max(0, min(char_x, frame.shape[1] - character_img.shape[1]))
        char_y = max(0, min(char_y, frame.shape[0] - character_img.shape[0]))
        
        # Check if ROI dimensions are valid
        if (char_y + character_img.shape[0] <= frame.shape[0] and 
            char_x + character_img.shape[1] <= frame.shape[1]):
            roi = frame[char_y:char_y + character_img.shape[0], 
                       char_x:char_x + character_img.shape[1]]
            
            # Check if ROI and character_img have the same dimensions
            if roi.shape == character_img.shape:
                result = cv2.addWeighted(roi, 0.7, character_img, 0.3, 0)
                frame[char_y:char_y + character_img.shape[0], 
                      char_x:char_x + character_img.shape[1]] = result
        
        return frame
    except Exception as e:
        logger.error(f"Error overlaying character: {str(e)}")
        return frame

def setup_camera(camera):
    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG['resolution'][0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG['resolution'][1])
        camera.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG['fps'])
        return True
    except Exception as e:
        logger.error(f"Error setting up camera: {str(e)}")
        return False

def gen_frames():
    camera = None
    try:
        # Try different camera backends
        backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_AVFOUNDATION]
        
        for backend in backends:
            try:
                camera = cv2.VideoCapture(0 + backend)
                if camera.isOpened() and setup_camera(camera):
                    logger.info(f"Successfully opened camera with backend {backend}")
                    break
            except:
                if camera:
                    camera.release()
                continue
        
        if not camera or not camera.isOpened():
            raise Exception("Could not open any camera")
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            raise Exception("Error loading face cascade classifier")
        
        last_detection_time = 0
        particle_interval = 0.1
        
        while True:
            success, frame = camera.read()
            if not success:
                logger.warning("Failed to read frame")
                continue
            
            try:
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                current_time = time.time()
                
                for (x, y, w, h) in faces:
                    frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    frame = overlay_character(frame, x, y, w, h)
                    frame = add_birthday_text(frame, x, y)
                    
                    if current_time - last_detection_time > particle_interval:
                        for _ in range(5):
                            particle_x = x + random.randint(0, w)
                            particle_y = y + random.randint(0, h)
                            frame = create_particle(frame, particle_x, particle_y)
                        last_detection_time = current_time
                
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                continue
    
    except Exception as e:
        logger.error(f"Critical error in gen_frames: {str(e)}")
    finally:
        if camera is not None:
            camera.release()
            logger.info("Camera released")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/favicon.ico')
def favicon():
    return '', 204  # Return a No Content response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)