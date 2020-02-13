import argparse 
import socketio 
import eventlet 
from flask import Flask
from PIL import Image
from io import BytesIO
import base64 
import sys 
from  data_augment import preprocess, INPUT_SHAPE
import numpy as np 
from keras.models import load_model
import argparse 
import os 
import shutil 
from datetime import datetime
import cv2 
import matplotlib.image as mpimg
sio = socketio.Server()
app = Flask(__name__)

MAX_SPEED = 25
MIN_SPEED = 7

speed_limit = MAX_SPEED 

# Define the codec and create VideoWriter object

video_name = 'recordvideo.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_name, fourcc, 20.0, (320, 160))

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # get current throttle value 
        throttle = float(data['throttle'])
        # get current steering_angle value  
        steering_angle = float(data['steering_angle'])
        speed = float(data['speed'])
        # image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = Image.open(BytesIO(base64.b64decode(data["image"])))

        try:
            image = np.asarray(image)

            if args.image_folder != '':
                timestamp  = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S%f')[:-3]
                path = os.path.join(args.image_folder, timestamp)
                mpimg.imsave('{}.jpg'.format(path), np.array(image).reshape((160, 320, 3)))
                tmp = mpimg.imread('{}.jpg'.format(path))
                out.write(tmp)
            image = preprocess(image)
            image = np.array([image])
            steering_angle = float(model.predict(image, batch_size=1))
            global speed_limit 
            if speed > speed_limit:
                speed_limit = MIN_SPEED 
            else:
                speed_limit = MAX_SPEED
            
            #     speed = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2
            print(f'{steering_angle}-------{throttle}-------{speed}')
            print("*****************************************************")
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)
        
        
    else:

        sio.emit('manual', data={}, skip_sid=True)

@sio.on('connect')
def connect(sid, environ):
    print("connect", sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__() 
    },
    skip_sid=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
                        'model', 
                        help="Put the trained model into predict function. Model should be the same path.",
                        type=str
                        )
    parser.add_argument(
        'image_folder', 
        help="Path to image folder. This is where the images will be saved.",
        type=str,
        default='',
        nargs='?')
    args = parser.parse_args()

    if args.image_folder != '':
        print(f"Create image folder at {args.image_folder}.")
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")
    model = load_model(args.model)
    app = socketio.Middleware(sio, app)
    IP = "0.0.0.1"
    # PORT = 1234
    # eventlet.wsgi.server(eventlet.listen((IP, 4567)), app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

    out.release()
