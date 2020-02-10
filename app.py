from flask import Flask, render_template, Response, jsonify, request
from camera import VideoCamera
from camera import predict
from data import Emotions
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

app = Flask(__name__)

video_camera = VideoCamera()
global_frame = None

Emotions = Emotions()

# ===================FOR DEVELOPMENT ONLY=================
app.debug = True
# =========================================

@app.route('/whatemotion')
def emotion():
    return render_template('emotions.html', emotion=predict())

# VIDEO RECORDING


@app.route('/record_status', methods=['POST'])
def record_status():
    global video_camera 
    

    json = request.get_json()

    status = json['status']

    if status == "true":
        video_camera.start_record()
        return jsonify(result="started")
    else:
        video_camera.stop_record()
        return jsonify(result="stopped")

def video_stream():
    global video_camera 
    global global_frame

    if video_camera == None:
        video_camera = VideoCamera()
        
    while True:
        frame = video_camera.get_frame()

        if frame != None:
            global_frame = frame
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')

@app.route('/video_viewer')
def video_viewer():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



#########################################
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')




# --bottom--
if __name__ == '__main__':
    app.run()
