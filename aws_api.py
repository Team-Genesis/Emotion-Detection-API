from flask import Flask, render_template, Response, jsonify, request, send_file
from flask_sqlalchemy import  SQLAlchemy
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from io import BytesIO
import random
import json

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////home/ishant/ishant_linux/boongg_project/web_api/database/filesstorage.db'
# db = SQLAlchemy(app)

#load model
model = model_from_json(open("fer-colab30.json", "r").read())
#load weights
model.load_weights('fer-colab30.h5')

#face_cascade = cv2.CascadeClassifier('/home/ishant/anaconda3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# class FileContents(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     fileName = db.Column(db.String(300))
#     data = db.Column(db.LargeBinary)
#     predicted_emotion = db.Column(db.String(300))
#     correct = db.Column(db.Boolean())
@app.route('/')
def hello():
    return "Hello World!"

@app.route('/upload', methods=['POST'])
def upload():
    # file = request.files['inputFile']
    # file_Name = file.filename
    # img_str = file.read()
    # newFile = FileContents(fileName=file_Name, data=file.read())
    # db.session.add(newFile)
    # db.session.commit()
    # file_data = FileContents.query.filter_by(id=newFile.id).first()
    
        
    test_img = cv2.imdecode(np.fromstring(request.files['file'].read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    # nparr = np.frombuffer(img_str, np.uint8)
    # test_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    faces_detected = face_cascade.detectMultiScale(test_img, 1.32, 5)



    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=test_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255
        import keras.backend.tensorflow_backend as tb
        tb._SYMBOLIC_SCOPE.value = True

        predictions = model.predict(img_pixels)
        
        

            #find max indexed array
        max_index = np.argmax(predictions[0])

        print(max_index)
        
        accuracy = predictions[0][max_index] * 100

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        print(emotions)
        # print(predicted_emotion)

        if predicted_emotion in ('angry', 'disgust', 'fear', 'sad'):
          predicted_emotion = 'Not Interested'
        elif predicted_emotion == 'neutral':
            predicted_emotion = "Neutral"
        else:
          predicted_emotion = 'Interested'

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        # print(predicted_emotion)

    resized_img = cv2.resize(test_img, (1000, 700))
    
    img_name = predicted_emotion + str(random.randrange(1, 10000000, 1)) + '.png'
    
    img_dir = 'static/' + img_name
    
    x = {
        "Image:" : img_name,
        "EMOTION:" : predicted_emotion,
        "ACCURACY:" : accuracy
    }
    
    json_result = json.dumps(x)
    
    cv2.imwrite(img_dir, resized_img)
    
    return json_result



# --bottom--
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
