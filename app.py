from flask import Flask, render_template, request
import sys
import os.path
import cv2
from tensorflow.keras.models import load_model
import numpy as np


TEMPLATE_DIR = os.path.abspath('../Flask/templates')
STATIC_DIR = os.path.abspath('../Flask/static')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def after():
    img = request.files['file1']
    path = os.path.abspath('../Flask/static/css/images/file.jpg')
    weight_path = os.path.abspath('../Flask/model_weights.h5')
    # path = r'D:\Microsoft VS Code\python\Flask\anh-stt-vui.jpg'
    img.save(path)

    
    image = cv2.imread(path) 
    # gray = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    cascade = cv2.CascadeClassifier('model.xml')
    faces = cascade.detectMultiScale(gray, 1.1, 2)
    
    model = load_model(weight_path)
    
    label_map = ['Angry', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']
    

    for i, (x, y, w, h) in enumerate(faces):
        cropped = image[y: y+h, x: x + w]
        
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite('static/xx.jpg', cropped)
        cropped = cv2.resize(cropped, (48, 48))
        cropped = np.reshape(cropped, (1, 48, 48, 1))
        pred = model.predict(cropped)
        prediction = np.argmax(pred)
        final_pred = label_map[prediction]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, final_pred, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imwrite('static/after.jpg', image)

    # gray = cv2.imread(path, 0) 
    
    return render_template('after.html', data=final_pred)

if __name__ == '__main__':
    app.run(debug=True)
