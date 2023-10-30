import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
import base64
import joblib

UPLOAD_FOLDER = './upload'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
# Replace with your model path
model = load_model(
    'models/flowers_model.h5')
le = joblib.load(
    'models/label_encoder.pkl')

# Define a function to predict the class of an input image


def predict_flower_class(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    predicted_flower = le.inverse_transform([predicted_class])[0]
    return predicted_flower


@app.route('/', methods=['GET', 'POST'])
def index():
    flower_prediction = None
    image_data = None
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file:
                path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(path)
                flower_prediction = predict_flower_class(path)
                # Read the image as bytes and convert to base64
                with open(path, 'rb') as image_file:
                    image_data = base64.b64encode(
                        image_file.read()).decode('utf-8')
                return redirect(url_for('result', flower_prediction=flower_prediction, image_data=image_data))

    return render_template('index.html')


@app.route('/result', methods=['GET', 'POST'])
def result():
    # Retrieve the data from query parameters
    flower_prediction = request.args.get('flower_prediction')
    image_data = request.args.get('image_data')

    return render_template('result.html', flower_prediction=flower_prediction, image_data=image_data)


if __name__ == '__main__':
    app.run(debug=True)
