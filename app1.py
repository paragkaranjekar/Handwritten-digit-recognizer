from wtforms.validators import DataRequired
from wtforms import FileField, SubmitField
from flask_wtf import FlaskForm
from flask import Flask, render_template, request
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

model = load_model('hdr_model.h5')
img_size = 28


def find_digit(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = 120
    gray[gray > threshold] = 255
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    resized = cv2.resize(thresh, (img_size, img_size),
                         interpolation=cv2.INTER_AREA)
    newimg = tf.keras.utils.normalize(resized, axis=1)
    newimg = np.array(newimg).reshape(-1, img_size, img_size, 1)
    predictions = model.predict(newimg)
    return str(np.argmax(predictions))


app = Flask(__name__)
app.config['SECRET_KEY'] = 'code-engine'


class MyForm(FlaskForm):
    image = FileField('Image', validators=[DataRequired()])
    submit = SubmitField('Submit')


@app.route('/', methods=['GET', 'POST'])
def home():
    form = MyForm()
    if form.validate_on_submit():
        # Process the uploaded image file here
        image_file = form.image.data
        filename = secure_filename(image_file.filename)
        nparr = np.fromstring(image_file.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Save the file or perform any required operations

        # Perform the prediction and return the result
        prediction = find_digit(image)
        print(prediction)
        return render_template('index.html', form=form, prediction=prediction)
    return render_template('index.html', form=form)


if __name__ == '__main__':
    app.run()
