from flask import Flask, request, render_template
import base64
import cv2
import numpy as np

img_size = 28
w1 = np.load('w1.npy')
b1 = np.load('b1.npy')
w2 = np.load('w2.npy')
b2 = np.load('b2.npy')


def find_digit(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = 120
    gray[gray > threshold] = 255
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    resized = cv2.resize(thresh, (img_size, img_size),
                         interpolation=cv2.INTER_AREA)

    image_array = np.array(resized)
    image_array = image_array.reshape(784, 1)

    def reLU(z):
        return np.maximum(z, 0)

    def softmax(z):
        return np.exp(z)/sum(np.exp(z))

    def forward_prop(w1, b1, w2, b2, x):
        z1 = w1.dot(x) + b1
        a1 = reLU(z1)
        z2 = w2.dot(a1) + b2
        a2 = softmax(z2)
        return a2

    A2 = forward_prop(w1, b1, w2, b2, image_array)
    predictions = np.argmax(A2, 0)

    return predictions[0]


app = Flask(__name__)
app.config['SECRET_KEY'] = 'code'


@app.route('/', methods=['POST', 'GET'])
def handle_upload():
    # Handle the data URL here
    data_url = request.form.get('data_url')
    answer = ""
    # Process the data URL as needed
    if data_url:
        image_data = data_url.split(',')[1]
        decoded_image_data = base64.b64decode(image_data)
        image_array = np.frombuffer(decoded_image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        answer = find_digit(image)

    return render_template('index.html', results=answer)


if __name__ == '__main__':
    app.run()
