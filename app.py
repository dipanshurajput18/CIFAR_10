from flask import Flask, request, render_template, redirect, url_for
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Load your pre-trained model and class names here
model_path = 'CIFAR_10.h5'  # Update this path
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

probability_model = tf.keras.models.load_model(model_path)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # Update with your class names

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((32, 32))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(image_path):
    img = load_and_preprocess_image(image_path)
    predictions = probability_model.predict(img)
    predicted_class = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class]
    return predicted_class_name

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            predicted_class_name = predict_image(filepath)
            return render_template('result.html', predicted_class=predicted_class_name, image_url=filepath, os=os)
    return render_template('upload.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
