from flask import Flask, request, jsonify
from flask_cors import CORS 
from PIL import Image
import os
import tensorflow as tf
import traceback

app = Flask(__name__)
CORS(app) 
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024

# Load a pre-trained model from TensorFlow Hub
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
    tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet', classes=1000)
])
model.trainable = False

# Define a function to preprocess the image
def preprocess_image(image_path):
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        return img_array
    except Exception as e:
        raise ValueError(f"Error in preprocessing image: {str(e)}")

# Define a function to make predictions
def predict_disease(image_path):
    try:
        img_array = preprocess_image(image_path)
        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)[0]
        return decoded_predictions[0][1]
    except Exception as e:
        raise ValueError(f"Error in making predictions: {str(e)}")

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' in request.files:
            image = request.files['image']
            image_path = os.path.join('uploads', image.filename)
            image.save(image_path)

            # Perform disease prediction
            disease_prediction = predict_disease(image_path)

            result = {'status': 'success', 'message': 'Image uploaded and processed successfully', 'prediction': disease_prediction}
            return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        result = {'status': 'error', 'message': str(e), 'traceback': traceback.format_exc()}
        return jsonify(result), 500

if __name__ == '__main__':
    app.run(debug=True)
