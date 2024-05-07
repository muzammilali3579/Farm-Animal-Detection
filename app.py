from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as img_prep

app = Flask(__name__)

# Load your trained model
model = load_model('animal_recognition_model.h5')

# Define classes if applicable
classes =['ArmaDillo', 'Bear', 'Birds', 'Cow', 'Crocodile', 'Deer', 'Elephant', 'Goat', 'Horse', 'Jaguar', 'Monkey', 'Rabbit', 'Skunk', 'Tiger', 'Wild Boar']

def preprocess_image(image_path):
    img = Image.open(image_path)
    
    # Convert PNG to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize((224, 224))  # Adjust input size as per your model requirement
    img = img_prep.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values
    return img

# Define function to predict
def predict(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    confidence = np.max(prediction)
    if confidence > 0.85:
        predicted_class = "Predicted class:"  # Replace this with the logic to get the predicted class
        # If classes are defined
        predicted_class = classes[np.argmax(prediction)]
        confidence=confidence*100
        confidences=int(confidence)
        return predicted_class,confidences
    else:
        return "No animal predicted","null"


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if request.method == 'POST':
        file = request.files['file']
        image_path = "temp_image.jpg"  # Temporary image path
        file.save(image_path)
        predicted_class, confidence = predict(image_path)
        return f"Predicted Class: {predicted_class} \n Confidence: {confidence}%"

if __name__ == '__main__':
    app.run()

