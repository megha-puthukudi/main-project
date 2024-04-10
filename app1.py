from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

damage_models = {
    'damage_or_not': load_model('model.h5'),
    'damage_severity': load_model('model1.h5')
}

damage_or_not_labels = {0: 'Damage', 1: 'no Damage'}
damage_severity_labels = {0: 'Minor', 1: 'Moderate', 2: 'Severe'}

def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def predict_car_damage(file_path):
    img = preprocess_image(file_path, target_size=(224, 224))
    prediction = damage_models['damage_or_not'].predict(img)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = damage_or_not_labels[predicted_class_index]
    return predicted_class_label, img

def predict_damage_severity(file_path):
    img = preprocess_image(file_path, target_size=(224, 224))
    prediction = damage_models['damage_severity'].predict(img)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = damage_severity_labels[predicted_class_index]
    return predicted_class_label, img

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("ind1.html")

@app.route("/about")
def about_page():
    return "Please subscribe Artificial Intelligence Hub..!!!"

@app.route("/submit", methods=['POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)

        damage_or_not, _ = predict_car_damage(img_path)
        if damage_or_not == 'Damage':
            damage_severity, _ = predict_damage_severity(img_path)
        else:
            damage_severity = 'N/A'

        return jsonify({
            'damage_or_not': damage_or_not,
            'damage_severity': damage_severity,
            'img_path': img_path
        })

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
