from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Assuming 'severity_dic' contains labels for severity classification
severity_dic = {0: 'minor', 1: 'moderate', 2: 'severe'}

model = load_model('model1.h5')

model.make_predict_function()

def predict_severity(img_path):
    i = image.load_img(img_path, target_size=(224, 224))
    i = image.img_to_array(i) / 255.0
    i = np.expand_dims(i, axis=0)  # Adding batch dimension
    p = model.predict(i)
    p_class = np.argmax(p, axis=1)[0]  # Assuming multi-class classification
    return severity_dic[p_class]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "Please subscribe Artificial Intelligence Hub..!!!"

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename
        img.save(img_path)

        p = predict_severity(img_path)

    return render_template("index.html", prediction=p, img_path=img_path)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
