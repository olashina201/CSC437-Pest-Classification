from flask import Flask, render_template, request
from keras.models import load_model
import os
import pickle
import cv2
import config
import numpy as np

app = Flask(__name__)

# Load models
model_paths = {
    'VGG16': os.path.join('models', 'VGG16-f-w-max.hdf5'),
}

loaded_models = {}
for model_name, model_path in model_paths.items():
    loaded_models[model_name] = load_model(model_path)

# Load label encoder
with open('le.pkl', 'rb') as file:
    le = pickle.loads(file)


def process_image(image):
    # Resize image
    frame = cv2.resize(image, dsize=(config.image_size, config.image_size))
    # Convert to tensor format
    frame = np.expand_dims(frame, axis=0)
    return frame


def predict_image(image, model):
    predict = model.predict(image)
    list_pred = np.argsort(predict)
    predict_name = le.inverse_transform([np.argmax(predict)])[0]
    more = "{} : {:.2f}%; {} : {:.2f}%; {} : {:.2f}%".format(
        le.inverse_transform([np.argmax(predict)])[0],
        predict[0][list_pred[0][-1]] * 100,
        le.inverse_transform([list_pred[0][-2]])[0],
        predict[0][list_pred[0][-2]] * 100,
        le.inverse_transform([list_pred[0][-3]])[0],
        predict[0][list_pred[0][-3]] * 100
    )
    return predict_name, more


@app.route("/", methods=['GET', 'POST'])
def home_page():
    if request.method == "POST":
        try:
            # Get uploaded file
            image_file = request.files['file']
            if image_file:
                # Save file
                path_to_save = os.path.join("static/file", image_file.filename)
                image_file.save(path_to_save)

                # Read image
                frame = cv2.imread(path_to_save)
                processed_image = process_image(frame)

                # Run selected model
                selected_model = request.form.get("models")
                if selected_model in loaded_models:
                    predict_name, more = predict_image(
                        processed_image, loaded_models[selected_model])
                else:
                    return render_template('index.html', msg='Invalid model selection')

                return render_template("index.html",
                                       image=image_file.filename,
                                       msg="Upload successful",
                                       models=selected_model,
                                       predict_name=predict_name,
                                       more=more)
            else:
                return render_template('index.html', msg='Please choose a file to upload')
        except Exception as ex:
            print(ex)
            return render_template('index.html', msg="Failed to recognize the image!")
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
