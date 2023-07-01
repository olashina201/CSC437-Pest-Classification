import os
import numpy as np
from keras.models import load_model
import cv2
import sys
import config

# Load the trained model
my_model = load_model(os.path.join("models", "ICT-w-max.hdf5"))

import pickle
with open('le.pkl', 'rb') as file:
    le = pickle.load(file)

# Read image from command line argument
image_org = cv2.imread(sys.argv[1])

# Resize image
image = cv2.resize(image_org, dsize=(config.image_size, config.image_size))

# Convert image to tensor format
image = np.expand_dims(image, axis=0)

# Perform prediction
predict = my_model.predict(image)

# Get the predicted class index
predicted_class_index = np.argsort(predict)[0][-2]

# Get the predicted class label
predicted_class_label = le.inverse_transform([predicted_class_index])[0]

print(predict)
print(np.argsort(predict))
print(predicted_class_index)
print(predicted_class_label)
print("This picture is:", predicted_class_label)

# Display the image with the predicted label
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2

cv2.putText(image_org, predicted_class_label, org, font, fontScale, color, thickness, cv2.LINE_AA)

cv2.imshow("Result", image_org)
cv2.waitKey()
