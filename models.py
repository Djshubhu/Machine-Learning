import tensorflow as tf
import cv2
import numpy as np

def preprocess_image(image):
    # Preprocess the image as required by your model
    image = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256))
    # image = image / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)
         

def get_prediction(image):
    model = tf.keras.models.load_model("S:\Machine Learning\Machine_Learning_model\Cat_dog\cat_dog.h5")
    preprocessed_image = preprocess_image(image)
    # print(preprocess_image)
    prediction = model.predict(preprocessed_image)
    # class_names = ['Class 0', 'Class 1', 'Class 2']  # Replace with your class names
    # predicted_class = np.argmax(prediction)
    # predicted_class_name = class_names[predicted_class]
    Flag=None
    if prediction<0.5:
        Flag='Cat'
    if prediction>0.5:
        Flag='Dog'
    # else:
    #     Flag='Other'
    return Flag