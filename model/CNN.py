from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
import numpy as np


def preprocess(_image):
    img = image.load_img(_image, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def encode(_image, model):
    img = preprocess(_image)
    feature_vec = model.predict(img)
    feature_vec = np.reshape(
        feature_vec, feature_vec.shape[1])   # (1,2048)->(2048,)
    return feature_vec


def CNN_model(_image):
    model = InceptionV3(weights='imagenet')
    model = Model(model.input, model.layers[-2].output)
    return encode(_image, model).reshape((1, 2048))
