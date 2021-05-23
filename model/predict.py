from PIL import Image
from pickle5 import load
from tensorflow.keras.models import Model
from tensorflow.keras import Input, layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from tensorflow.keras.layers import add
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3
import numpy as np
from numpy import array

# importing word to idx and viceversa

from model.pickles import wordtoix, ixtoword
from model.CNN import CNN_model
from model.RNN import RNN_model

# const
max_train_sequence_length = 34
vocab_size = 1652
embedding_dim = 200


def greedySearch(photo):
    in_text = 'startseq'
    model = RNN_model()
    for i in range(max_train_sequence_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_train_sequence_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


def get_caption(image):
    photo = CNN_model(image)
    caption = greedySearch(photo)
    return caption
