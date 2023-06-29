
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import tensorflow_hub as hub
from sklearn.metrics import confusion_matrix, classification_report
from model import*
from callbacks import *
import pickle

def loading_model(path):
    model = load_model(path, custom_objects={'PositionalEmbedding':PositionalEmbedding,
     'TransformerEncoder':TransformerEncoder,
     'TransformerDecoder':TransformerDecoder})
    return model


model_Vi2Eng = loading_model('models/Bản sao của vie_eng_M3.h5')
model_Eng2Vi = loading_model('models/Bản sao của eng_vie_M3.h5')


eng_vectorization= pickle.load(open('models/eng_vec_2.pkl','rb'))
vie_vectorization = pickle.load(open('models/vie_vec_2.pkl','rb'))

new_vectorizer_en = TextVectorization(max_tokens=eng_vectorization['config']['max_tokens'],
                                          output_mode='int',
                                          output_sequence_length=eng_vectorization['config']['output_sequence_length'])
new_vectorizer_en.set_weights(eng_vectorization['weights'])


new_vectorizer_vi = TextVectorization(max_tokens=vie_vectorization['config']['max_tokens'],
                                          output_mode='int',
                                          output_sequence_length=vie_vectorization['config']['output_sequence_length'])
new_vectorizer_vi.set_weights(vie_vectorization['weights'])
import numpy as np
from model import *
# Predict new value:
def decode_sequence(input_sentence,eng_vectorization,vie_vectorization,transformer):
    eng_vocab = eng_vectorization.get_vocabulary()
    eng_index_lookup = dict(zip(range(len(eng_vocab)), eng_vocab))
    max_decoded_sentence_length = 20
    tokenized_input_sentence = vie_vectorization([input_sentence])
    decoded_sentence = "[start]"
    print(1)
    for i in range(max_decoded_sentence_length):
        print(2)

        tokenized_target_sentence = eng_vectorization([decoded_sentence])[:, :-1]
        print(tokenized_target_sentence)
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])

        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = eng_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break
    return decoded_sentence





input ='Hello'
print(decode_sequence(input,new_vectorizer_en,new_vectorizer_vi,model_Eng2Vi))