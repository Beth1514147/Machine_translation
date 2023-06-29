import numpy as np
from model import *
import streamlit as st
# Predict new value:
def decode_sequence(input_sentence,eng_vectorization,vie_vectorization,transformer,type):
    if type == 'V2E':
        eng_vocab = eng_vectorization.get_vocabulary()
        eng_index_lookup = dict(zip(range(len(eng_vocab)), eng_vocab))
        max_decoded_sentence_length = 20
        tokenized_input_sentence = vie_vectorization([input_sentence])
        decoded_sentence = "[start]"
        for i in range(max_decoded_sentence_length):
            tokenized_target_sentence = eng_vectorization([decoded_sentence])[:, :-1]
            # print(tokenized_target_sentence)
            predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])

            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = eng_index_lookup[sampled_token_index]
            decoded_sentence += " " + sampled_token

            if sampled_token == "[end]":
                break
    elif type == 'E2V':
        eng_vocab = vie_vectorization.get_vocabulary()
        eng_index_lookup = dict(zip(range(len(eng_vocab)), eng_vocab))
        max_decoded_sentence_length = 20
        tokenized_input_sentence = eng_vectorization([input_sentence])
        decoded_sentence = "[start]"
        for i in range(max_decoded_sentence_length):
            tokenized_target_sentence = vie_vectorization([decoded_sentence])[:, :-1]
            # print(tokenized_target_sentence)
            predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])

            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = eng_index_lookup[sampled_token_index]
            decoded_sentence += " " + sampled_token

            if sampled_token == "[end]":
                break
        
    return decoded_sentence


def show_translation(from_language,to_language,input_text,new_vectorizer_en,new_vectorizer_vi,model,type):
    if from_language == 'English' and to_language == 'Vietnamese':
        if input_text is not None:
            input_text= input_text.lower()
            a= decode_sequence(input_text,new_vectorizer_en,new_vectorizer_vi,model,type = 'E2V')
            st.write(a)
    elif from_language == 'Vietnamese' and to_language == 'English':
        if input_text is not None:
            input_text= input_text.lower()
            a= decode_sequence(input_text,new_vectorizer_en,new_vectorizer_vi,model,type = 'V2E')
            st.write(a)
    elif from_language == to_language : 
        if input_text is not None:
            a = input_text
            st.write(a)
    return a