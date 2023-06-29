# Import library
import streamlit as st
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
from clean_data import *
import pickle

# Set-up page config:
st.set_page_config(
    page_title="Data Science Capstone Project", 
    initial_sidebar_state="expanded"
)

st.sidebar.image('images/logo.png')
st.sidebar.title('Data Science Capstone Project')

# ------------------------------------------- #
#        LOAD MODEL, VECTORIZE FACTOR        #
# ------------------------------------------#

@st.cache_resource
def loading_model(path):
    model = load_model(path,
                        custom_objects={'KerasLayer':hub.KerasLayer,
       'PositionalEmbedding': PositionalEmbedding,
       'TransformerEncoder': TransformerEncoder,
       'TransformerDecoder': TransformerDecoder,
       }
       )
    return model


model_Vi2Eng = loading_model('models/Bản sao của vie_eng_M3.h5')
model_Eng2Vi = loading_model('models/Bản sao của eng_vie_M3.h5')


# Loading vectorization factor:

eng_vectorization_en2Vi= pickle.load(open('models/eng_vec_2.pkl','rb'))
vie_vectorization_en2Vi = pickle.load(open('models/vie_vec_2.pkl','rb'))

new_vectorizer_en_en2Vi = TextVectorization(max_tokens=eng_vectorization_en2Vi['config']['max_tokens'],
                                          output_mode='int',
                                          output_sequence_length=eng_vectorization_en2Vi['config']['output_sequence_length'])
new_vectorizer_en_en2Vi.set_weights(eng_vectorization_en2Vi['weights'])


new_vectorizer_vi_en2Vi = TextVectorization(max_tokens=vie_vectorization_en2Vi['config']['max_tokens'],
                                          output_mode='int',
                                          output_sequence_length=vie_vectorization_en2Vi['config']['output_sequence_length'])
new_vectorizer_vi_en2Vi.set_weights(vie_vectorization_en2Vi['weights'])


eng_vectorization_Vi2en= pickle.load(open('models/eng_vec_final.pkl','rb'))
vie_vectorization_Vi2en = pickle.load(open('models/vie_vec_final.pkl','rb'))

new_vectorizer_en_vi2En = TextVectorization(max_tokens=eng_vectorization_Vi2en['config']['max_tokens'],
                                          output_mode='int',
                                          output_sequence_length=eng_vectorization_Vi2en['config']['output_sequence_length'])
new_vectorizer_en_vi2En.set_weights(eng_vectorization_Vi2en['weights'])


new_vectorizer_vi_vi2En = TextVectorization(max_tokens=vie_vectorization_Vi2en['config']['max_tokens'],
                                          output_mode='int',
                                          output_sequence_length=vie_vectorization_Vi2en['config']['output_sequence_length'])
new_vectorizer_vi_vi2En.set_weights(vie_vectorization_Vi2en['weights'])

st.write('Loading successful')

# +------------------------------------+ 
# |            GUI                    |
# +-----------------------------------+


menu = ("Project overview" , "Data preparation & Modelling", 'New Translation')
choice = st.sidebar.selectbox('Content', menu)

# Project intruction
if choice == menu[0]: 
    st.title('PROJECT OVERVIEW')
    st.caption('This is a capstone project from Data Science & Machine learning course')
    st.caption('Team member:')
    st.caption('1. Huy Nhat Bui')
    st.caption('2. Minh Van Bui')
    st.caption('3. Vy Nguyen Tran Ha ')

    
    st.divider()

    st.header("1. Business Objective/Problem:")
    st.write('A new request for translation function from English - Vietnamese and Vietnamese - English')
    st.image('images/e-translator.png')
    st.image('images/introduction.png')

    st.header("2. Machine translation:")
    st.markdown('#### What is "Machine transaltion": ')
    st.write('Machine translation is the application of computers to the task of translating texts from one natural language to another')

    st.header("3. Transformer model: ")
    st.markdown('#### What is "Transformer": ')
    st.write('A transformer is a deep learning model, using the recently discovered self-attention mechanism,notable for requiring less training time compared to older long short-term memory (LSTM) models,thus enabling large (language) datasets, such as the Wikipedia Corpus and Common Crawl, to be used for training due to parallelization.The model processes all tokens, parsed by a byte pair encoding, simultaneously and subsequently calculates attention weights between them in successive layers. The augmentation of seq2seq models with the transformer attention mechanism was first implemented in the context of machine translation by Bahdanau, Cho, and Bengio in 2014.The model is now used not only in natural language processing, computer vision, but also in audio, and multi-modal processing.')
    st.write('Transformers were developed to solve the problem of sequence transduction, or neural machine translation. That means any task that transforms an input sequence to an output sequence. This includes speech recognition, text-to-speech transformation, etc..')
    st.video('images/seq2seq_1.mp4')
    st.markdown('#### Looking under the hood ')
    st.write('The model is composed of an encoder and a decoder')
    st.write('The encoder processes each item in the input sequence, it compiles the information it captures into a vector (called the context). After processing the entire input sequence, the encoder sends the context over to the decoder, which begins producing the output sequence item by item.')
    st.video('images/seq2seq_4.mp4')

    st.header("4. Reference:")
    st.write("a. Coursework from Data Science and Machine learning course")
    st.write('b. Internet sources:')

    st.write ('- https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/')
    st.write("- https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)#:~:text=A%20transformer%20is%20a%20deep,used%20for%20training%20due%20to")

# Data understanding & modelling:
if choice == menu[1]:
    st.title('Data Preparation & Modelling for machine translation')
    tab1,tab2 = st.tabs(['Data Preparation','Modelling'])

    with tab1: # Data preparation 
        st.title('Data preparation:')
        st.header('Input:')
        cols = st.columns(2)
        with cols[0]:
            
            st.write('a. Team was provided 2 datasets, with details:')
            st.write('Pair of dataset covers:')
            st.write('▪ en_sents.csv ~ 254,089 Eng sentences')
            st.write('▪ en_sents.csv ~ 254,089 Eng sentences')
        with cols[1]:
            st.image('images/input.png')


        st.header('Data preparation process:')
        st.image('images/process.png')
    with tab2: # Modeling
        st.title('Modelling and results:')
        st.header('1.Eng-Vie Translation Model:')
        # col1, col2, col3 = st.columns(3)
        st.image('images/model_1.png')
        st.write('Results:')
        st.image('images/result_model1.png')
        st.header('2.Vie-Eng Translation Model:')
        # col1, col2, col3 = st.columns(3)
        st.image('images/model_2.png')
        st.write('Results:')
        st.image('images/result_model2.png')
            
        


if choice == menu[2]:
    a = None
    input_text = None
    st.title('New Translation:')
    st.markdown('This function support to translate words, single sentence or muli-sentences from English to Vietnamese or backwards')
    st.divider() 

    option = st.radio('Choose option single sentence or muliple sentences', ('Single','Multiple'))

    
    with st.expander(" Click here for single words/sentences translation"):
        cols = st.columns(2)
        from_language = cols[0].selectbox('From:',('English','Vietnamese'))
        to_language = cols[1].selectbox('To:',('Vietnamese','English'))
        if option == 'Single':
            input_text = st.text_input('Input your sentences, press Enter:')

            st.write('You want to translate from {} to {} sentence: {}'.format(from_language,to_language,input_text))

            if from_language == 'English' and to_language == 'Vietnamese':
                if input_text is not None:
                    input_text= input_text.lower()
                    a= decode_sequence(input_text,new_vectorizer_en_en2Vi,new_vectorizer_vi_en2Vi,model_Eng2Vi,type = 'E2V')
                    st.write(a)
            elif from_language == 'Vietnamese' and to_language == 'English':
                if input_text is not None:
                    input_text= input_text.lower()
                    a= decode_sequence(input_text,new_vectorizer_en_vi2En,new_vectorizer_vi_vi2En,model_Vi2Eng,type = 'V2E')
                    st.write(a)
            elif from_language == to_language : 
                if input_text is not None:
                    a = input_text
                    st.write(a)
        
        if option == 'Multiple' :
            input_text = st.text_area('Input your sentences, press Ctrl Enter:')
            input_text = input_text.split('.')
            for item in input_text:
                if from_language == 'English' and to_language == 'Vietnamese':
                    if input_text is not None and len(item) >0:
                        item= clean_data(item)
                        a= decode_sequence(item,new_vectorizer_en_en2Vi,new_vectorizer_vi_en2Vi,model_Eng2Vi,type = 'E2V')
                        st.write(a)
                elif from_language == 'Vietnamese' and to_language == 'English':
                    if input_text is not None and len(item) >0:
                        item= clean_data(item)
                        a= decode_sequence(item,new_vectorizer_en_vi2En,new_vectorizer_vi_vi2En,model_Vi2Eng,type = 'V2E')
                        st.write(a)
                elif from_language == to_language : 
                    if input_text is not None and len(item) >0:
                        a = input_text
                        st.write(a)
                st.write()



    


