import streamlit as st
import os
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model
import sys

st.set_page_config(layout='wide')

with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title("LipStream")
    st.info("This application in developed for the people with hearing disability")

st.title('LipStream for lip-reading')

tab1,tab2 = st.tabs(["Demonstration","About"])
with tab1:
    options = os.listdir(os.path.join('./app','data', 's1'))
    selected_video = st.selectbox('Choose the video',options=options)
    col1, col2 = st.columns(2)

    if options:
        with col1:
            st.info('The video below is to be converted')
            file_path = os.path.join('./app','data', 's1', selected_video)
            file_name = selected_video.split('.')[0]
            mp4_path = os.path.join('./app','mp4',f'{file_name}.mp4')
            video = open(mp4_path, 'rb')
            video_bytes = video.read()
            st.video(video_bytes)

        with col2:
            video, annotations = load_data(tf.convert_to_tensor(file_path))
            st.info('This is the output of the machine learning model')
            model = load_model()
            yhat = model.predict(tf.expand_dims(video, axis=0))
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            st.text(decoder)
            st.info('Decode the raw tokens to words')
            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            st.text(converted_prediction)

with tab2:
    st.markdown("""
                <p style="font-size:20px; text-align:center">LipNet is an architecture specifically designed for lip-reading, which aims to convert sequences of lip movements into corresponding textual transcriptions. The LipNet architecture combines three main components: a spatiotemporal convolutional neural network (CNN), a recurrent neural network (RNN), and a connectionist temporal classification (CTC) loss function. The CTC loss function is employed to train the LipNet architecture. CTC allows the network to learn directly from sequences of lip movements without requiring aligned phoneme labels.</p>
                <img style="display: block; margin-left: auto; margin-right: auto;padding: 5px;" src="https://www.researchgate.net/publication/357309047/figure/fig1/AS:1104647620169757@1640379924865/Schematic-design-of-VSR-architecture-a-LipNet-architecture-baseline-b-four.png" />
                """,unsafe_allow_html=True)