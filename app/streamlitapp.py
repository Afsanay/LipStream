import streamlit as st
import os
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model
import sys

st.set_page_config(layout='wide')

with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title("LipRead")
    st.info("This application in developed for the people with hearing disability")

st.title('LipNet Full Stack App')
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
