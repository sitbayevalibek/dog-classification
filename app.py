import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly
import plotly.express as px
import platform
import torch

plt = platform.system()
if plt =='Linux': pathlib.PosixPath = pathlib.WindowsPath

#title
st.title('Dog classification model')
#rasm joylash
file = st.file_uploader('Upload image', type=['png', 'jpeg', 'jfif', 'svg'])
if file:
    st.image(file)
    #PIL Image convert
    img = PILImage.create(file)
    #model
    model = load_learner('dog_model.pkl')

    #prediction
    pred, probs = model.predict(img)

    st.success(f'Bashorat: {pred}')
    st.info(f'Aniqlik: {probs*100:.2f}%')
    # plotting
    fig=px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)