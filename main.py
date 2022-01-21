import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

st.title('Live Digit Recognition')
st.markdown('Use a neural network to read digits as they are drawn in real time')

@st.cache(allow_output_mutation=True, show_spinner=True, hash_funcs={"MyUnhashableClass": lambda _: None})
def load_model():
    return tf.keras.models.load_model('notebooks/digit_model1.h5')

model = load_model() # Cached model



def predict_digit(image):
    array1 = abs((np.array(image.getdata()).reshape((28,28)) - 255)) 
    # array1 = (np.array(image.getdata()).reshape((28,28)) ) 
    final_img_data = (np.expand_dims(array1, axis=0) / 255.0)
    
    # print(final_img_data, final_img_data.shape)

    prediction = model.predict(final_img_data)

    print('Prediction:', np.argmax(prediction))
    print('Output Array:', prediction)
    return np.argmax(prediction), prediction


stroke_width = st.sidebar.slider("Stroke width: ", 1, 10)
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width * 10 + 25,
    stroke_color=hex(00000),
    background_color=0,
    update_streamlit=True,
    height=400,
    drawing_mode=drawing_mode,
    key="canvas",
)

if canvas_result.image_data is not None:
    data = canvas_result.image_data
    gray_image = (255 - data[:, :, 3])
    # print(gray_image.shape, type(gray_image))
    
    format_image = Image.fromarray(gray_image).convert('F').resize((28,28))
    # print(format_image.getdata())

    # gray_image = ImageOps.grayscale((init_image))
    # print((gray_image))
    if st.button('Predict'):
        number, softmax = predict_digit(format_image)
        st.write(f'`Prediction: {number}`')
        st.write(f'`Confidence: {int(softmax[0][number]*100)}%`')