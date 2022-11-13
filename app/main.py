import streamlit as st
import numpy as np
import cv2
import imutils
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import gdown
import copy

st.set_page_config(
    page_title="CSE3505 MRI Classifier",
    page_icon=":brain:",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "CSE3505 J Component Final Review - MRI Brain Tumor Detection by Siddharth Suresh, Harsh Deshwal, Kanishka Ghosh",

    }
)

@st.cache
def load_h5_model():
    # urllib.request.urlretrieve('https://doc-0k-1g-docs.googleusercontent.com/docs/securesc/qg8o8rudll8uros581cppi7lavb7pha0/b4dkjrff1df0a76ehqgafcui991k1gba/1668339825000/01960677327599930294/12654046349221109949/1eVTq8I0uUgxgTqB5e6LA13wF99manJ6j?e=download&ax=AEKYgyQzyzexsNKMLMY9pjVLmboYKmyPMkYR536Sq4OnvuHE3YgHC-nSnzBfw7N1Y4A6G1DNMKdece6Cguc_2cb4ghsPMWmm71atQPvTzf-awQq9OT2ztrTFWDmKLngpp6GsnA3M8qS4WWLa5H1GZY8OoMMy2lBMBT3N76SJ0aguYscEe6WXxRWDGLPRby_06tZd60ZnxCpoevne8xFXLXzR5ATAKa-4ZUWFvy5XITIfLzQ4RLbpe_P39YOVhpnRtOEpO1_yF4QXjDzps58VBgJRt_XhH-ioCDyy_bDIUht-wR-E5kIUhwHj3ut_Jdm4mnDwOO72cm7ZkLY7VaDJSVXjB8LdfHsIXnFARg0tMrMFSKz9kusG7FdMlJD8uOunfNsmwZbEWABksRKXjKcleR_wZfyRGzbFAxq1yPSuSWsaCwZdoftFAoDiyztk4iHsA_fxr7lkJuv9x2kP6zh5pN0jaMi9LEGrx_olRbVWoZ3c48WX31zD0zgs0Ex-y6kfhBDKTuqMtdejdWArInjYlIzRzxfLOv60m0VefNsqhSM1YqG5NKx9BaM49ulc_9t8jWgxq2FiQmXMqhqn3IcyNYFhLxUI_b5OXMvdOO8hGlNaRqPOe1xrLzYDToynZ93bBQSYk7TguYky7Y3qJfnbAcqq8IFbPnfqFBUtYSXTtBRWNWFlJIzl550wSHxh0rqo2pU7Vc_XVqnDsx8btnBFc0r41JAr7vw4CZXB1SbSV3N8PK9koq9hyH2oknq4-mkeA4s-ji0Hs-tNEB8u1EblljSQGuJ2LHgp7XuABwNqwxsmg8TJ1wB2u6HadeGC0YdjQai318g3_lYZ5YgqCX9K0RQzYquru2wRu_lJYV9yod_hoehhQM91qrnnecendui39ZdktA&uuid=fafc07ea-1c4e-43df-8b06-04182f8b4e4a&authuser=0', 'model.h5')
    # print("Downloaded")
    # return load_model('model.h5')
    id = "1eVTq8I0uUgxgTqB5e6LA13wF99manJ6j"
    output = "model.h5"
    gdown.download(id=id, output=output, quiet=False)
    return load_model("model.h5")

model = load_h5_model()
        
def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        #conver image to cv2
        print(img.shape)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)

def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG-15 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(
            img,
            dsize=img_size,
            interpolation=cv2.INTER_CUBIC
        )
        set_new.append(preprocess_input(img))
    return np.array(set_new)

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return image_data


def main():
    st.header('CSE3505 J Component Final Review')
    st.title('MRI Brain Tumor Detection')
    st.subheader('Presented by')
    st.markdown('''
    - Harsh Deshwal 20BPS1145
    - Siddharth Suresh 20BPS1042
    - Kanishka Ghosh 20BPS1125
    ''')
    st.subheader('Upload the scan to test')
    image_data = load_image()
    if image_data is not None:
        # get image
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)
        # make the size 224x224
        # crop the image
        img = cv2.resize(img, (224, 224),interpolation=cv2.INTER_CUBIC)
        print(img.shape)
        img = crop_imgs(np.array([img]), add_pixels_value=0)[0]
        print(img.shape)
        img = cv2.resize(img, (224, 224),interpolation=cv2.INTER_CUBIC)
        print(img.shape)
        #preprocess
        img = preprocess_input(img)
        #predict
        prediction = model.predict(np.array([img]))
        #get the class
        class_ = np.argmax(prediction)
        #get the probability
        prob = prediction[0][class_]
        #display the result
        prob = round(prob*100, 2)
        if prob > 0.5:
            st.markdown(f'''
            ##### The model predicts that the image <span style="color:lightred">**has a tumor**</span> with a probability of **{prob}**
            ''',unsafe_allow_html=True)
        else:
            st.markdown(f'''
            ##### The model predicts that the image <span style="color:lightgreen">**does not have a tumor**</span> with a probability of **{100-prob}**
            ''',unsafe_allow_html=True)

if __name__ == '__main__':
    main()
