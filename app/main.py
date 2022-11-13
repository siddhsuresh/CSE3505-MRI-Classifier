import streamlit as st
import numpy as np
import cv2
import imutils
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input

model = load_model('https://github.com/siddhsuresh/CSE3505-MRI-Classifier/blob/main/app/model.h5')

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
    - Harsh Deshwal 20BPS11xx
    - Siddharth Suresh 20BPS1042
    - Kanishka Ghosh 20BPS11xx
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
