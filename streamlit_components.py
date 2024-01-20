import streamlit as st
import numpy as np
import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation


def welcome():
    st.title('WelCome to Image Processing techniques')

    st.subheader('A unique app that shows different image processing algorithms.'
                 + ' You can choose the options from the left side. It has a sidebar with '
                 +'Colour Filter, Blur Face and Detection, Face Count, Vehicle detection & Count and Background Remove')

    st.image('image_logo.jpg', use_column_width=True)
    st.subheader('Made by Anit')

def Color_filter():
    st.header("You Chosen :- ")
    st.subheader("Colour filter")
    uploaded_file = st.file_uploader("Upload a image file ", type=[".jpg", ".jpeg", ".png"])

    type_name = st.sidebar.selectbox("Choose thresholding type", ["Grayscale Filter",
                                                                     "Brightness Adjustment",
                                                                  "Dark Adjustment",
                                                                  "Sketch Filter", "Cartoon"])
    col1, col2 = st.columns(2)

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        # display original image
        col1.subheader("Original Image")
        col1.image(img, channels="BGR")

    # params

    thresh = st.sidebar.slider('For Brightness Adjustment', 0, 100)
    thresh1 = st.sidebar.slider('For Dark Adjustment', -100, 0)
    thresh2 = st.sidebar.slider('For Sketch Adjustment', 0.0, 400.0)

    if (st.sidebar.button('Show Results')):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if type_name == "Grayscale Filter":
            res = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif type_name == "Brightness Adjustment":
            res1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            res = cv2.convertScaleAbs(res1, beta= thresh)
        elif type_name == "Dark Adjustment":
            res1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            res = cv2.convertScaleAbs(res1, beta= thresh1)
        elif type_name == "Sketch Filter":
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            inverted_image = 255 - gray_image
            blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)
            inverted_blurred = 255 - blurred
            res = cv2.divide(gray_image, inverted_blurred, scale=thresh2)

        elif type_name == "Cartoon":
            # Edges
            gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray1, 5)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 9, 9)

            # Cartoonization
            res1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            color = cv2.bilateralFilter(res1, 9, 255, 255)
            res = cv2.bitwise_and(color, color, mask=edges)

        # display result
        col2.subheader(type_name)
        col2.image(res)
        st.header(type_name)
        st.image(res, width=640)


def face_detection():
    st.header("You Chosen :- ")
    st.subheader("Blur Face and Detection")
    uploaded_file = st.file_uploader("Upload a image file ", type=[".jpg", ".jpeg", ".png"])

    col1, col2 = st.columns(2)
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        col1.subheader("Original Image")
        col1.image(img, channels="BGR")

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    if st.sidebar.button('Blur Faces '):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, minNeighbors=3, minSize=(30,30))
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            ans = img[y:y + h, x:x + w, :]
            blur = cv2.GaussianBlur(ans, (91,91), 0)
            img[y:y+h, x:x+w] = blur
            # with st.container():
            #     for col in st.columns(1):
            #         col.image(ans, width=150, channels="BGR")

        col2.subheader("Blur Faces")
        col2.image(img, channels="BGR")
        st.header(" Blur Face Detection:")
        st.image(img, width=640, channels="BGR")

def face_count():
    st.header("You Chosen :- ")
    st.subheader("Face Count")
    uploaded_file = st.file_uploader("Upload a image file ", type=[".jpg", ".jpeg", ".png"])

    col1, col2 = st.columns(2)

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        col1.subheader("Original Image")
        col1.image(img, channels="BGR")

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    if st.sidebar.button('Face Count '):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 7)
        count = 0
        for (x, y, w, h) in faces:
            ans = img[y:y + h, x:x + w, :]
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)  #bgr formate
            count +=1
            with st.container():
                for col in st.columns(1):
                    col.image(ans, width=150, channels="BGR")

        # cv2.putText(img,'face num:'+ str(count), (60, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        col2.subheader("Detected Faces")
        col2.image(img, channels="BGR")
        st.image(img, width=640, channels="BGR")
        st.header("Total face")
        st.subheader(str(count))

def vechiclework():
    st.header("You Chosen :- ")
    st.subheader("Car detection and Count")
    uploaded_file = st.file_uploader("Upload a image file ", type=[".jpg", ".jpeg", ".png"])

    col1, col2 = st.columns(2)

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        col1.subheader("Original Image")
        col1.image(img, channels="BGR")

    face_cascade = cv2.CascadeClassifier('car1.xml')

    if st.sidebar.button('Detect '):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.15, 2)
        count =0
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # bgr formate
            count += 1

        col2.subheader("vehicle detection")
        col2.image(img, channels="BGR")
        st.image(img, width=640, channels="BGR")
        st.header("Total Vehicle")
        st.subheader(str(count))

def background():
    st.header("You Chosen:- ")
    st.subheader("Background Remove")
    uploaded_file = st.file_uploader("Upload a image file ", type=[".jpg", ".jpeg", ".png"])

    col1, col2 = st.columns(2)
    thresh = st.sidebar.slider('For edge Adjustment', 0.0, 1.0)

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        col1.subheader("Original Image")
        col1.image(img, channels="BGR")

    segmentor = SelfiSegmentation()

    if st.sidebar.button('Remove'):
        imgout = segmentor.removeBG(img, (0,0,0), threshold= thresh)
        col2.subheader("Background Changed")
        col2.image(imgout, channels="BGR")
        st.image(imgout, width= 640, channels="BGR")


def main():
    # st.image(os.path.join("image_procssing.jpg"), use_column_width= True)
    # st.header("     ...WelCome to Image Processing Method..    ")
    st.sidebar.title("Image Processing ")
    function_selected = st.sidebar.selectbox("Choose OpenCV function",
                                             [ "WelCome",
                                              "Colour Filter",
                                              "Blur Face",
                                              "Face Count",
                                               "Car detection and Count",
                                               "Background Remove"
                                               ])
    if function_selected == "WelCome":
        welcome()
    elif function_selected == "Colour Filter":
        Color_filter()
    elif function_selected == "Blur Face":
        face_detection()
    elif function_selected == "Face Count":
        face_count()
    elif function_selected == "Car detection and Count":
        vechiclework()
    elif function_selected == "Background Remove":
        background()
    else:
        st.write("Choose right option")


if __name__ == "__main__":
    main()
