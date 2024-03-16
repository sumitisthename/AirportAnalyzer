# streamlit_background.py
import streamlit as st

def st_set_background(image_url):
    component = """
        <style>
        .stApp {
            background-image: url("%s");
            background-size: cover;
        }
        </style>
    """ % image_url
    st.write(component, unsafe_allow_html=True)

if __name__ == "__main__":
    image_url = 'https://i.postimg.cc/0NjsRkSh/Airport.jpg'
    st_set_background(image_url)


