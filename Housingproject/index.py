import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from PIL import Image
import Graphs
from sklearn.preprocessing import StandardScaler, minmax_scale
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

st.set_page_config(
    page_icon="1.jpg",
    page_title="House Price Predictor | App"
)
path1='CleanHousing.csv'
path2='USAHousing.csv'
with open("Style.css") as f:
    st.markdown(f"<style>{f.read()} </style>", unsafe_allow_html=True)

data1 = pd.read_csv(path1)
data1.drop(columns='Unnamed: 0', inplace=True)

with st.sidebar:
    image_path = "2.png"
    image = Image.open(image_path)
    st.sidebar.image(image, use_column_width=True)
    selections = option_menu(
        menu_title=None,
        options=['Home', 'Display DataSet', "Data Visualization", 'Inserting Data', 'Display Prediction', 'How Model Predicts'],
        icons=['house-fill', 'bi-display-fill', "bi-database-add", 'bi-database-fill-up', 'bi-trash-fill', 'bi-search'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#091118fa", "border-radius": "0", "font-color": "white"},
            "icon": {"color": "white", "font-size": "23px"},
            "hr": {"color": "rgb(255, 255, 255)"},
            "nav-link": {"color": "white", "font-size": "15px", "text-align": "left", "margin": "5px", "--hover-color": "blue", "border": "None"},
            "nav-link-selected": {"background-color": "#81B622"}
        }
    )

if selections == "Home":
    st.title("Welcome to The House Price Prediction App")
    st.markdown("""
        This application leverages the power of linear regression to help you estimate the price of a house based on various features. Whether you're a real estate professional, a prospective homebuyer, or just someone interested in the housing market, this tool can provide valuable insights.
        ### Key Features:
        - **User-Friendly Interface**: Easily input features such as square footage, number of bedrooms, and location.
        - **Accurate Predictions**: Our model is trained on a robust dataset to ensure reliable price estimations.
        - **Real-Time Results**: Get instant predictions as you adjust the input parameters.
        ### How It Works:
        1. **Input Features**: Enter the relevant details about the house you are evaluating.
        2. **Predict Price**: Click the 'Predict' button to see the estimated price based on our linear regression model.
        Explore the app, experiment with different inputs, and gain a better understanding of how various factors influence house prices. Thank you for using our House Price Prediction App!
    """)

elif selections == "Display DataSet":
    data = pd.read_csv(path2)
    data.drop(columns='Address', inplace=True)
    st.title("House Dataset")
    st.write("Total Number of Rows and Columns: ", data.shape)
    st.table(data.head(15))
    data1 = pd.read_csv(path1)
    data1.drop(columns='Unnamed: 0', inplace=True)
    st.title("After Scaling The Dataset")
    st.subheader("To Better Understand The Model")
    st.write("Total Number of Rows and Columns: ", data1.shape)
    st.table(data1.head(15))

elif selections == "Data Visualization":
    option = st.selectbox("Select Any one Option", ["Outlier Detection", "Relationship between Features and Target"])
    if option == "Outlier Detection":
        Graphs.outlier_Detection()
    if option == "Relationship between Features and Target":
        Graphs.Relationship()

elif selections == "Inserting Data":
    st.title("House Price Prediction")
    income = st.number_input("Enter Avg. Area Income")
    age = st.number_input("Enter Avg. Area House Age")
    rooms = st.number_input("Enter Avg. Area Number of Rooms")
    bedrooms = st.number_input("Enter Avg. Area Number of Bedrooms")
    population = st.number_input("Enter Area Population")
    button = st.button("Predict House Price")

    if button:
        data = {
            'Avg. Area Income': [income],
            'Avg. Area House Age': [age],
            'Avg. Area Number of Rooms': [rooms],
            'Avg. Area Number of Bedrooms': [bedrooms],
            'Area Population': [population]
        }
        df = pd.DataFrame(data)
        st.write(df)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        with open("LinearRegressionModel", "rb") as file:
            Linear_Regression_Model1 = pickle.load(file)
        pred = Linear_Regression_Model1.predict(scaled_data)
        st.title(f"Predicted House Price: ${pred[0]:,.2f}")

elif selections == "Display Prediction":
    Graphs.predication_by_Model()

elif selections == "How Model Predicts":
    st.title("How the Model Predicts House Prices")
    st.markdown("""
        1. **Data Collection**: The dataset is collected containing various features that are believed to influence house prices.
        2. **Data Preprocessing**: The dataset is cleaned and scaled to ensure all features contribute equally to the prediction.
        3. **Model Training**: A linear regression model is trained on the preprocessed dataset.
        4. **Prediction**: The trained model is used to predict house prices based on the input features.
        ### Key Steps:
        - **Feature Scaling**: Ensuring all features are on a similar scale.
        - **Model Training**: Using linear regression to find the best fit line.
        - **Price Prediction**: Applying the model to input features to estimate the house price.
    """,unsafe_allow_html=True)
    Graphs.predication_by_Model()
