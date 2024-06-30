import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from PIL import Image
import Graphs
from sklearn.preprocessing import StandardScaler,minmax_scale
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import requests



st.set_page_config(
        page_icon="1.jpg",
        page_title="House Price Predicter | app"

    )


# StyleURL='https://raw.githubusercontent.com/Salman7292/House-Price-Predication/main/Housingproject/Style.css'
# with open(StyleURL) as f:
#     st.markdown(f"<style>{f.read()} </style>", unsafe_allow_html=True)

StyleURL = 'https://raw.githubusercontent.com/Salman7292/House-Price-Predication/main/Housingproject/Style.css'
style_response = requests.get(StyleURL)
st.markdown(f"<style>{style_response.text}</style>", unsafe_allow_html=True)

CleanHousingURL='https://raw.githubusercontent.com/Salman7292/House-Price-Predication/main/Housingproject/CleanHousing.csv'
USAHousingURL='https://raw.githubusercontent.com/Salman7292/House-Price-Predication/main/Housingproject/USAHousing.csv'
data1 =pd.read_csv(CleanHousingURL)
data1.drop(columns='Unnamed: 0',inplace=True)
with st.sidebar:
    image_url = "https://github.com/Salman7292/House-Price-Predication/blob/main/Housingproject/2.png?raw=true"
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))


    # Display logo in sidebar
    st.sidebar.image(image, use_column_width=True)

    # Define the navigation menu using option_menu
    selections = option_menu(
        menu_title=None,
        options=['Home', 'Display DataSet',"Data Visulization",'Inserting Data', 'Display Predication', 'How Model Perdict'],
        icons=['house-fill', 'bi-display-fill', "bi-bar-chart-line-fill",'bi-database-fill-up', 'bi-easel-fill', 'bi-gear'],
        menu_icon="cast",  # Optional: Change the menu icon
        default_index=0 ,  # Optional: Set the default selected option
        styles={
        "container": {"padding": "5!important","background-color":"#091118fa", "border-radius": ".0","font-color":"white"},
        "icon": {"color": "white", "font-size": "23px"}, 
         "hr": {
         "color": "rgb(255, 255, 255)"
          },
       "nav-link": {"color":"white","font-size": "15px", "text-align": "left", "margin":"5px", "--hover-color": "blue","border":"None"},
        "nav-link-selected": {"background-color": "#81B622"},


# .menu .container-xxl[data-v-5af006b8] {
#     background-color: var(--secondary-background-color);
#     /* border-radius:.5rem; */
# }
}

    )



if selections == "Home":
    st.title("Wellcome to The House Price perdication app")
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
   


    data =pd.read_csv(USAHousingURL)
    data.drop(columns='Address',inplace=True)
    st.title("House Dataset")
    st.write("Total Numnber Rows and Columns : ",data.shape)
    st.table(data.head(15))

    # after Scaling the Data set 
    data1 =pd.read_csv(CleanHousingURL)
    data1.drop(columns='Unnamed: 0',inplace=True)
    st.title("After Scaling The DataSet")
    st.subheader("To Batter Understand The Model")
    st.write("Total Numnber Rows and Columns : ",data1.shape)
    st.table(data1.head(15))


elif selections == "Data Visulization":
    st.title("Data Visulization")
    st.header("Outliers Detection and Removel")
    Graphs.outlier_Detection()
    Graphs.Relationship()

elif selections == 'Inserting Data':

    st.title("Input To The Model")

    submationform = st.form("Submission for input to Model")

    # Fix the slider by making step a float
    avg_area_income = submationform.slider(
        label="Avg. Area Income",
        min_value=1796.631190,
        max_value=107701.748400,
        step=2000.0  # Make step a float
    )

    Avg_Area_House_Age = submationform.slider(
        label="Avg. Area House Age",
        min_value=2.644304,
        max_value=9.519088	,
        step=1.0 # Make step a float
    )

    Avg_Area_Number_of_Rooms = submationform.slider(
        label="Avg. Area Number of Rooms",
        min_value=3.236194,
        max_value=10.759588,
        step=1.0 # Make step a float
    )

    Avg_Area_Number_of_Bedrooms = submationform.slider(
        label="Avg. Area Number of Bedrooms",
        min_value=2.000000,
        max_value=6.500000,
        step=0.1# Make step a float
    )

    Area_Population	 = submationform.slider(
        label="Area Population",
        min_value=172.610686,
        max_value=69621.713380,
        step=100.00# Make step a float
    )

    # Add a submit button
    submit_button = submationform.form_submit_button("Predict")

    # Handle form submission
    if submit_button:
        # st.write(f"Avg. Area Income selected            : {avg_area_income}")
        # st.write(f" Avg_Area_House_Age selected         : { Avg_Area_House_Age}")
        # st.write(f"Avg_Area_Number_of_Rooms selected    : {Avg_Area_Number_of_Rooms}")
        # st.write(f"Avg_Area_Number_of_Bedrooms selected : {Avg_Area_Number_of_Bedrooms}")
        # st.write(f"Area_Population selected             : {Area_Population}")
        dic = {
            "Avg. Area Income": [avg_area_income],
            "Avg. Area House Age": [Avg_Area_House_Age],
            "Avg. Area Number of Rooms": [Avg_Area_Number_of_Rooms],
            "Avg. Area Number of Bedrooms": [Avg_Area_Number_of_Bedrooms],
            "Area Population": [Area_Population]
        }
        d1 = pd.DataFrame(dic)
        st.table(d1)

        # z_ScoreScaling1=StandardScaler()
        # Transoform_Result=z_ScoreScaling1.fit_transform(data1.drop(columns="Price"))
        # # st.table(Transoform_Result)      
        # scaled_all_input_feature =pd.DataFrame(Transoform_Result,columns=d1.columns)
        # # st.table(scaled_all_input_feature.head())
        # scaled_slider_input=z_ScoreScaling1.fit_transform(d1)
        # scaled_all_slider_input_feature =pd.DataFrame(scaled_slider_input,columns=d1.columns)
        # # st.table(scaled_all_slider_input_feature)
        



    

        with open("LinearRegressionModel","rb") as files:
            Linear_Regression_Model=pickle.load(files)
        
        predication=Linear_Regression_Model.predict(d1)
        st.subheader(f"House Price Predication : {np.round((predication[0][0])*286.47).astype(int)} PKR ")
        st.balloons()

elif selections == 'Display Predication':
    st.header("Display all The Predication")
    X=data1.drop(columns="Price")
    Y=data1['Price']
    Y=pd.DataFrame(Y)
    Y.rename(columns={"Price":"Actual Price"},inplace=True)
    with open("LinearRegressionModel","rb") as file:
            Linear_Regression_Model1=pickle.load(file)
        
    predication1=Linear_Regression_Model1.predict(X)
    # st.table(predication1)
   
    predicated_price=pd.DataFrame(predication1)
    predicated_price.rename(columns={0:"Predicted Price"},inplace=True)
    
    combind_Result=pd.concat([Y,predicated_price],axis=1)
    st.table(combind_Result.head(20))

elif  selections == 'How Model Perdict':
    st.title("How Model Predicted The Data")
    st.markdown("""


                Actully this Model Used The Linear Regression Algurithm Which Basically Draw the Best fit Line Through out The Data and 
                on the basis of this Best Fit line Model can Predicted the data
                And Remember that this Best line actully finding the Line Equation which are Given below 
                Y = mx + c


                """,unsafe_allow_html=True)
    Graphs.predication_by_Model()
    
    




    

    















    

