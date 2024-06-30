import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pickle

CleanHousingURL='https://raw.githubusercontent.com/Salman7292/House-Price-Predication/main/Housingproject/CleanHousing.csv'
ScaledusHousingURL='https://raw.githubusercontent.com/Salman7292/House-Price-Predication/main/Housingproject/ScaledusHousing.csv'
data=pd.read_csv(ScaledusHousingURL)
data.drop(columns='Unnamed: 0',inplace=True)
Scaled_data1=pd.read_csv(CleanHousingURL)
Scaled_data1.drop(columns='Unnamed: 0',inplace=True)



def outlier_Detection():
    # st.header("Unclean Data")
   
    # st.table(data.head())

    # st.header("clean Data")
    
    # st.table(Scaled_data1.head())
   
    # Create a figure with subplots
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(5, 2, figsize=(10, 16))

    # Plotting boxplots
    sns.boxplot(x=data['Avg. Area Income'], data=data, ax=ax1)
    ax1.set_title("Outlier After Scaling in Area Income")

    sns.boxplot(x=Scaled_data1['Avg. Area Income'], data=Scaled_data1, ax=ax2)
    ax2.set_title("Removing the Outlier after Scaling in Area Income")

    sns.boxplot(x=data['Avg. Area House Age'], data=data, ax=ax3, color="red")
    ax3.set_title("Outlier After Scaling in Avg. Area House Age")

    sns.boxplot(x=Scaled_data1['Avg. Area House Age'], data=Scaled_data1, ax=ax4, color="red")
    ax4.set_title("Removing the Outlier after Scaling in Avg. Area House Age")

    sns.boxplot(x=data['Avg. Area Number of Rooms'], data=data, ax=ax5, color="green")
    ax5.set_title("Outlier After Scaling in Avg. Area Number of Rooms")

    sns.boxplot(x=Scaled_data1['Avg. Area Number of Rooms'], data=Scaled_data1, ax=ax6, color="green")
    ax6.set_title("Removing the Outlier after Scaling in Avg. Area Number of Rooms")

    sns.boxplot(x=data['Area Population'], data=data, ax=ax7, color="orange")
    ax7.set_title("Outlier After Scaling in Area Population")

    sns.boxplot(x=Scaled_data1['Area Population'], data=Scaled_data1, ax=ax8, color="orange")
    ax8.set_title("Removing the Outlier after Scaling in Area Population")

    sns.boxplot(x=data['Price'], data=data, ax=ax9, color="gray")
    ax9.set_title("Outlier After Scaling in Price (Target Feature)")

    sns.boxplot(x=Scaled_data1['Price'], data=Scaled_data1, ax=ax10, color="gray")
    ax10.set_title("Removing the Outlier Before Scaling in Price (Target Feature)")

    # Adjust subplots to fit in the figure area
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)



def Relationship():
    # Set the title of the app
    st.title("Regression Plots for Correlation Analysis")
    X=Scaled_data1.drop(columns='Price')

    Correltion=X.corr()
    

    # Create a figure with subplots
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))

    # Plotting regression plots
    sns.regplot(x=Scaled_data1['Avg. Area Income'], y=Scaled_data1['Price'], scatter=True, ax=ax1)
    ax1.set_title("Positive Correlation B/w Avg. Area Income and Price", fontsize=15)

    sns.regplot(x=Scaled_data1['Avg. Area House Age'], y=Scaled_data1['Price'], scatter=True, ax=ax2, color="y")
    ax2.set_title("Positive Correlation B/w Avg. Area House Age and Price", fontsize=15)

    sns.regplot(x=Scaled_data1['Avg. Area Number of Rooms'], y=Scaled_data1['Price'], scatter=True, color='blue', ax=ax3)
    ax3.set_title("Weak Correlation Avg. Area Number of Rooms VS Price", fontsize=15)

    sns.regplot(x=Scaled_data1['Avg. Area Number of Bedrooms'], y=Scaled_data1['Price'], scatter=True, color='blue', ax=ax4)
    ax4.set_title("Negative Correlation B/w Avg. Area Number of Bedrooms VS Price", fontsize=15)

    sns.regplot(x=Scaled_data1['Area Population'], y=Scaled_data1['Price'], scatter=True, color='red', ax=ax5)
    ax5.set_title("Almost Zero Correlation B/w Area Population VS Price", fontsize=15)

    sns.heatmap(Correltion,cmap="plasma", ax=ax6)
    ax6.set_title("Correlation ", fontsize=15)

    # Adjust layout
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)



def predication_by_Model():
    X=Scaled_data1.drop(columns="Price")
    Y=Scaled_data1['Price']
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


    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    sns.regplot(x=combind_Result['Actual Price'], y=combind_Result['Predicted Price'],scatter=True,ax=ax1,color="green")
    ax1.set_title("Predication Vs Actual Price", fontsize=15)

    sns.regplot(x=combind_Result['Actual Price'], y=combind_Result['Predicted Price'],scatter=False,ax=ax2,color="red")
    ax2.set_title("Best Fit line", fontsize=15)
    # Adjust layout
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)
    


    

  





