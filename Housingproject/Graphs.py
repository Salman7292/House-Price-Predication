import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pickle

data = pd.read_csv("ScaledusHousing.csv")
data.drop(columns='Unnamed: 0', inplace=True)
Scaled_data1 = pd.read_csv("CleanHousing.csv")
Scaled_data1.drop(columns='Unnamed: 0', inplace=True)

def outlier_Detection():
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(5, 2, figsize=(10, 16))
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
    ax8.set_title("Removing the Outlier Before Scaling in Price (Target Feature)")
    plt.tight_layout()
    st.pyplot(fig)

def Relationship():
    st.title("Regression Plots for Correlation Analysis")
    X = Scaled_data1.drop(columns='Price')
    Correlation = X.corr()
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
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
    sns.heatmap(Correlation, cmap="plasma", ax=ax6)
    ax6.set_title("Correlation", fontsize=15)
    plt.tight_layout()
    st.pyplot(fig)

def predication_by_Model():
    X = Scaled_data1.drop(columns="Price")
    Y = Scaled_data1['Price']
    Y = pd.DataFrame(Y)
    Y.rename(columns={"Price": "Actual Price"}, inplace=True)
    with open("LinearRegressionModel", "rb") as file:
        Linear_Regression_Model1 = pickle.load(file)
    prediction1 = Linear_Regression_Model1.predict(X)
    predicted_price = pd.DataFrame(prediction1)
    predicted_price.rename(columns={0: "Predicted Price"}, inplace=True)
    combined_Result = pd.concat([Y, predicted_price], axis=1)
    st.table(combined_Result.head(20))
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    sns.regplot(x=combined_Result['Actual Price'], y=combined_Result['Predicted Price'], scatter=True, ax=ax1, color="green")
    ax1.set_title("Prediction Vs Actual Price", fontsize=15)
    sns.regplot(x=combined_Result['Actual Price'], y=combined_Result['Predicted Price'], scatter=False, ax=ax2, color="red")
    ax2.set_title("Best Fit Line", fontsize=15)
    plt.tight_layout()
    st.pyplot(fig)
