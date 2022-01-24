import streamlit as st    #Importing the required librairies
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR 
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

nba = pd.read_csv("https://raw.githubusercontent.com/shreyabhadwal/NBA-Points-Scored-Prediction-/main/nba_2013.csv")
nba.fillna(0,inplace=True)

nba_split = nba.iloc[:,1:29] #Removing name
nba_split = nba_split.drop(["bref_team_id","pos"], axis = 1) #Dropping categorical variables 


training_data, testing_data = train_test_split(nba_split, test_size=0.2, random_state=25) #Spliiting data

x_train = training_data.drop(["pts"],axis =1) #Dropping variable that needs to be predicted
y_train = training_data["pts"] 

x_test = testing_data.drop(["pts"],axis =1)
y_test = testing_data["pts"]

sc = StandardScaler()    #Standardization (Removing the mean and scaling to unit variance)
sc.fit(x_train)
x_train = sc.transform(x_train)
sc.fit(x_test)
x_test = sc.transform(x_test)

x_nba_split = nba_split.drop(["pts"], axis = 1) #Removing the y variable


st.title("Predicting Points Scored by NBA Players in the 2013-14 Season")  #Headings of the app
st.markdown('### _Exploring Different Regression Models_ ')
         
dataset_regressor = st.sidebar.selectbox(
    "Select Regressor", ("KNN", "SVR", "Random Forest"))

st.markdown(f'### {dataset_regressor}')


def add_parameter_ui(regressor_name):   #Creating a sidebar where users can choose which regressor they want
    params = dict()
    
    if regressor_name == "KNN":
        K = st.sidebar.slider("K",1,15)   #Asking user for value of K
        params["K"] = K
    
    elif regressor_name == "SVR":
        Ker = st.sidebar.selectbox('Kernel', ["linear", "poly", "rbf"]) #Asking user which kernel
        params["Ker"] = Ker
        
    else:
        N = st.sidebar.slider("Number of Trees in Forest",10,100) #Asking user number of trees in forest
        max_depth = st.sidebar.slider("Maximum Depth of Tree",2,15)  #Asking user maximum depth of tree
        params["N"] = N
        params["max_depth"] = max_depth
    
    return params  #Return a dictionary with the values given as inputs

params = add_parameter_ui(dataset_regressor)  #Calling the function defined above

def get_regressor(regressor_name,params): #Creating the regression model
    
    if regressor_name == "KNN":
        reg = KNeighborsRegressor(n_neighbors=params["K"]) #Values given to the function are taken from dictionary
       
    
    elif regressor_name == "SVR":
        reg = SVR(kernel=params["Ker"])
        
    else:
        reg = RandomForestRegressor(n_estimators=params["N"], 
                                   max_depth=params["max_depth"], 
                                   random_state=25)
    
    return reg  #Return the model

reg = get_regressor(dataset_regressor, params) #Calling the function defined above

reg.fit(x_train,y_train)   #Fitting the training data to the model
y_pred = reg.predict(x_test) #Making predictions on test data
acc = reg.score(x_test,y_test) #Getting the accuracy 

st.markdown(f'##### Accuracy {acc}')

# Plotting the data

pca = PCA(1)
x_test_reduced = pca.fit_transform(x_test) #Reducing multi dimensional data to single variable using PCA
y_test = np.resize(y_test, (97,1)) #Resizing arrays 
y_pred= np.resize(y_pred,(97,1))


fig = plt.figure()  #Plotting scatter plots
plt.scatter(x_test_reduced, y_test , color = 'red',
            label = "Principal Component vs Actual Values")
plt.scatter(x_test_reduced, y_pred, color = 'blue',
            label = "Principal Component vs Predicted Values")
plt.legend()
plt.title(f"{dataset_regressor} Regression Model")
plt.xlabel('X (Reduced to 1 Principal Component)')
plt.ylabel('Points Scored')


st.pyplot(fig)    
st.caption("The above graph is plotted only for about 100 data points of the dataset.")