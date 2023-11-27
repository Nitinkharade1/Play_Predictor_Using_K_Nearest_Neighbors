import numpy as np 
import pandas as pd 
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def MarvellousPlayPredictor(data_path):
    
    # Step1 : Load data
    data = pd.read_csv(r"C:\\Users\\user\\OneDrive\\Desktop\\python\\ML\\PlayPredictor.csv",index_col=0)

    print("Size of Actual dataset",len(data))

    # Step2 : Clean, Prepare and manipulate data
    feature_names = ['Whether','Temperature']

    print("Names of Features",feature_names)

    whether = data.Whether
    Temperature = data.Temperature
    play = data.Play

    #creating labelEncoder
    le = preprocessing.labelEncoder()

    # Converting string labels into numbers.
    whether_encoded = le.fit_transform(whether)
    print(whether_encoded)

    # Converting string labels into numbers.
    temp_encoded = le.fit_transform(Temperature)
    label = le.fit_transform(play)
    print(temp_encoded)

    #combining weather and temp into single listof tuples
    features = list(zip(wether_encoded,temp_encoded)) 

    #Step3 : Train Data
    model = KNeighborsClassifier(n_neighbours=3)

    #Train the model using the training sets
    model.fit(features,label)

    #Step 4: Test Data
    predicted = model.predict([[0,2]]) # 0:Overcast, 2:Mild
    print(predicted)

def main():

    print("play predictor application using k Nearest Knighbour algorithm")

    MarvellousPlayPredictor("PlayPredictor.csv")