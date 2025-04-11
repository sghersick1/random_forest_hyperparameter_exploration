'''
author: Sam Hersick
assignment: CS484 pa3
date: 4/11/25
purpose: python script to preprocess the data and split it using train_test_split
notes: 
    1. you should output the names of the features and check if they are already numeric
    2. Drop the one feature that obviously needs dropped
    3. Perform any other preprocessing then split into training/testing data
'''

#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#load in the diabetes data, preprocess, return the split data
def load_prep_split():
    #read in the data set into a dataframe
    df = pd.read_csv("diabetes_binary.csv", encoding="utf-8")

    #print the features and drop the 'ID' column
    print(f"The features are:\n{", ".join(df.columns)}")
    df.drop(columns=['ID'], inplace=True)

    #check if the data is numeric
    print("\n", df.dtypes)

    X = df.drop(columns=['Diabetes_binary'])
    y = df['Diabetes_binary']
    return train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

#main method
if __name__ == "__main__":
    load_prep_split()