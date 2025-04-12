'''
author: Sam Hersick
assignment: CS484 pa3
date: 4/11/25
purpose: python script for creating pipeline and gridsearch for accuracy scoring metric 
notes: 
    1. Create a pipeline
    2. Create a parameter grid
        a. class_weight
        b. n_estimators
        c. oob_score
        d. max_features
        e. max_depth
    3. Create/run two GridSearchCV. They use the same paramater grid. One scores with 'accuracy', other uses 'f1' 
    4. Save the grid search results into dataframes, then into csv files. use cv_results_ (gridsearch variable).
    5. Output a classification report on the training data for each model
    6. Output a classification report on the testing data for each model
'''

#imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from prep_split_data import load_prep_split
import time

#main method
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_prep_split()

    #Create a Pipeline
    pipe = Pipeline([('rf', RandomForestClassifier(random_state=1))])

    #Create a parameter grid
    p_grid = {
        'rf__class_weight':[None, "balanced"], #use class weight to handle imbalance or not
        'rf__n_estimators': [50, 100, 200], #number of trees
        'rf__oob_score': [True, False], #whether to use out-of-bag samples
        'rf__max_features': [None, "sqrt"], #max number of features to consider when looking for the best split
        'rf__max_depth':[None, 5] #max depth of the tree
    }

    #create the parameter grid
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=p_grid,
        scoring='accuracy',
        cv=2,
        verbose=1,
        error_score='raise'
    )

    start_time = time.time()

    #run the gridsearch, print best score and best model
    gs.fit(X_train, y_train)
    print(gs.best_score_)
    print(gs.best_estimator_)
    
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds") #print the run time for the grid search

    #print classification report for training data and testing data
    print("\nTRAINING REPORT:")
    print(classification_report(y_train, gs.predict(X_train)))

    print("\nTESTING REPORT:")
    print(classification_report(y_test, gs.predict(X_test)))

    #save grid search results to csv file
    pd.DataFrame(gs.cv_results_).to_csv("rf_grid_accuracy.csv", index=False)

    print("\n\nSAVED AND FINISHED!")