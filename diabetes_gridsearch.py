'''
author: Sam Hersick
assignment: CS484 pa3
date: 4/11/25
purpose: python script for creating pipeline and gridsearch for f1 scoring metric 
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
import argparse
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from prep_split_data import load_prep_split
import time

#Create a RandomForest gridsearch using given scoring_metric
def create_gridsearchcv(scoring_metric):
    #Create a Pipeline
    pipe = Pipeline([('rf', RandomForestClassifier(random_state=1))])

    #Create a parameter grid (same for both accuracy & f1 grid searches)
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
        scoring=scoring_metric,
        cv=2,
        verbose=1,
        error_score='raise'
    )

    #Return the grid search object
    return gs

#Print classification report for training data and testing data
def display_classification_reports(gs, X_train, X_test, y_train, y_test):
    print("\nTRAINING REPORT:")
    print(classification_report(y_train, gs.predict(X_train)))

    print("\nTESTING REPORT:")
    print(classification_report(y_test, gs.predict(X_test)))

#main method
def perform_gs(scoring_metric, output_path):
    #Load the traing & testing data
    X_train, X_test, y_train, y_test = load_prep_split()

    #Get the grid search that we will be using
    gs = create_gridsearchcv(scoring_metric)

    start_time = time.time() #start a timer to display how long the grid search runs for

    #Run the gridsearch, print best score and best model (looking at hyperparameter combination)
    print(f"\n\nPerforming grid search for -- {scoring_metric}...\n")
    gs.fit(X_train, y_train)
    print(f"grid search finished. results...\n\n{gs.best_score_}\n{gs.best_estimator_}")
    
    #Print the run time for the grid search
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    #Print classification reports for training & testing data
    display_classification_reports(gs, X_train, X_test, y_train, y_test)

    #Save grid search results to csv file
    pd.DataFrame(gs.cv_results_).to_csv(output, index=False)

    '''IMPORTANT NOTE:
    - I used VIM search/replace to go in and manually add "None" to all of the blank spaces in the csv because 
    Pandas interprets the None value for class_weight, max_depth, and max_features as a missing value
    
    1. Highlight text
    2. :s/,,/,None,/g --> run twice because of three missing cells in a row'''
    
    print(f"\n\nSAVED AND FINISHED!\n\tGrid Search results savd to {output}")


#main method
if __name__ == "__main__":
    #Take in scoring metric and output file as CLI arguements
    parser = argparse.ArgumentParser(description="Run GridSearchCV for RandomForestClassifier")

    parser.add_argument("--scoring", type=str, choices=["f1", "accuracy", "balanced_accuracy"], required=True, help="Scoring metric for grid search (required)")
    parser.add_argument("--output", type=str, help="Output CSV file for grid search results")

    args = parser.parse_args()

    #save arguments for CLI
    scoring = args.scoring
    output = args.output or f"rf_grid_{scoring}.csv" #output to requested file or default generated file path

    perform_gs(scoring, output)