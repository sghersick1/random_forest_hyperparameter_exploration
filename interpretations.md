# PA3 Interpretations

## Interpretations after Training

Answer the following questions after completing your training steps, before analyzing the hyperparameters in a Jupyter notebook.

### How well did each grid search appear to work based on the *training* data's classification report?

#### Overview
Overall, based on the training data's classification report the test results appear to be strong when scored on accuracy, and moderately well when scored on f1.

##### Looking at Classification Report & Class imbalance
Something super important to consider however is how well it is performing for each class.
- **Classes:** Class 1 data: **163751** observations, Class 0 data: **26509** observations

These following results can be found in accuracy_output.txt and f1_output.txt files.
- **Accuracy:** We saw good results for class 0 (precision: 0.87, recall: 0.98, f1: 0.93) but extremely poor performance on class 1 (precision: 0.56, recall: 0.13, f1: 0.22) espescially when it comes to recall and false negatives.
- **f1:** With f1 we saw a noticeable change in results. Class 0 (precision: 0.95, recall: 0.71, f1: 0.81) still performed well overall but dropped a few points from the accuracy grid search. Class 1 however(precision: 0.30, recall: 0.78, f1: 0.44) we saw a decrease in precision, but an increase in recall and f1. This means that overall we are predicting more positives. However, more of our class 1 predictions are wrong incorrect when we overall are predicting a greater amount of class 1's. This is because we are casting a larger net, overall scoring our gridsearch on f1 is helping the model overcome class imbalance by improving class 1 metrics. But as a whole the model is dropping from a 0.86 (accuracy grid search) accuracy, to a 0.72 (f1 grid search) accuracy.

### How does your interpretation change (or not change) after seeing the test data's classification report? Why do you think the results are what they are?

- **Classes:** Class 1 data: **54583** observations, Class 0 data: **8837** observations
- **Change in interpretation:** After seeing the Testing classification reports my interpretations do not change at all. We see from both of the testing classification reports that the data is extremely similar to the training reports with no real noticable change.
- **Why no change?** I believe the results are the way that they are because of the nature of random forests and grid searches. When doing grid searches we use cross validation to improve the robustness of our results. With random forests we also use bootstrapping and oob_score to constantly test our data and never before seen data. Random forests naturally are very good at not overfitting because they are made up of many different decision trees, and use concepts from ensemble learning. This allows them be very good at generalizing to new data (as we can see in our results). All of these factors, as well as the other hyper parameters we used (such as max_depth, and max_features which prevent overfitting) lead to a robust model that really shouldn't perform worse on the testing data.

## Interpretation after Analysis

Your Jupyter notebook should have markdown with analysis after each output or graph created to analyze the cv results. Below answer the big picture questions related to your analysis.

### Did the effect of hyperparameter values match between metrics, or differ? In what way?


### What about the analysis graphs surprised you? If nothing surprised you, what about the graphs were as expected?


### Did either or both searches result in models with overfitting? How can you tell?


### Did the best model perform as well on the test data as you expected? How so?


### This data was imbalanced. What did we do because of that in our process? Was it effective?


### What else could we try later to deal with the imbalance?

