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

- **Test Classes:** Class 1 data: **54583** observations, Class 0 data: **8837** observations
- **Change in interpretation:** After seeing the Testing classification reports my interpretations do not change at all. We see from both of the testing classification reports that the data is extremely similar to the training reports with no real noticable change.
- **Why no change?** I believe the results are the way that they are because of the nature of random forests and grid searches. When doing grid searches we use cross validation to improve the robustness of our results. With random forests we also use bootstrapping and oob_score to constantly test our data and never before seen data. Random forests naturally are very good at not overfitting because they are made up of many different decision trees, and use concepts from ensemble learning. This allows them be very good at generalizing to new data (as we can see in our results). All of these factors, as well as the other hyper parameters we used (such as max_depth, and max_features which prevent overfitting) lead to a robust model that really shouldn't perform worse on the testing data.

## Interpretation after Analysis

Your Jupyter notebook should have markdown with analysis after each output or graph created to analyze the cv results. Below answer the big picture questions related to your analysis.

### Did the effect of hyperparameter values match between metrics, or differ? In what way?

The effect of hyperparameter values between metric greatly differed. Some hyperparameter's had essentially no effect on the results at all. These include n_estimators and oob_score, where we saw no difference in test_score across the different values that we tested. Next there was max_features which had an extremely small impact on the test_score. Max features actually didn't affect the test score for accuracy grid search, however, with (and only with the None class weigth) we saw a variation in test score during the f1 grid search. This showed that max_features=None actually outperformed max_features=sqrt whenever the class weight was None. Lastly there were two hyperparameters with huge impacts in the data: class_weight and max_depth. Max_depth (the hyper parameter I chose) only slightly affected the accuracy grid search (only with class weight balanced). On the f1 grid search however setting the max_features to 5 greatly reduced the no class weight test scores, and **GREATLY** improved the balanced weight test scores. Class weight had a small impact on the accuracy grid search (decreasing the scores), but on the f1 grid search it had a massive impact on not only the test scores itself, but also on all of the other hyperparameters as well and how they interacted with the test scores. Overall I think it is fair to say that the class_weights had the greatest effect.

### What about the analysis graphs surprised you? If nothing surprised you, what about the graphs were as expected?

The large difference between accuracy and f1 distributions was surprising. I expected them two graphs to be different but not this drastically. The f1 graphs were lower overall and had much more spread than the accuracy graphs. It was clear the f1 metric took more concepts into account because it was shown to be much more sensitive to the hyperparameters than accuracy. I was also shocked to see how class weight not only impacted the score, but it also impacted the other hyperparameters effects of the score. Very interesting!

### Did either or both searches result in models with overfitting? How can you tell?

No they did not really ever show signs of overfitting. The testing and training f1's and accuracys were almost exactly identical with no drop in performance from the training to testing data. With overfitting we would be able to see higher and higher scores on the training data while decreasing scores in the testing data and this was just never really the case.

### Did the best model perform as well on the test data as you expected? How so?

When using accuracy, the best model performed slightly better than expected in terms of test accuracy around 86%. I thought that even though it would be close, we would see a 5-10% drop off in accuracy on the testing data. This never happened so we know the model generalized very well. Even the f1 model was able to reach a 72% accuracy on the test data while only dropping 4% from the training data. This shows how even while fighting really hard to handle class imabalance, random forests can still be extremely robust models.

### This data was imbalanced. What did we do because of that in our process? Was it effective?

To address the imbalance, we included `class_weight` as a hyperparameter in our grid search and tested both `"None"` and `"balanced"` settings. We also performed a grid search that scored on f1 which will help us select hyper parameters that can better combat class imablance. Using class_weight balanced ending up working very well in the f1 grid search and far out performing class weight none with an ~0.18 improvement in mean test score. 

### What else could we try later to deal with the imbalance?

- Since we already used grid search scoring of f1, we could expirement with grid search scoring of balanced accuracy to see if that can help us handle class imbalance (very easy to do with my code).
- We could also perform methods that effect the data directly such as downsmapling or upsampling. These could be an effective way to handle class imbalance in a more hand's on way that would definitely need to be explored if we were trying to optimize the model for professional use.