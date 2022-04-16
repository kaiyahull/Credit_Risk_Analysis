# Credit Risk Analysis

## Overview
The purpose of this analysis is to use supervised machine learning to determine the credit card risk the lender takes on with their borrowers. Credit card risk is unbalanced because the low-risk loans far outnumber the number of high-risk loans. This dataset from LendingClub has 68,470 `low_risk` and 347 `high_risk` records. To account for the imbalance in types of loans and provide meaningful insight, this analysis uses the `imbalanced-learn` and `scikit-learn` libraries and the `RandomOverSampler`, `SMOTE`, and `ClusterCentroids` algorithms to either oversample or under sample the data. The analysis also uses the combinational `SMOTEENN` algorithm to over and under sample the data. Finally, we use who machine learning models `BalancedRandomForestClassifier` and `EasyEnsembleClassifier` to reduce bias. This analysis compares the performance of these machine learning methods to predict credit risk and provides a recommendation for the best model. 

## Results
<ins>Oversampling</ins>

RandomOverSampler
* Accuracy: 0.65
* Precision:
* Recall:

<img src="images/randomoversampler_accuracy.png">
<img src="images/randomoversampler_classification.png">

SMOTE
* Accuracy: 0.66
* Precision:
* Recall:

<img src="images/SMOTE_accuracy.png">
<img src="images/SMOTE_classification.png">

<ins>Undersampling</in>
ClusterCentroids
* Accuracy: 0.54
* Precision:
* Recall:

<img src="images/cc_accuracy.png">
<img src="images/cc_classification.png">

<ins>Combination</ins>
SMOTEENN
* Accuracy: 0.65
* Precision:
* Recall: 

<img src="images/SMOTEENN_accuracy.png">
<img src="images/SMOTEENN_classification.png">

<ins>Learning Models</ins>
BalancedRandomForestClassifier
* Accuracy: 0.79
* Precision: 
* Recall:

<img src="images/forest_accuracy.png">
<img src="images/forest_classification.png">


EasyEnsembleClassifier
* Accuracy: 0.93
* Precision:
* Recall:


<img src="images/adaboost_accuracy.png">
<img src="images/adaboost_classification.png">




## Summary
