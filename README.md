# Credit Risk Analysis Challenge

## Overview
In working with a lead data scientist at Fast Lending, Data Analyst Inc (DAI) modeled several different machine learning tools to predict credit risk.
Using a credit card dataset from Lending Club, a peer-to-peer lending services company, DAI completed the following:

* oversampled the data using RandomOversampler and SMOTE algorithms,
* undersampled the data using the ClusterCentroid algorithm,
* used a combinatorial approach (over- and under-sampling) with SMOTEENN, and 
* used two machine learning ensembler models which reduce bias - BalancedRandomForestClassifier and EasyEnsembleClassifier.


## Results

**Random Oversampling**

![Random Sampler](https://user-images.githubusercontent.com/35401581/145309702-7b2a3af3-0e84-4476-a865-e6826c0dba8b.png)

The random oversampling produced a balanced accuracy score of 66% which is decent.  However, the high risk precision score is only 1%.  Because there is an imbalance of a large number of low risk loans, the low risk precision score is 100% with a sensitivity score of 60%.

**SMOTE**

![SMOTE](https://user-images.githubusercontent.com/35401581/145309818-dbc7ada4-cc41-48c3-ae8a-570ef4dbecd8.png)

New high risk loan examples were synthesized in SMOTE to even out the imbalance between samples of the low and high risk loan portfolio.  In this model, performance was very close to the same as the random oversampling model.  This model also has an accuracy score of 66%, only a 1% high risk precision score, and a 100% low risk prescision.  The average F1 score is improved to 82% as compared to random oversampling at 75%.  The sensitivity rate is 61% overall.

**Cluster Centroids**

![Cluster Centroids](https://user-images.githubusercontent.com/35401581/145309830-1525d320-5301-4e68-8286-e982379bc910.png)

In this model, DAI undersampled a cluster of the majority class of low risk loans in hopes of improving regression accuracy.  Unfortunately, this model underperformed the prior oversampling models with a cluster centroid accuracy score of 54% and an average F1 score of 56%.  Due to the high number of false positives, the low risk loan's sensitivity rate is low at 40% and a 40% rate overall.  The precision ratios are very close to those of the prior two models with a very low 1% high risk loan precision rate.

**SMOTEENN**

![SMOTEENN](https://user-images.githubusercontent.com/35401581/145309882-994db0c2-8944-4ee9-aecf-0e07ae63b015.png)

In this model DAI combined undersampling and oversampling combined with the EditedNearestNeighbor method (SMOTEENN).  Again, the results were very close to the previous models most closely following Random Oversampling.  The accuracy score is 68%.  The high risk precision rate is 1% and the low risk is 100% with a low risk sensitivity rate of 57% and an overall F1 factor of 72%.  

**Balanced Random Forest Classifier**

![Balanced Random Forest Classifier](https://user-images.githubusercontent.com/35401581/145722381-57b66e09-d2eb-407b-9fa4-a708539fafd0.png)

A balanced random forest randomly under-samples each boostrap sample to balance it. ... If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.  This method appears to outperform the prior methods in some ways. The accuracy score is 79%.  The average F1 score is 93%.  The high risk loans have a 87% recall rate and the average recall rate is 70%.  However, The precision score for the high-risk loans is still low at 3% and the F1 score for this category is also low at 6%.  

**Easy Enemble Classifier**

![Easy Ensemble Classifier](https://user-images.githubusercontent.com/35401581/145309901-2b4c4b46-18d9-468a-bf7c-67ed4007e69d.png)

This algorithm is known as EasyEnsembleClassifier. The classifier is an ensemble of AdaBoost learners trained on different balanced boostrap samples. The balancing is achieved by random under-sampling.  This model outperforms all other models based on an accuracy rate of 93%.  However, the precision of high risk loans is still low at 9% with a sensitivity or recall ratio of 92% but with an F1 factor of 16%.

## Summary
In reviewing all six models, the ensembler methods - the Easy Ensemble Classifer and Balanced Random Forest Classifier models - yielded the best results, respectively. 
* The best candidate is Easy Ensembler Classifier which has the best overall accuracy rate of 93%.  The overall sensitivity (recall) rate was the highest at 94% as was the overall F1 score at 97%.  However, when predicting high risk candidates the the precision rate was still low at 9%.  Perhaps there is room for improvement.
* The Balanced Random Forest Classifier has a decent but not as solid accuracy rate of 78%.  However, the other ratios were not as good as the Easy Ensemble Classifier.  

   

