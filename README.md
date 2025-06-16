**Stratified Regression in Target Encoding**

**Overview**

This repository contains experimental code for evaluating the effectiveness of applying stratified cross-validation in target encoding while using categorical feature encoding techniques namely, **Target Encoding** and **GLMM Encoding** within a supervised learning framework using **scikit-learn**-compatible models.

The files in the repository fall into two main categories:

1. **TargetEncoder Experiments** (TargetEncoder.py):

These scripts investigate the application of the TargetEncoder in scikit-learn, with a particular focus on understanding the impact of incorporating stratified k-fold cross-validation during the encoding process. The aim is to assess whether stratification enhances the generalization performance of downstream machine learning models when TargetEncoder in scikit-learn is used for high-cardinality categorical variables.

2. **GLMMEncoder Experiments**:


