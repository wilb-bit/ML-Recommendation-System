# Recommendation system of beers and reviewers. 
Data taken from:
https://www.kaggle.com/rdoume/beerreviews

Data split into test, train and val datasets. Given that this is a univeristy assignment the test dataset, as shown has no label (review), is used as a holdout set which we predict on and then are graded on.

Train set is used to train our data and the val dataset we predict on and validate how well the algorithm did. 

Due to file size train set has been split into 3 files (train1, train2, train3). When runnning please recombine into one file (train.tsv)



&nbsp;
&nbsp;
## For this project I explore the best collaborative based algorithms and their hyperparamters for this dataset.

A3.py has sweep parameter (default set to false) (will need to open and edit file if you wish to change it)

If sweep is set to true will perform a gridsearch of all algorithms and print best hyperparameters to console.
If set to false will predict on the test set for all algorithms.



The following algorithms (suprise library) are run in the following order:

KNNwithMeans (output file: A3-2.tsv)  
SVD (output file: A3-3.tsv)  
SVDpp (output file: A3-1.tsv)  
SlopeOne  
CoClustering  
Hybrid (Ensemble of SVDpp and KNNwithMeans) (output file: A3-4.tsv)


BEFORE RUNNING:

Install requirements.txt with pip, then use conda to install scikit-surprise (with below command):

conda install -c conda-forge scikit-surprise
