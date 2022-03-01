A3.py has sweep parameter (default set to false) (will need to open and edit file if you wish to change it)

If sweep is set to true will perform a gridsearch of all algorithms and print best hyperparameters to console.
If set to false will predict on the test set for all algorithms.


The following algorithms are run in the following order:
KNNwithMeans (A3-2.tsv)					
SVD (A3-3.tsv)
SVDpp (A3-1.tsv)
SlopeOne
CoClustering
Hybrid (Ensemble of SVDpp and KNNwithMeans) (A3-4.tsv)

BEFORE RUNNING:
Install requirements.txt with pip, then use conda to install scikit-surprise (with below command):
conda install -c conda-forge scikit-surprise