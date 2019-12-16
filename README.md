# classification-using-ecg
 Classification of cardiac arrhythmia using ECG data from https://archive.ics.uci.edu/ml/datasets/Arrhythmia 

This work is divided in three phases: 

         1) Data Analysis 
            1.1) Tests of Significance to understad the dependencies of the observed values.
            1.2) Data Imputation methods for handling missing values 
            1.3) Distributions fittings to most informative features
            1.4) Data visualization with PCA and Manifolds techniques
         2) Multiple Classification
            2.1) Performance Analysis of different kind of Learning Algorithms.
                2.1.1) Each Algorithm is tuned with k-cross-validation, you can find the tuning functions in the folder:"Functions/<Algorithm>"
            2.2) We proposed different kind of Feature Selection (Pearson Correlation and Recursive Feature Elimination) you can also use Boruta Algorithm
                 for feature selection ( you can find the implementation inside Function/SVM/svm.py ) 
         3) Two Classifier Classification Architecture:
            3.1) In this section we splitted the multiple classification task in two phases: 
                        1) Binary Classification to predict if a patient is sick or not.
                        2) (In case of a sick patient) Multiple classification to predict the type of arrhythmia.
## Two Classifier Architecture
<img src="https://github.com/Antonio-Cruciani/classification-using-ecg/blob/master/TwoClassArch.png" title="Architecture" alt="Architecture" height=500 width=1000>


List of Learning Algorithms:

        1) Support Vector Machine
        2) Random Forest
        3) ADA Boost
        4) Gradient Boosting
        5) ExtraTree
        6) Extreme Gradient Boosting
        7) Logistic Regression
        8) Multilayer Perceptron 
        9) K-Nearest Neighbour

