
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.metrics import precision_recall_fscore_support,recall_score,precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import RFE, SelectKBest
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,average_precision_score
import math as mt
from boruta import BorutaPy
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.multiclass import OneVsRestClassifier


# This is not complete

def train_svm_k_fold_MI(matrix,target, gamma,linear = True,nfeatures = 15,nsplits= 10,penalty="l2",C=1,multi_class = "ovr",kernel="rbf",degree=3,probability=False,decision_function_shape="ovr"):
    scores = []
    confusion =[]
    features = []
    parameters = {
        "Gamma":gamma,
        "Linear": linear,
        "C":C,
        "Kernel":kernel,
        "Degree": degree,
        "Average":[],
        "Scores":[],
        "Features":[]
    }
   
    if(linear):
        best_svc = LinearSVC(penalty="l2",C = C,multi_class ="ovr")
    else: 
        best_svc = SVC(C = C,kernel = kernel ,gamma = gamma,degree=degree,probability = probability,decision_function_shape= decision_function_shape )
    cv = KFold(n_splits=nsplits, random_state=42, shuffle=False)
    for train_index, test_index in cv.split(matrix):
        #print("Train Index: ", train_index, "\n")
        #print("Test Index: ", test_index)
        X_train, X_test, y_train, y_test = matrix[train_index], matrix[test_index], target[train_index], target[test_index]
        # ---------------- FEATURE SELECTION ------------------------
        select = SelectKBest(mutual_info_classif,k=nfeatures)
        
        # Calculating mutual information on the training set and getting the "restricted training set"
        x_train_fs = select.fit_transform(X_train,y_train)
        # Getting the 3 most relevant features in the training set
        #mutual_information_deleted = np.asarray(mutual_information_deleted)
        mask = select.get_support()
        j = 0
        indexes = []
        for i in mask:
            if i == True:
                indexes.append(j)
            j+=1
        
        # --------------- TRAINING ------------------------------
        # Training the model
        best_svc.fit(x_train_fs, y_train)

        #--------------- TESTING -------------------------------
        # Getting the scores of the model on the test set
        svc_predictions = best_svc.predict(X_test[:,indexes])
        # getting accuracy
        scores.append(best_svc.score(X_test[:,indexes], y_test))
        parameters["Features"].append(indexes)
       
        # getting confusion matrix
        confusion.append(confusion_matrix(y_test,svc_predictions))
    parameters["Scores"].append(scores)
    parameters["Average"] = np.average(scores)
    return (scores,confusion,parameters)


# This algorithm requires 2 days for the training phase.

def train_svm_k_fold_Boruta(matrix,target,gamma,linear = True,nfeatures = 5, nsplits= 10,penalty="l2",C=1,multi_class = "ovr",kernel="rbf",degree=3,probability=False,decision_function_shape="ovr"):
    scores = []
    confusion =[]
    features = []
    parameters = {
        "Gamma":gamma,
        "Linear": linear,
        "C":C,
        "Kernel":kernel,
        "Degree": degree,
        "Average":[],
        "Scores":[],
        "Features":[]
    }
    if(linear):
        best_svc = LinearSVC(penalty="l2",C = C,multi_class ="ovr")
    else: 
        best_svc = SVC(C = C,kernel = kernel ,gamma = gamma,degree=degree,probability = probability,decision_function_shape= decision_function_shape )
    cv = KFold(n_splits=nsplits, random_state=42, shuffle=False)
    for train_index, test_index in cv.split(matrix):
        #print("Train Index: ", train_index, "\n")
        #print("Test Index: ", test_index)
        X_train, X_test, y_train, y_test = matrix[train_index], matrix[test_index], target[train_index], target[test_index]
        # ---------------- FEATURE SELECTION ------------------------
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
        # define Boruta feature selection method
        feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1, max_iter = 50, perc = 90)
        # find all relevant features - 5 features should be selected
        feat_selector.fit(X_train, y_train)
        # check selected features - first 5 features are selected
        feat_selector.support_
        # call transform() on X to filter it down to selected features
        X_train_filtered = feat_selector.transform(X_train)
        
        final_features = list()
        indexes = np.where(feat_selector.support_ == True)
        #print("Features Rilevanti :")
        
        for x in np.nditer(indexes):
            final_features.append(x)
        
        #print(final_features)
        
       
        # --------------- TRAINING ------------------------------
        # Training the model
        best_svc.fit(X_train[:,final_features[:,nfeatures]], y_train)

        #--------------- TESTING -------------------------------
        # Getting the scores of the model on the test set
        svc_predictions = best_svc.predict(X_test[:,final_features[:,nfeatures]])
        # getting accuracy
        scores.append(best_svc.score(X_test[:,final_features[:,nfeatures]], y_test))
        parameters["Features"].append(final_features[:,nfeatures])
        # getting confusion matrix
        confusion.append(confusion_matrix(y_test,svc_predictions))
    parameters["Scores"].append(scores)
    parameters["Average"] = np.average(scores)
    return (scores,confusion,parameters)


def train_svm_k_fold_pca(matrix,target, gamma,linear = True,nsplits= 10,penalty="l2",C=1,multi_class = "ovr",kernel="rbf",degree=3,probability=False,decision_function_shape="ovr"):
    scores = []
    confusion =[]
    features = []
    parameters = {
        "Gamma":gamma,
        "Linear": linear,
        "C":C,
        "Kernel":kernel,
        "Degree": degree,
        "Average":[],
        "Scores":[],
        "PCA_Param":[],
        "Macro":[],
        "Micro":[],
        "Weighted":[]
    }
   
    if(linear):
        best_svc = LinearSVC(penalty="l2",C = C,multi_class ="ovr")
    else: 
        best_svc = SVC(C = C,kernel = kernel ,gamma = gamma,degree=degree,probability = probability,decision_function_shape= decision_function_shape )
    cv = KFold(n_splits=nsplits, random_state=42, shuffle=False)
    for train_index, test_index in cv.split(matrix):
        #print("Train Index: ", train_index, "\n")
        #print("Test Index: ", test_index)
        X_train, X_test, y_train, y_test = matrix[train_index], matrix[test_index], target[train_index], target[test_index]
        # ---------------- FEATURE SELECTION ------------------------
        X_train = X_train 
        X_test = X_test 
        #print("SHAPE TRAIN",X_train.shape)
        #print("SHAPE VAL", X_test.shape)
        
        #components =  min(X_train.shape[0],X_train.shape[1])-1
        components = .95
        #components = "mle"
        pca =  PCA(n_components = components)
        pca.fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        explained_variance = pca.explained_variance_ratio_
        print("Somma explained variance: ",sum(explained_variance))
            
        
        parameters["PCA_Param"].append(pca.get_params())
        # --------------- TRAINING ------------------------------
        # Training the model
        best_svc.fit(X_train_pca, y_train)

        #--------------- TESTING -------------------------------
        # Getting the scores of the model on the test set
        svc_predictions = best_svc.predict(X_test_pca)
        # getting accuracy
        scores.append(best_svc.score(X_test_pca, y_test))
        # Macro
        parameters["Macro"].append(precision_recall_fscore_support(y_test, svc_predictions, average='macro'))
        # Micro
        parameters["Micro"].append(precision_recall_fscore_support(y_test, svc_predictions, average='micro'))
        # Weighted
        parameters["Weighted"].append(precision_recall_fscore_support(y_test, svc_predictions, average='weighted'))
        
        #parameters["Features"].append(indexes)
       
        # getting confusion matrix
        #confusion.append(confusion_matrix(y_test,svc_predictions))
    parameters["Scores"].append(scores)
    parameters["Average"] = np.average(scores)
    return (scores,confusion,parameters)



def train_svm_k_fold_pear(matrix,target, gamma,linear = True,nsplits= 10,penalty="l2",C=1,multi_class = "ovr",kernel="rbf",degree=3,probability=False,decision_function_shape="ovr"):
    scores = []
    confusion =[]
    features = []
    parameters = {
        "Gamma":gamma,
        "Linear": linear,
        "C":C,
        "Kernel":kernel,
        "Degree": degree,
        "Average":[],
        "Scores":[],
        "Features":[],
        "Macro":[],
        "Micro":[],
        "Weighted":[]
    }
   
    if(linear):
        best_svc = LinearSVC(penalty="l2",C = C,multi_class ="ovr")
    else: 
        best_svc = SVC(C = C,kernel = kernel ,gamma = gamma,degree=degree,probability = probability,decision_function_shape= decision_function_shape )
    cv = KFold(n_splits=nsplits, random_state=42, shuffle=False)
    for train_index, test_index in cv.split(matrix):
        #print("Train Index: ", train_index, "\n")
        #print("Test Index: ", test_index)
        X_train, X_test, y_train, y_test = matrix[train_index], matrix[test_index], target[train_index], target[test_index]
        # ---------------- FEATURE SELECTION ------------------------
        X_train_df = pd.DataFrame(X_train)
        corr = X_train_df.corr()
        
        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i+1, corr.shape[0]):
                if corr.iloc[i,j] >= 0.9:
                    if columns[j]:
                        columns[j] = False
        
        j = 0
        indexes = []
        for i in columns:
            if i == True:
                indexes.append(j)
            j+=1
            
        x_train_fs = X_train[:,indexes]
        
        # --------------- TRAINING ------------------------------
        # Training the model
        best_svc.fit(x_train_fs, y_train)

        #--------------- TESTING -------------------------------
        # Getting the scores of the model on the test set
        svc_predictions = best_svc.predict(X_test[:,indexes])
        # getting accuracy
        scores.append(best_svc.score(X_test[:,indexes], y_test))
        # Macro
        parameters["Macro"].append(precision_recall_fscore_support(y_test, svc_predictions, average='macro'))
        # Micro
        parameters["Micro"].append(precision_recall_fscore_support(y_test, svc_predictions, average='micro'))
        # Weighted
        parameters["Weighted"].append(precision_recall_fscore_support(y_test, svc_predictions, average='weighted'))
        
        
        parameters["Features"].append(indexes)
       
        # getting confusion matrix
        confusion.append(confusion_matrix(y_test,svc_predictions))
    parameters["Scores"].append(scores)
    parameters["Average"] = np.average(scores)
    return (scores,confusion,parameters)









def train_svm_k_fold_RFE(matrix,target, gamma,linear = True,nfeatures = 15,nsplits= 10,penalty="l2",C=1,multi_class = "ovr",kernel="rbf",degree=3,probability=False,decision_function_shape="ovr"):
    scores = []
    confusion =[]
    features = []
    parameters = {
        "Gamma":gamma,
        "Linear": linear,
        "C":C,
        "Kernel":kernel,
        "Degree": degree,
        "Average":[],
        "Scores":[],
        "Features":[],
        "Macro":[],
        "Micro":[],
        "Weighted":[]
    }
   
    if(linear):
        best_svc = LinearSVC(penalty="l2",C = C,multi_class ="ovr")
    else: 
        best_svc = SVC(C = C,kernel = kernel ,gamma = gamma,degree=degree,probability = probability,decision_function_shape= decision_function_shape )
    cv = KFold(n_splits=nsplits, random_state=42, shuffle=False)
    for train_index, test_index in cv.split(matrix):
        #print("Train Index: ", train_index, "\n")
        #print("Test Index: ", test_index)
        X_train, X_test, y_train, y_test = matrix[train_index], matrix[test_index], target[train_index], target[test_index]
        # ---------------- FEATURE SELECTION ------------------------
        
        rforest = RandomForestClassifier(random_state=101)
        rfe = RFE(estimator = rforest,n_features_to_select  = nfeatures)
        rfe.fit(X_train,y_train)
        support = rfe.support_ 
        
        j = 0
        indexes = []
        for i in support:
            if i == True:
                indexes.append(j)
            j+=1
            
        x_train_fs = X_train[:,indexes]
        
        # --------------- TRAINING ------------------------------
        # Training the model
        best_svc.fit(x_train_fs, y_train)

        #--------------- TESTING -------------------------------
        # Getting the scores of the model on the test set
        svc_predictions = best_svc.predict(X_test[:,indexes])
        # getting accuracy
        scores.append(best_svc.score(X_test[:,indexes], y_test))
        # Macro
        parameters["Macro"].append(precision_recall_fscore_support(y_test, svc_predictions, average='macro'))
        # Micro
        parameters["Micro"].append(precision_recall_fscore_support(y_test, svc_predictions, average='micro'))
        # Weighted
        parameters["Weighted"].append(precision_recall_fscore_support(y_test, svc_predictions, average='weighted'))
        
        parameters["Features"].append(indexes)
        
        # getting confusion matrix
        confusion.append(confusion_matrix(y_test,svc_predictions))
    parameters["Scores"].append(scores)
    parameters["Average"] = np.average(scores)
    return (scores,confusion,parameters)



def train_svm_k_fold_FA(matrix,target, gamma,linear = True,nsplits= 10,penalty="l2",C=1,multi_class = "ovr",kernel="rbf",degree=3,probability=False,decision_function_shape="ovr"):
    scores = []
    confusion =[]
    features = []
    parameters = {
        "Gamma":gamma,
        "Linear": linear,
        "C":C,
        "Kernel":kernel,
        "Degree": degree,
        "Average":[],
        "Scores":[],
        "PCA_Param":[],
        "Macro":[],
        "Micro":[],
        "Weighted":[]
    }
   
    if(linear):
        best_svc = LinearSVC(penalty="l2",C = C,multi_class ="ovr")
    else: 
        best_svc = SVC(C = C,kernel = kernel ,gamma = gamma,degree=degree,probability = probability,decision_function_shape= decision_function_shape )
    cv = KFold(n_splits=nsplits, random_state=42, shuffle=False)
    for train_index, test_index in cv.split(matrix):
        #print("Train Index: ", train_index, "\n")
        #print("Test Index: ", test_index)
        X_train, X_test, y_train, y_test = matrix[train_index], matrix[test_index], target[train_index], target[test_index]
        # ---------------- FEATURE SELECTION ------------------------
        X_train = X_train 
        X_test = X_test 
        #print("SHAPE TRAIN",X_train.shape)
        #print("SHAPE VAL", X_test.shape)
        
        #components =  min(X_train.shape[0],X_train.shape[1])-1
        #components = .95
        #components = "mle"
        transformer =  FactorAnalysis(n_components=7,svd_method="randomized")
        X_train_transformed= transformer.fit_transform(X_train)
        X_test_transformed= transformer.fit_transform(X_test)
        
            
        
        parameters["PCA_Param"].append(transformer.get_params())
        # --------------- TRAINING ------------------------------
        # Training the model
        best_svc.fit(X_train_transformed, y_train)

        #--------------- TESTING -------------------------------
        # Getting the scores of the model on the test set
        svc_predictions = best_svc.predict(X_test_transformed)
        # getting accuracy
        scores.append(best_svc.score(X_test_transformed, y_test))
        # Macro
        parameters["Macro"].append(precision_recall_fscore_support(y_test, svc_predictions, average='macro'))
        # Micro
        parameters["Micro"].append(precision_recall_fscore_support(y_test, svc_predictions, average='micro'))
        # Weighted
        parameters["Weighted"].append(precision_recall_fscore_support(y_test, svc_predictions, average='weighted'))
        
        #parameters["Features"].append(indexes)
       
        # getting confusion matrix
        #confusion.append(confusion_matrix(y_test,svc_predictions))
    parameters["Scores"].append(scores)
    parameters["Average"] = np.average(scores)
    return (scores,confusion,parameters)



def train_svm_k_fold(matrix,target, gamma,linear = True,nsplits= 10,penalty="l2",C=1,multi_class = "ovr",kernel="rbf",degree=3,probability=False,decision_function_shape="ovr"):
    scores = []
    confusion =[]
    features = []
    parameters = {
        "Gamma":gamma,
        "Linear": linear,
        "C":C,
        "Kernel":kernel,
        "Degree": degree,
        "Average":[],
        "Scores":[],
        "Features":[i for i in range(0,matrix.shape[1])],
        "Macro":[],
        "Micro":[],
        "Weighted":[]
    }

    if(linear):
        best_svc = LinearSVC(penalty="l2",C = C,multi_class ="ovr")
    else: 
        best_svc = SVC(C = C,kernel = kernel ,gamma = gamma,degree=degree,probability = probability,decision_function_shape= decision_function_shape )
    cv = KFold(n_splits=nsplits, random_state=42, shuffle=False)
    for train_index, test_index in cv.split(matrix):
        #print("Train Index: ", train_index, "\n")
        #print("Test Index: ", test_index)
        X_train, X_test, y_train, y_test = matrix[train_index], matrix[test_index], target[train_index], target[test_index]
        
        # --------------- TRAINING ------------------------------
        # Training the model
        best_svc.fit(X_train, y_train)

        #--------------- TESTING -------------------------------
        # Getting the scores of the model on the test set
        svc_predictions = best_svc.predict(X_test)
        # getting accuracy
        scores.append(best_svc.score(X_test, y_test))
        # Macro
        parameters["Macro"].append(precision_recall_fscore_support(y_test, svc_predictions, average='macro'))
        # Micro
        parameters["Micro"].append(precision_recall_fscore_support(y_test, svc_predictions, average='micro'))
        # Weighted
        parameters["Weighted"].append(precision_recall_fscore_support(y_test, svc_predictions, average='weighted'))
        
        
       
        # getting confusion matrix
        confusion.append(confusion_matrix(y_test,svc_predictions))
    parameters["Scores"].append(scores)
    parameters["Average"] = np.average(scores)
    return (scores,confusion,parameters)



def test_svm(Datasets,best_scores,imputations,title,fs=True,pca=False):
    u= 0 
    test_accuracy_list = []
    test_precision_list = []
    test_recall_list = []
    for dataset in Datasets:
        #print(phases[u])
        # Getting Parameters for the specific dataset
        c = best_scores[u][2]['C']
        g = best_scores[u][2]['Gamma']
        k = best_scores[u][2]['Kernel']
        deg = best_scores[u][2]['Degree']
        if (fs and not pca):
            # Getting the features of the best cross validation execution
            index = 0
            argmax = 0
            p = 0
            for i in best_scores[u][2]['Scores'][0]:

                if i> argmax:
                    argmax = i
                    index = p
                p+=1
            best_feats = best_scores[u][2]['Features'][index]
            x_train = dataset[0][:,best_feats]
            x_test = dataset[2][:,best_feats]
        elif(not fs and not pca):
            best_feats = [i for i in range(0,dataset[0].shape[1])]
            x_train = dataset[0][:,best_feats]
            x_test = dataset[2][:,best_feats]
        elif(not fs and pca):
            
            #components =  min(X_train.shape[0],X_train.shape[1])-1
            components = .95
            #components = "mle"
            pca =  PCA(n_components = components)
            pca.fit(dataset[0])
            x_train = pca.transform(dataset[0])
            x_test =  pca.transform(dataset[2])
            explained_variance = pca.explained_variance_ratio_
            print("Somma explained variance: ",sum(explained_variance))
            
            
        # Training the model with the entire training set
        SVC_MODEL = SVC(C = c,kernel = k ,gamma = g,degree=deg,probability = False,decision_function_shape= "ovr" )
        SVC_MODEL.fit(x_train, dataset[1])
        # Getting accuracy of the model
        svc_predictions = SVC_MODEL.predict(x_test)

    

        # getting accuracy
        test_recall_list.append(recall_score(svc_predictions,dataset[3],average = "macro"))
        test_precision_list.append(precision_score(svc_predictions,dataset[3],average = "macro"))
        test_accuracy_list.append(SVC_MODEL.score(x_test, dataset[3]))
        #print (SVC_MODEL.score(dataset[2][:,best_feats], dataset[3]))
        #print(svc_predictions)
        print("RECALL "+imputations[u],recall_score(svc_predictions,dataset[3],average = "macro"))
        print("PRECISION "+imputations[u],precision_score(svc_predictions,dataset[3],average = "macro"))
        
        #precision_recall_classes(confusion_matrix(dataset[3],svc_predictions))
        
        
        
        
        
        u+=1
    
    
    
    svc_scores_df = pd.DataFrame(test_accuracy_list,columns=["Accuracy"],index=imputations)
    fig, ax = plt.subplots(figsize=(15,7))
    svc_scores_df.plot(kind="bar",ax=ax)
    ax.set_title("Plot Test Accuracy of the SVC Model on the different kind of datasets with "+title)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Datasets')
    fig.tight_layout()
    plt.show()
    print(svc_scores_df)
    return(test_accuracy_list,test_recall_list,test_precision_list)




# Parameter TUNING



def tune_svm(Kernels,Degrees,Cs,Gamma,nfeats,Trains,Datasets,path):

	best_scores_rfe = []
	best_scores_pear = []
	summary_rfe = []
	summary_pear = []
	phases = ["DATASET ELIMINAZIONE COLONNE","IMPUTATION MEDIA","IMPUTATION MEDIANI","IMPUTATION MODA","IMPUTATION REGRESSIONE LINEARE DETERMINISTICA","IMPUTATION REGRESSIONE LINEARE STOCASTICA"]
	otherresults_rfe = []
	otherresults_pear = []
	best_scores_nofs=[]
	summary_nofs = []
	otherresults_nofs = []
	best_scores_pca=[]
	summary_pca = []
	otherresults_pca = []

	u = 0 
	for Tset in Trains:
	    print(phases[u])
	    esecuzioni_rfe = []
	    esecuzioni_pear = []
	    esecuzioni_nofs = []
	    esecuzioni_pca = []
	    for i in Kernels:
	        if i!= "linear":
	            for k in Degrees:
	                for o in Cs:
	                    for g in Gamma:
	                        # RFE
	                        scores_rfe,confusion_rfe,executions_rfe = train_svm_k_fold_RFE(Tset[0],Tset[1],g,linear = False,nsplits=10,penalty="l2",C=o,multi_class = "ovr",kernel=i,degree=k,probability=False,decision_function_shape="ovr")
	                        esecuzioni_rfe.append(executions_rfe)
	                        otherresults_rfe.append([np.average(scores_rfe),confusion_rfe,executions_rfe])
	                        # Pearson
	                        scores_pear,confusion_pear,executions_pear = train_svm_k_fold_pear(Tset[0],Tset[1],g,linear = False,nsplits=10,penalty="l2",C=o,multi_class = "ovr",kernel=i,degree=k,probability=False,decision_function_shape="ovr")
	                        esecuzioni_pear.append(executions_pear)
	                        otherresults_pear.append([np.average(scores_pear),confusion_pear,executions_pear])
	                        # No FS
	                        scores_nofs,confusion_nofs,executions_nofs = train_svm_k_fold(Tset[0],Tset[1],g,linear = False,nsplits=10,penalty="l2",C=o,multi_class = "ovr",kernel=i,degree=k,probability=False,decision_function_shape="ovr")
	                        esecuzioni_nofs.append(executions_nofs)
	                        otherresults_nofs.append([np.average(scores_nofs),confusion_nofs,executions_nofs])
	                        # PCA
	                        scores_pca,confusion_pca,executions_pca = train_svm_k_fold_pca(Tset[0],Tset[1],g,linear = False,nsplits=10,penalty="l2",C=o,multi_class = "ovr",kernel=i,degree=k,probability=False,decision_function_shape="ovr")
	                        esecuzioni_pca.append(executions_pca)
	                        otherresults_pca.append([np.average(scores_pca),confusion_pca,executions_pca])
	                        
	                        
	                        print("FINE ESECUZIONE SVM ")
	                        print("Kernel ",i)
	                        print("Grado",k)
	                        print("C",o)
	                        print("Gamma",g)
	                        print(" RISULTATI ACCURACY ")
	                        print("------ RFE------")
	                        print(np.average(scores_rfe))
	                        print("-----PEAR-------")
	                        print(np.average(scores_pear))
	                        print(" ----- NO FS----- ")
	                        print(np.average(scores_nofs))
	                        print(" ----- PCA ----- ")
	                        print(np.average(scores_pca))
	        else:
	            for o in Cs:
	                # RFE
	                scores_rfe,confusion_rfe,executions_rfe = train_svm_k_fold_RFE(Tset[0],Tset[1],1,linear = True,nsplits=10,penalty="l2",C=o,multi_class = "ovr",kernel=i,degree=0,probability=False,decision_function_shape="ovr")
	                esecuzioni_rfe.append(executions_rfe)
	                otherresults_rfe.append([np.average(scores_rfe),confusion_rfe,executions_rfe])
	                # Pearson
	                scores_pear,confusion_pear,executions_pear = train_svm_k_fold_pear(Tset[0],Tset[1],1,linear = True,nsplits=10,penalty="l2",C=o,multi_class = "ovr",kernel=i,degree=0,probability=False,decision_function_shape="ovr")
	                esecuzioni_pear.append(executions_pear)
	                otherresults_pear.append([np.average(scores_pear),confusion_pear,executions_pear])
	                # NO FS
	                scores_nofs,confusion_nofs,executions_nofs = train_svm_k_fold(Tset[0],Tset[1],1,linear = True,nsplits=10,penalty="l2",C=o,multi_class = "ovr",kernel=i,degree=0,probability=False,decision_function_shape="ovr")
	                esecuzioni_nofs.append(executions_nofs)
	                otherresults_nofs.append([np.average(scores_nofs),confusion_nofs,executions_nofs])
	                # PCA
	                scores_pca,confusion_pca,executions_pca = train_svm_k_fold_pca(Tset[0],Tset[1],1,linear = True,nsplits=10,penalty="l2",C=o,multi_class = "ovr",kernel=i,degree=0,probability=False,decision_function_shape="ovr")
	                esecuzioni_pca.append(executions_pca)
	                otherresults_pca.append([np.average(scores_pca),confusion_pca,executions_pca])
	                
	                
	                print(" FINE ESECUZIONE SVM")
	                print("Kenrel",i)
	                print("C",o)
	                print(" RISULTATI ACCURACY ")
	                print("------ RFE------")
	                print(np.average(scores_rfe))
	                print("-----PEAR-------")
	                print(np.average(scores_pear))
	                print("-----NO FS-------")
	                print(np.average(scores_nofs))
	                print(" ----- PCA ----- ")
	                print(np.average(scores_pca))
	                

	    print("\n\n -------- FINE DATASETS -------------- \n\n ")
	    summary_rfe.append(esecuzioni_rfe)
	    summary_pear.append(esecuzioni_pear)
	    summary_nofs.append(esecuzioni_nofs)
	    summary_pca.append(esecuzioni_pca)
	    # RFE
	    j = 0
	    j_best = 0
	    best_avg_result_rfe = 0
	    for i in esecuzioni_rfe:
	        if i['Average']> best_avg_result_rfe:
	            best_avg_result_rfe = i['Average']
	            j_best = j
	        j+=1
	    print(" Best Score RFE: ",best_avg_result_rfe, " Index :",j_best)
	    best_scores_rfe.append([best_avg_result_rfe,j_best,esecuzioni_rfe[j_best]])
	    # Pearson
	    j = 0
	    j_best = 0
	    best_avg_result_pear = 0
	    for i in esecuzioni_pear:
	        if i['Average']> best_avg_result_pear:
	            best_avg_result_pear = i['Average']
	            j_best = j
	        j+=1
	    print(" Best Score Pearson: ",best_avg_result_pear, " Index :",j_best)
	    best_scores_pear.append([best_avg_result_pear,j_best,esecuzioni_pear[j_best]])
	    # NO FS
	    j = 0
	    j_best = 0
	    best_avg_result_nofs = 0
	    for i in esecuzioni_nofs:
	        if i['Average']> best_avg_result_nofs:
	            best_avg_result_nofs = i['Average']
	            j_best = j
	        j+=1
	    print(" Best Score noFS: ",best_avg_result_nofs, " Index :",j_best)
	    best_scores_nofs.append([best_avg_result_nofs,j_best,esecuzioni_nofs[j_best]])
	    # PCA
	    j = 0
	    j_best = 0
	    best_avg_result_pca = 0
	    for i in esecuzioni_pca:
	        if i['Average']> best_avg_result_pca:
	            best_avg_result_pca = i['Average']
	            j_best = j
	        j+=1
	    print(" Best Score PCA: ",best_avg_result_pca, " Index :",j_best)
	    best_scores_pca.append([best_avg_result_pca,j_best,esecuzioni_pca[j_best]])
	    
	    u+=1
	    
	    
	np.save(path+"/otherresults_nofs.npy", otherresults_nofs)
	np.save(path+"/otherresults_rfe.npy", otherresults_rfe)
	np.save(path+"/otherresults_pear.npy", otherresults_pear)
	np.save(path+"/otherresults_pca.npy", otherresults_pca)


	np.save(path+"/best_scores_nofs.npy", best_scores_nofs)
	np.save(path+"/best_scores_rfe.npy", best_scores_rfe)
	np.save(path+"/best_scores_pear.npy", best_scores_pear)
	np.save(path+"/best_scores_pca.npy", best_scores_pca)


