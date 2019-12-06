import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support,recall_score,precision_score
from sklearn.feature_selection import RFE, SelectKBest
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer,scale,MinMaxScaler,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,average_precision_score
from scipy import stats
import math as mt
import scipy as sp
import missingno as mno

from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score

#from Functions.Evaluation.class_precision import precision_recall_classes

def precision_recall_classes(confusion_matrix,target):
    precisions=[]
    recalls = []
    #print(confusion_matrix)
    #print("SHAPE",confusion_matrix.shape)
    #print("Len Target ",len(target))
    #print("TARGET VALUES ", target)
    for i in range(0,len(target)):
        #print("CLASS",target[i])
        true_pos = confusion_matrix[i][i]
        #print("True positive", true_pos)
        false_pos = np.sum(confusion_matrix[i,:]) - true_pos
        #print("False pos",false_pos)
        false_neg = np.sum(confusion_matrix[:,i]) - true_pos
        #print("False neg", false_neg)
        precision = np.sum(true_pos / (true_pos + false_pos))
        #print("Precision ",precision)
        recall = np.sum(true_pos / (true_pos + false_neg))
        #print("Recall ",recall)
        precisions.append(precision)
        recalls.append(recall)
    
    single_precision_recall={
        "Precision":precisions,
        "Recall":recalls
    }
    #if(len(target)!= confusion_matrix.shape[0])
    precision_recall = pd.DataFrame(single_precision_recall,index=target)
    #print(single_precision_recall)
    return (precision_recall.dropna(thresh=2))





# Funzioni per la Feature selection e PCA

def Pearson_FS(dataset):
    dataset_df = pd.DataFrame(dataset)
    corr = dataset_df.corr()
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
    return(indexes)

def RFE_FS(X_train,y_trains,nfeatures):
   
    model = RandomForestClassifier()
    #rforest = RandomForestClassifier(random_state=101)
    rfe = RFE(estimator = model,n_features_to_select  = nfeatures)
    rfe.fit(X_train,y_trains)

    support = rfe.support_ 

    j = 0
    indexes = []
    for i in support:
        if i == True:
            indexes.append(j)
        j+=1
    return(indexes)

def get_PCA(X_train,X_test):
    components = .95
    #components = "mle"
    pca =  PCA(n_components = components)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    return(X_train_pca,X_test_pca)






def random_forest_cross_rfe(matrix,target,n_trees=None,criterion=None,max_depth=None,nsplits = 10,nfeatures=15 ):
    scores = []
    recalls = []
    confusion =[]
    features = []
    precisions = []
    parameters = {
        "N_trees":n_trees,
        "Criterion": criterion,
        "Max_depth":max_depth,
        "Average":[],
        "Scores":[],
        "Features":[],
        "Macro":[],
        "Micro":[],
        "Weighted":[]
    }
    
    random_f_class = RandomForestClassifier(n_estimators=n_trees,criterion=criterion,max_depth=max_depth)
    
    cv = KFold(n_splits=nsplits, random_state=42, shuffle=False)
    for train_index, test_index in cv.split(matrix):
        #print("Train Index: ", train_index, "\n")
        #print("Test Index: ", test_index)
        X_train, X_test, y_train, y_test = matrix[train_index], matrix[test_index], np.array(target)[train_index.astype(int)], np.array(target)[test_index.astype(int)]
        # ---------------- FEATURE SELECTION ------------------------
        model = RandomForestClassifier()
        #rforest = RandomForestClassifier(random_state=101)
        rfe = RFE(estimator = model,n_features_to_select  = nfeatures)
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
        random_f_class.fit(x_train_fs, y_train)

        #--------------- TESTING -------------------------------
        # Getting the scores of the model on the test set
        rndf_predictions = random_f_class.predict(X_test[:,indexes])
        # getting accuracy
        scores.append(random_f_class.score(X_test[:,indexes], y_test))
        recalls.append(recall_score(rndf_predictions,y_test,average = "macro"))
        precisions.append(precision_score(rndf_predictions,y_test,average = "macro"))
        # Macro
        parameters["Macro"].append(precision_recall_fscore_support(y_test, rndf_predictions, average='macro'))
        # Micro
        parameters["Micro"].append(precision_recall_fscore_support(y_test, rndf_predictions, average='micro'))
        # Weighted
        parameters["Weighted"].append(precision_recall_fscore_support(y_test, rndf_predictions, average='weighted'))
        
        parameters["Features"].append(indexes)
       
        # getting confusion matrix
        confusion.append(confusion_matrix(y_test,rndf_predictions))
    parameters["Scores"].append(scores)
    parameters["Average"].append([np.average(scores),np.average(recalls),np.average(precisions)])
    print("AVG SCORES = ",  np.average(scores))
    return (scores,confusion,parameters)
    
def random_forest_cross_pear(matrix,target,n_trees=None,criterion=None,max_depth=None,nsplits = 10,nfeatures=15 ):
    scores = []
    recalls = []
    confusion =[]
    features = []
    precisions = []
    parameters = {
        "N_trees":n_trees,
        "Criterion": criterion,
        "Max_depth":max_depth,
        "Average":[],
        "Scores":[],
        "Features":[],
        "Macro":[],
        "Micro":[],
        "Weighted":[]
    }
   
    random_f_class = RandomForestClassifier(n_estimators=n_trees,criterion=criterion,max_depth=max_depth)
    cv = KFold(n_splits=nsplits, random_state=42, shuffle=False)
    for train_index, test_index in cv.split(matrix):
        #print("Train Index: ", train_index, "\n")
        #print("Test Index: ", test_index)
        #print(type(matrix))
        #print(type(target))
        X_train, X_test, y_train, y_test = matrix[train_index], matrix[test_index], np.array(target)[train_index.astype(int)], np.array(target)[test_index.astype(int)]
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
        random_f_class.fit(x_train_fs, y_train)

        #--------------- TESTING -------------------------------
        # Getting the scores of the model on the test set
        pear_rndf_pred = random_f_class.predict(X_test[:,indexes])
        # getting accuracy
        
        scores.append(random_f_class.score(X_test[:,indexes], y_test))
        recalls.append(recall_score(pear_rndf_pred,y_test,average = "macro"))
        precisions.append(precision_score(pear_rndf_pred,y_test,average = "macro"))
        parameters["Macro"].append(precision_recall_fscore_support(y_test, pear_rndf_pred, average='macro'))
        # Micro
        parameters["Micro"].append(precision_recall_fscore_support(y_test, pear_rndf_pred, average='micro'))
        # Weighted
        parameters["Weighted"].append(precision_recall_fscore_support(y_test, pear_rndf_pred, average='weighted'))
        
        
        parameters["Features"].append(indexes)
       
        # getting confusion matrix
        confusion.append(confusion_matrix(y_test,pear_rndf_pred))
        
    parameters["Scores"].append(scores)
    parameters["Average"].append([np.average(scores),np.average(recalls),np.average(precisions)])
    print("AVG SCORES = ",  np.average(scores))
    return (scores,confusion,parameters)


def random_forest_cross_pca(matrix,target, n_trees=None,criterion=None,max_depth=None,nsplits = 10):
    precisions = []
    recalls = []
    scores = []
    confusion =[]
    features = []
    parameters = {
        "N_trees":n_trees,
        "Criterion": criterion,
        "Max_depth":max_depth,
        "Average":[],
        "Scores":[],
        "PCA_Param":[],
        "Macro":[],
        "Micro":[],
        "Weighted":[]
    }
   
    random_f_class = RandomForestClassifier(n_estimators=n_trees,criterion=criterion,max_depth=max_depth)
    cv = KFold(n_splits=nsplits, random_state=42, shuffle=False)
    for train_index, test_index in cv.split(matrix):
        #print("Train Index: ", train_index, "\n")
        #print("Test Index: ", test_index)
        X_train, X_test, y_train, y_test = matrix[train_index], matrix[test_index],np.array(target)[train_index.astype(int)], np.array(target)[test_index.astype(int)]
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
        random_f_class.fit(X_train_pca, y_train)

        #--------------- TESTING -------------------------------
        # Getting the scores of the model on the test set
        pear_rndf_pred = random_f_class.predict(X_test_pca)
        # getting accuracy
        scores.append(random_f_class.score(X_test_pca, y_test))
        recalls.append(recall_score(pear_rndf_pred,y_test,average = "macro"))
        precisions.append(precision_score(pear_rndf_pred,y_test,average = "macro"))
        # Macro
        parameters["Macro"].append(precision_recall_fscore_support(y_test, pear_rndf_pred, average='macro'))
        # Micro
        parameters["Micro"].append(precision_recall_fscore_support(y_test, pear_rndf_pred, average='micro'))
        # Weighted
        parameters["Weighted"].append(precision_recall_fscore_support(y_test, pear_rndf_pred, average='weighted'))
        #parameters["Features"].append(indexes)
       
        # getting confusion matrix
        #confusion.append(confusion_matrix(y_test,svc_predictions))
    parameters["Scores"].append(scores)
    parameters["Average"].append([np.average(scores),np.average(recalls),np.average(precisions)])
    print("AVG SCORES = ",  np.average(scores))
    return (scores,confusion,parameters)


def random_forest_cross_nofs(matrix,target, n_trees=None,criterion=None,max_depth=None,nsplits = 10):
    precisions = []
    recalls = []
    scores = []
    confusion =[]

    parameters = {
        "N_trees":n_trees,
        "Criterion": criterion,
        "Max_depth":max_depth,
        "Average":[],
        "Scores":[],
        "Macro":[],
        "Micro":[],
        "Weighted":[]
    }
   
    random_f_class = RandomForestClassifier(n_estimators=n_trees,criterion=criterion,max_depth=max_depth)
    cv = KFold(n_splits=nsplits, random_state=42, shuffle=False)
    for train_index, test_index in cv.split(matrix):
        #print("Train Index: ", train_index, "\n")
        #print("Test Index: ", test_index)
        X_train, X_test, y_train, y_test = matrix[train_index], matrix[test_index],np.array(target)[train_index.astype(int)], np.array(target)[test_index.astype(int)]
        # ---------------- FEATURE SELECTION ------------------------
        X_train = X_train 
        X_test = X_test 
        #print("SHAPE TRAIN",X_train.shape)
        #print("SHAPE VAL", X_test.shape)
        
       
        
      
        # --------------- TRAINING ------------------------------
        # Training the model
        random_f_class.fit(X_train, y_train)

        #--------------- TESTING -------------------------------
        # Getting the scores of the model on the test set
        pear_rndf_pred = random_f_class.predict(X_test)
        # getting accuracy
        scores.append(random_f_class.score(X_test, y_test))
        recalls.append(recall_score(pear_rndf_pred,y_test,average = "macro"))
        precisions.append(precision_score(pear_rndf_pred,y_test,average = "macro"))
        # Macro
        parameters["Macro"].append(precision_recall_fscore_support(y_test, pear_rndf_pred, average='macro'))
        # Micro
        parameters["Micro"].append(precision_recall_fscore_support(y_test, pear_rndf_pred, average='micro'))
        # Weighted
        parameters["Weighted"].append(precision_recall_fscore_support(y_test, pear_rndf_pred, average='weighted'))
        #parameters["Features"].append(indexes)
       
        # getting confusion matrix
        #confusion.append(confusion_matrix(y_test,svc_predictions))
    parameters["Scores"].append(scores)
    parameters["Average"].append([np.average(scores),np.average(recalls),np.average(precisions)])
    print("AVG SCORES = ",  np.average(scores))
    return (scores,confusion,parameters)


def training_random_forest_pearson(Datasets,imputations):

	u = 0
	scores_rf_pear = []
	lista_n = []
	confusion_matrix_test_forest_gini = []
	predicted_forest_pear_gini = []

	for i in Datasets:
	    print("INIZIO ",imputations[u])
	    x_train = i[0]
	    y_train = i[1]
	    x_test = i[2]
	    y_test = i[3]
	    best_num, best_meas = 0,0
	    g = 0
	    n,f = [],[]

	    for i in range(50,500,4):
	        #Rand_for = RandomForestClassifier(n_estimators=i)
	        #result_c_v_randomForest = cross_val_score(Rand_for, x_train, y_train, scoring='recall',cv=5).mean()
	        scores_c_v_randomForest,confusion_c_v_randomForest,parameters_c_v_random_forest = random_forest_cross_pear(x_train,y_train,n_trees = i,criterion = 'gini',max_depth= None,nsplits = 5,nfeatures = 10)

	        n.append(i)
	        f.append(parameters_c_v_random_forest['Average'][0][1])
	        if parameters_c_v_random_forest['Average'][0][1] > best_meas:
	            best_meas = parameters_c_v_random_forest['Average'][0][1]
	            best_params = parameters_c_v_random_forest
	            best_num = i
	        print ('n = {0:3d}; Recall = {1:0.4f}'.format(i, parameters_c_v_random_forest['Average'][0][1]))
	        print ('n = {0:3d}; Precision = {1:0.4f}'.format(i, parameters_c_v_random_forest['Average'][0][2]))
	        g+=1
	    print("INDEX : ",g)
	    print("Best = {0:0.4f} con n = {1:3d}".format(best_meas, best_num))
	    lista_n.append([n,f])
	    #--- TEST
	    number_of_t = best_num
	    foresta =RandomForestClassifier(n_estimators=number_of_t,criterion='gini',max_depth=None)
	    features = best_params['Features'][4]
	    foresta.fit(x_train[:,features], y_train)
	    foresta_pred = foresta.predict(x_test[:,features])
	    accuracy = foresta.score(x_test[:,features], y_test)
	    rec = recall_score(foresta_pred,y_test,average = "macro")
	    prec = precision_score(foresta_pred,y_test,average = "macro")
	    print("ACCURACY SUL TEST ",accuracy)
	    single_precision_recall_random_f_pearson=precision_recall_classes(confusion_matrix(y_test,foresta_pred),list(set(y_test)))
	    scores_rf_pear.append([accuracy,rec,prec,single_precision_recall_random_f_pearson])
	    confusion_matrix_test_forest_gini.append(confusion_matrix(y_test,foresta_pred))
	    predicted_forest_pear_gini.append(foresta_pred)
	    print("FINE ",imputations[u])
	    print("---------------------------------\n\n")
	    u+=1

	return(scores_rf_pear,lista_n,confusion_matrix_test_forest_gini,predicted_forest_pear_gini)


def training_random_forest_rfe(Datasets,imputations):

	u = 0
	scores_rf_rfe = []
	lista_n = []
	confusion_matrix_test_forest_gini_rfe = []
	predicted_forest_rfe_gini = []

	for i in Datasets:
	    print("INIZIO ",imputations[u])
	    x_train = i[0]
	    y_train = i[1]
	    x_test = i[2]
	    y_test = i[3]
	    best_num, best_meas = 0,0
	    g = 0
	    n,f = [],[]

	    for i in range(100,500,10):
	        #Rand_for = RandomForestClassifier(n_estimators=i)
	        #result_c_v_randomForest = cross_val_score(Rand_for, x_train, y_train, scoring='recall',cv=5).mean()
	        scores_c_v_randomForest,confusion_c_v_randomForest,parameters_c_v_random_forest = random_forest_cross_rfe(x_train,y_train,n_trees = i,criterion = 'gini',max_depth= None,nsplits = 5,nfeatures = 10)

	        n.append(i)
	        f.append(parameters_c_v_random_forest['Average'][0][1])
	        if parameters_c_v_random_forest['Average'][0][1] > best_meas:
	            best_meas = parameters_c_v_random_forest['Average'][0][1]
	            best_params = parameters_c_v_random_forest
	            best_num = i
	        print ('n = {0:3d}; Recall = {1:0.4f}'.format(i, parameters_c_v_random_forest['Average'][0][1]))
	        print ('n = {0:3d}; Precision = {1:0.4f}'.format(i, parameters_c_v_random_forest['Average'][0][2]))
	        g+=1
	    print("INDEX : ",g)
	    print("Best = {0:0.4f} con n = {1:3d}".format(best_meas, best_num))
	    lista_n.append([n,f])
	    #--- TEST
	    number_of_t = best_num
	    foresta =RandomForestClassifier(n_estimators=number_of_t,criterion='gini',max_depth=None)
	    features = best_params['Features'][4]
	    foresta.fit(x_train[:,features], y_train)
	    foresta_pred = foresta.predict(x_test[:,features])
	    accuracy = foresta.score(x_test[:,features], y_test)
	    rec = recall_score(foresta_pred,y_test,average = "macro")
	    prec = precision_score(foresta_pred,y_test,average = "macro")
	    print("ACCURACY SUL TEST ",accuracy)
	    single_precision_recall_random_f_rfe=precision_recall_classes(confusion_matrix(y_test,foresta_pred),list(set(y_test)))
	    scores_rf_rfe.append([accuracy,rec,prec,single_precision_recall_random_f_rfe])
	    confusion_matrix_test_forest_gini_rfe.append(confusion_matrix(y_test,foresta_pred))
	    
	    predicted_forest_rfe_gini.append(foresta_pred)
	    print("FINE ",imputations[u])
	    print("---------------------------------\n\n")
	    u+=1

	return(scores_rf_rfe,lista_n,confusion_matrix_test_forest_gini_rfe,predicted_forest_rfe_gini)



def training_random_forest_pca(Datasets,imputations):

	u = 0
	scores_rf_pca = []
	lista_n = []
	confusion_matrix_test_forest_gini_pca = []
	predicted_forest_pca_gini = []

	for i in Datasets:
	    print("INIZIO ",imputations[u])
	    x_train = i[0]
	    y_train = i[1]
	    x_test = i[2]
	    y_test = i[3]
	    best_num, best_meas = 0,0
	    g = 0
	    n,f = [],[]

	    for i in range(50,500,4):
	        #Rand_for = RandomForestClassifier(n_estimators=i)
	        #result_c_v_randomForest = cross_val_score(Rand_for, x_train, y_train, scoring='recall',cv=5).mean()
	        scores_c_v_randomForest,confusion_c_v_randomForest,parameters_c_v_random_forest = random_forest_cross_pca(x_train,y_train,n_trees = i,criterion = 'gini',max_depth= None,nsplits = 5)

	        n.append(i)
	        f.append(parameters_c_v_random_forest['Average'][0][1])
	        if parameters_c_v_random_forest['Average'][0][1] > best_meas:
	            best_meas = parameters_c_v_random_forest['Average'][0][1]
	            best_params = parameters_c_v_random_forest
	            best_num = i
	        print ('n = {0:3d}; Recall = {1:0.4f}'.format(i, parameters_c_v_random_forest['Average'][0][1]))
	        print ('n = {0:3d}; Precision = {1:0.4f}'.format(i, parameters_c_v_random_forest['Average'][0][2]))
	        g+=1
	    print("INDEX : ",g)
	    print("Best = {0:0.4f} con n = {1:3d}".format(best_meas, best_num))
	    lista_n.append([n,f])
	    #--- TEST
	    number_of_t = best_num
	    foresta =RandomForestClassifier(n_estimators=number_of_t,criterion='gini',max_depth=None)
	   

	    components = .95
	    pca =  PCA(n_components = components)
	    pca.fit(x_train)
	    X_train_pca = pca.transform(x_train)
	    X_test_pca = pca.transform(x_test)
	    explained_variance = pca.explained_variance_ratio_
	    print("Somma explained variance: ",sum(explained_variance))
	            
	    foresta.fit(X_train_pca, y_train)
	    foresta_pred = foresta.predict(X_test_pca)
	    accuracy = foresta.score(X_test_pca, y_test)
	    rec = recall_score(foresta_pred,y_test,average = "macro")
	    prec = precision_score(foresta_pred,y_test,average = "macro")
	    print("ACCURACY SUL TEST ",accuracy)
	    single_precision_recall_random_f_pca=precision_recall_classes(confusion_matrix(y_test,foresta_pred),list(set(y_test)))
	    scores_rf_pca.append([accuracy,rec,prec,single_precision_recall_random_f_pca])
	    confusion_matrix_test_forest_gini_pca.append(confusion_matrix(y_test,foresta_pred))
	    predicted_forest_pca_gini.append(foresta_pred)
	    print("FINE ",imputations[u])
	    print("---------------------------------\n\n")
	    u+=1
	return(scores_rf_pca,lista_n,confusion_matrix_test_forest_gini_pca,predicted_forest_pca_gini)

def training_random_forest_nofs(Datasets,imputations):

	u = 0
	scores_rf_nofs = []
	lista_n = []
	confusion_matrix_test_forest_gini_nofs = []
	predicted_forest_nofs_gini = []

	for i in Datasets:
	    print("INIZIO ",imputations[u])
	    x_train = i[0]
	    y_train = i[1]
	    x_test = i[2]
	    y_test = i[3]
	    best_num, best_meas = 0,0
	    g = 0
	    n,f = [],[]

	    for i in range(50,500,4):
	        #Rand_for = RandomForestClassifier(n_estimators=i)
	        #result_c_v_randomForest = cross_val_score(Rand_for, x_train, y_train, scoring='recall',cv=5).mean()
	        scores_c_v_randomForest,confusion_c_v_randomForest,parameters_c_v_random_forest = random_forest_cross_nofs(x_train,y_train,n_trees = i,criterion = 'gini',max_depth= None,nsplits = 5)

	        n.append(i)
	        f.append(parameters_c_v_random_forest['Average'][0][1])
	        if parameters_c_v_random_forest['Average'][0][1] > best_meas:
	            best_meas = parameters_c_v_random_forest['Average'][0][1]
	            best_params = parameters_c_v_random_forest
	            best_num = i
	        print ('n = {0:3d}; Recall = {1:0.4f}'.format(i, parameters_c_v_random_forest['Average'][0][1]))
	        print ('n = {0:3d}; Precision = {1:0.4f}'.format(i, parameters_c_v_random_forest['Average'][0][2]))
	        g+=1
	    print("INDEX : ",g)
	    print("Best = {0:0.4f} con n = {1:3d}".format(best_meas, best_num))
	    lista_n.append([n,f])
	    #--- TEST
	    number_of_t = best_num
	    foresta =RandomForestClassifier(n_estimators=number_of_t,criterion='gini',max_depth=None)
	   

	    
	   
	    foresta.fit(x_train, y_train)
	    foresta_pred = foresta.predict(x_test)
	    accuracy = foresta.score(x_test, y_test)
	    rec = recall_score(foresta_pred,y_test,average = "macro")
	    prec = precision_score(foresta_pred,y_test,average = "macro")
	    print("ACCURACY SUL TEST ",accuracy)
	    single_precision_recall_random_f_nofs=precision_recall_classes(confusion_matrix(y_test,foresta_pred),list(set(y_test)))
	    scores_rf_nofs.append([accuracy,rec,prec,single_precision_recall_random_f_nofs])
	    confusion_matrix_test_forest_gini_nofs.append(confusion_matrix(y_test,foresta_pred))
	    predicted_forest_nofs_gini.append(foresta_pred)
	    print("FINE ",imputations[u])
	    print("---------------------------------\n\n")
	    u+=1
	
	
	return(scores_rf_nofs,lista_n,confusion_matrix_test_forest_gini_nofs,predicted_forest_nofs_gini)



def training_testing_boosting_algorithms_no_FS(Datasets,imputations):
	scores_ada_nofs=[]
	scores_gb_nofs = []
	scores_et_nofs = []
	scores_xgb_nofs = []
	j = 0
	for dataset in Datasets:
	    print ("DATASET ",imputations[j])
	    
	    
	    #--------- Training modello con nofs
	    x_train  = dataset[0]
	    x_test = dataset[2]
	    y_train = dataset[1]
	    y_test = dataset[3]
	    #---- ADA
	    AB = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 12))
	    AB.fit(x_train,y_train)
	    AB_predictions = AB.predict(x_test)
	    
	    AB_scores = AB.score(x_test,y_test)
	    rec_AB = recall_score(AB_predictions,y_test,average = "macro")
	    prec_AB = precision_score(AB_predictions,y_test,average = "macro")
	    scores_ada_nofs.append([AB_scores,rec_AB,prec_AB,precision_recall_classes(confusion_matrix(y_test,AB_predictions),list(set(y_test)))])
	    print("ADA-BOOST")
	    print("Accuracy", AB_scores)
	    print("Precision",prec_AB)
	    print("Recall",rec_AB)
	    #---- GB
	    GB = GradientBoostingClassifier(n_estimators=500)
	    GB.fit(x_train, y_train)
	    GB_predictions = GB.predict(x_test)
	    GB_scores = GB.score(x_test,y_test)
	    rec_GB = recall_score(GB_predictions,y_test,average = "macro")
	    prec_GB = precision_score(GB_predictions,y_test,average = "macro")
	    print(len(set(GB_predictions)))
	    print(len(set(y_test)))
	    print("TEST")
	    print(set(y_test))
	    print("PREDETTE")
	    print(set(GB_predictions))
	    scores_gb_nofs.append([GB_scores,rec_GB,prec_GB,precision_recall_classes(confusion_matrix(y_test,GB_predictions),list(set(y_test)))])
	    print("GRADIENT BOOSTING")
	    print("Accuracy", GB_scores)
	    print("Precision",prec_GB)
	    print("Recall",rec_GB)
	    
	    
	    #---- ET
	    ET = ExtraTreesClassifier(n_estimators=100)
	    ET.fit(x_train,y_train)
	    ET_predictions = ET.predict(x_test)
	    ET_scores = ET.score(x_test,y_test)
	    rec_ET = recall_score(ET_predictions,y_test,average = "macro")
	    prec_ET = precision_score(ET_predictions,y_test,average = "macro")
	    
	    scores_et_nofs.append([ET_scores,rec_ET,prec_ET,precision_recall_classes(confusion_matrix(y_test,ET_predictions),list(set(y_test)))])
	    
	    print("EXTRA TREE")
	    print("Accuracy", ET_scores)
	    print("Precision",prec_ET)
	    print("Recall",rec_ET)
	    
	    
	    
	    #---- XGB
	    
	    XGB = XGBClassifier(learning_rate = 0.05, n_estimators=100, max_depth=15)
	    XGB.fit(x_train, y_train)
	    XGB_predictions = XGB.predict(x_test)
	    predictions = [round(value) for value in XGB_predictions]
	    acc_XGB = accuracy_score(y_test, XGB_predictions)
	    rec_XGB = recall_score(y_test, XGB_predictions,average = "macro")
	    prec_XGB = precision_score(y_test, XGB_predictions,average="macro")
	    scores_xgb_nofs.append([acc_XGB,rec_XGB,prec_XGB,precision_recall_classes(confusion_matrix(y_test,XGB_predictions),list(set(y_test)))])
	    
	    print("X-GRADIENT BOOSTING")
	    print("Accuracy", acc_XGB)
	    print("Precision",prec_XGB)
	    print("Recall",rec_XGB)
	    j+=1

	return(scores_ada_nofs,scores_gb_nofs,scores_et_nofs,scores_xgb_nofs)



def training_testing_boosting_algorithms_Pearson(Datasets,imputations):
	scores_ada_pearson=[]
	scores_gb_pearson = []
	scores_et_pearson = []
	scores_xgb_pearson = []
	j = 0
	for dataset in Datasets:
	    print ("DATASET ",imputations[j])
	    indexes_p = Pearson_FS(dataset[0])
	    
	    #--------- Training modello con Pearson
	    x_train_p = dataset[0][:,indexes_p]
	    x_test_p = dataset[2][:,indexes_p]
	    y_train = dataset[1]
	    y_test = dataset[3]
	    #---- ADA
	    AB = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 12))
	    AB.fit(x_train_p,y_train)
	    AB_predictions = AB.predict(x_test_p)
	    
	    AB_scores = AB.score(x_test_p,y_test)
	    rec_AB = recall_score(AB_predictions,y_test,average = "macro")
	    prec_AB = precision_score(AB_predictions,y_test,average = "macro")
	    scores_ada_pearson.append([AB_scores,rec_AB,prec_AB,precision_recall_classes(confusion_matrix(y_test,AB_predictions),list(set(y_test)))])
	    print("ADA-BOOST")
	    print("Accuracy", AB_scores)
	    print("Precision",prec_AB)
	    print("Recall",rec_AB)
	    #---- GB
	    GB = GradientBoostingClassifier(n_estimators=500)
	    GB.fit(x_train_p, y_train)
	    GB_predictions = GB.predict(x_test_p)
	    GB_scores = GB.score(x_test_p,y_test)
	    rec_GB = recall_score(GB_predictions,y_test,average = "macro")
	    prec_GB = precision_score(GB_predictions,y_test,average = "macro")
	    print(len(set(GB_predictions)))
	    print(len(set(y_test)))
	    print("TEST")
	    print(set(y_test))
	    print("PREDETTE")
	    print(set(GB_predictions))
	    scores_gb_pearson.append([GB_scores,rec_GB,prec_GB,precision_recall_classes(confusion_matrix(y_test,GB_predictions),list(set(y_test)))])
	    print("GRADIENT BOOSTING")
	    print("Accuracy", GB_scores)
	    print("Precision",prec_GB)
	    print("Recall",rec_GB)
	    
	    
	    #---- ET
	    ET = ExtraTreesClassifier(n_estimators=100)
	    ET.fit(x_train_p,y_train)
	    ET_predictions = ET.predict(x_test_p)
	    ET_scores = ET.score(x_test_p,y_test)
	    rec_ET = recall_score(ET_predictions,y_test,average = "macro")
	    prec_ET = precision_score(ET_predictions,y_test,average = "macro")
	    
	    scores_et_pearson.append([ET_scores,rec_ET,prec_ET,precision_recall_classes(confusion_matrix(y_test,ET_predictions),list(set(y_test)))])
	    
	    print("EXTRA TREE")
	    print("Accuracy", ET_scores)
	    print("Precision",prec_ET)
	    print("Recall",rec_ET)
	    
	    
	    
	    #---- XGB
	    
	    XGB = XGBClassifier(learning_rate = 0.05, n_estimators=100, max_depth=15)
	    XGB.fit(x_train_p, y_train)
	    XGB_predictions = XGB.predict(x_test_p)
	    predictions = [round(value) for value in XGB_predictions]
	    acc_XGB = accuracy_score(y_test, XGB_predictions)
	    rec_XGB = recall_score(y_test, XGB_predictions,average = "macro")
	    prec_XGB = precision_score(y_test, XGB_predictions,average="macro")
	    scores_xgb_pearson.append([acc_XGB,rec_XGB,prec_XGB,precision_recall_classes(confusion_matrix(y_test,XGB_predictions),list(set(y_test)))])
	    
	    print("X-GRADIENT BOOSTING")
	    print("Accuracy", acc_XGB)
	    print("Precision",prec_XGB)
	    print("Recall",rec_XGB)
	    j+=1
	return(scores_ada_pearson,scores_gb_pearson,scores_et_pearson,scores_xgb_pearson)

def training_testing_boosting_algorithms_RFE(Datasets,imputations):
	scores_ada_rfe =[]
	scores_gb_rfe = []
	scores_et_rfe = []
	scores_xgb_rfe = []
	j = 0
	for dataset in Datasets:
	    print ("DATASET ",imputations[j])
	    indexes_rfe = RFE_FS(dataset[0],dataset[1],15)
	    #indexes_rfe = RFE_FS(dataset[0],15)
	    
	    #--------- Training modello con Pearson
	    x_train_rfe = dataset[0][:,indexes_rfe]
	    x_test_rfe = dataset[2][:,indexes_rfe]
	    y_train = dataset[1]
	    y_test = dataset[3]
	    #---- ADA
	    AB = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 12))
	    AB.fit(x_train_rfe,y_train)
	    AB_predictions = AB.predict(x_test_rfe)
	    
	    AB_scores = AB.score(x_test_rfe,y_test)
	    rec_AB = recall_score(AB_predictions,y_test,average = "macro")
	    prec_AB = precision_score(AB_predictions,y_test,average = "macro")
	    scores_ada_rfe.append([AB_scores,rec_AB,prec_AB,precision_recall_classes(confusion_matrix(y_test,AB_predictions),list(set(y_test)))])
	    print("ADA-BOOST")
	    print("Accuracy", AB_scores)
	    print("Precision",prec_AB)
	    print("Recall",rec_AB)
	    #---- GB
	    GB = GradientBoostingClassifier(n_estimators=500)
	    GB.fit(x_train_rfe, y_train)
	    GB_predictions = GB.predict(x_test_rfe)
	    GB_scores = GB.score(x_test_rfe,y_test)
	    rec_GB = recall_score(GB_predictions,y_test,average = "macro")
	    prec_GB = precision_score(GB_predictions,y_test,average = "macro")
	    print(len(set(GB_predictions)))
	    print(len(set(y_test)))
	    print("TEST")
	    print(set(y_test))
	    print("PREDETTE")
	    print(set(GB_predictions))
	    scores_gb_rfe.append([GB_scores,rec_GB,prec_GB,precision_recall_classes(confusion_matrix(y_test,GB_predictions),list(set(y_test)))])
	    print("GRADIENT BOOSTING")
	    print("Accuracy", GB_scores)
	    print("Precision",prec_GB)
	    print("Recall",rec_GB)
	    
	    
	    #---- ET
	    ET = ExtraTreesClassifier(n_estimators=100)
	    ET.fit(x_train_rfe,y_train)
	    ET_predictions = ET.predict(x_test_rfe)
	    ET_scores = ET.score(x_test_rfe,y_test)
	    rec_ET = recall_score(ET_predictions,y_test,average = "macro")
	    prec_ET = precision_score(ET_predictions,y_test,average = "macro")
	    
	    scores_et_rfe.append([ET_scores,rec_ET,prec_ET,precision_recall_classes(confusion_matrix(y_test,ET_predictions),list(set(y_test)))])
	    
	    print("EXTRA TREE")
	    print("Accuracy", ET_scores)
	    print("Precision",prec_ET)
	    print("Recall",rec_ET)
	    
	    
	    
	    #---- XGB
	    
	    XGB = XGBClassifier(learning_rate = 0.05, n_estimators=100, max_depth=15)
	    XGB.fit(x_train_rfe, y_train)
	    XGB_predictions = XGB.predict(x_test_rfe)
	    predictions = [round(value) for value in XGB_predictions]
	    acc_XGB = accuracy_score(y_test, XGB_predictions)
	    rec_XGB = recall_score(y_test, XGB_predictions,average = "macro")
	    prec_XGB = precision_score(y_test, XGB_predictions,average="macro")
	    scores_xgb_rfe.append([acc_XGB,rec_XGB,prec_XGB,precision_recall_classes(confusion_matrix(y_test,XGB_predictions),list(set(y_test)))])
	    
	    print("X-GRADIENT BOOSTING")
	    print("Accuracy", acc_XGB)
	    print("Precision",prec_XGB)
	    print("Recall",rec_XGB)
	    j+=1

	return(scores_ada_rfe,scores_gb_rfe,scores_et_rfe,scores_xgb_rfe)


def training_testing_boosting_algorithms_PCA(Datasets,imputations):
	scores_ada_pca =[]
	scores_gb_pca = []
	scores_et_pca = []
	scores_xgb_pca = []
	j = 0
	for dataset in Datasets:
	    print ("DATASET ",imputations[j])
	    
	    #--------- Training modello con Pearson
	    x_train_pca,x_test_pca = get_PCA(dataset[0],dataset[2])
	    y_train = dataset[1]
	    y_test = dataset[3]
	    #---- ADA
	    AB = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 12))
	    AB.fit(x_train_pca,y_train)
	    AB_predictions = AB.predict(x_test_pca)
	    
	    AB_scores = AB.score(x_test_pca,y_test)
	    rec_AB = recall_score(AB_predictions,y_test,average = "macro")
	    prec_AB = precision_score(AB_predictions,y_test,average = "macro")
	    scores_ada_pca.append([AB_scores,rec_AB,prec_AB,precision_recall_classes(confusion_matrix(y_test,AB_predictions),list(set(y_test)))])
	    print("ADA-BOOST")
	    print("Accuracy", AB_scores)
	    print("Precision",prec_AB)
	    print("Recall",rec_AB)
	    #---- GB
	    GB = GradientBoostingClassifier(n_estimators=500)
	    GB.fit(x_train_pca, y_train)
	    GB_predictions = GB.predict(x_test_pca)
	    GB_scores = GB.score(x_test_pca,y_test)
	    rec_GB = recall_score(GB_predictions,y_test,average = "macro")
	    prec_GB = precision_score(GB_predictions,y_test,average = "macro")
	    print(len(set(GB_predictions)))
	    print(len(set(y_test)))
	    print("TEST")
	    print(set(y_test))
	    print("PREDETTE")
	    print(set(GB_predictions))
	    scores_gb_pca.append([GB_scores,rec_GB,prec_GB,precision_recall_classes(confusion_matrix(y_test,GB_predictions),list(set(y_test)))])
	    print("GRADIENT BOOSTING")
	    print("Accuracy", GB_scores)
	    print("Precision",prec_GB)
	    print("Recall",rec_GB)
	    
	    
	    #---- ET
	    ET = ExtraTreesClassifier(n_estimators=100)
	    ET.fit(x_train_pca,y_train)
	    ET_predictions = ET.predict(x_test_pca)
	    ET_scores = ET.score(x_test_pca,y_test)
	    rec_ET = recall_score(ET_predictions,y_test,average = "macro")
	    prec_ET = precision_score(ET_predictions,y_test,average = "macro")
	    
	    scores_et_pca.append([ET_scores,rec_ET,prec_ET,precision_recall_classes(confusion_matrix(y_test,ET_predictions),list(set(y_test)))])
	    
	    print("EXTRA TREE")
	    print("Accuracy", ET_scores)
	    print("Precision",prec_ET)
	    print("Recall",rec_ET)
	    
	    
	    
	    #---- XGB
	    
	    XGB = XGBClassifier(learning_rate = 0.05, n_estimators=100, max_depth=15)
	    XGB.fit(x_train_pca, y_train)
	    XGB_predictions = XGB.predict(x_test_pca)
	    predictions = [round(value) for value in XGB_predictions]
	    acc_XGB = accuracy_score(y_test, XGB_predictions)
	    rec_XGB = recall_score(y_test, XGB_predictions,average = "macro")
	    prec_XGB = precision_score(y_test, XGB_predictions,average="macro")
	    scores_xgb_pca.append([acc_XGB,rec_XGB,prec_XGB,precision_recall_classes(confusion_matrix(y_test,XGB_predictions),list(set(y_test)))])
	    
	    print("X-GRADIENT BOOSTING")
	    print("Accuracy", acc_XGB)
	    print("Precision",prec_XGB)
	    print("Recall",rec_XGB)
	    j+=1

	return(scores_ada_pca,scores_gb_pca,scores_et_pca,scores_xgb_pca)


