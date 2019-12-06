from sklearn.metrics import precision_recall_fscore_support,recall_score,precision_score
from sklearn.feature_selection import RFE, SelectKBest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,average_precision_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def train_logR_k_fold_pear(matrix,target, nsplits= 10,penalty="l2",C=1,multi_class = "ovr"):
    scores = []
    confusion =[]
    features = []
    parameters = {
        "C":C,
        "Penalty":penalty,
        "Average":[],
        "Scores":[],
        "Features":[],
        "Macro":[],
        "Micro":[],
        "Weighted":[]
    }
   
    if(penalty =="elasticnet" or penalty=="l1"):
        solver = 'saga'
    else:
        solver = 'newton-cg'
    if(penalty == 'elasticnet'):
        
        logReg = LogisticRegression(penalty = penalty,C=C,multi_class= multi_class,solver = solver,l1_ratio =0.5)
    else:
        logReg = LogisticRegression(penalty = penalty,C=C,multi_class= multi_class,solver = solver)
        
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
        logReg.fit(x_train_fs, y_train)

        #--------------- TESTING -------------------------------
        # Getting the scores of the model on the test set
        logReg_Predictions = logReg.predict(X_test[:,indexes])
        # getting accuracy
        scores.append(logReg.score(X_test[:,indexes], y_test))
        # Macro
        parameters["Macro"].append(precision_recall_fscore_support(y_test, logReg_Predictions, average='macro'))
        # Micro
        parameters["Micro"].append(precision_recall_fscore_support(y_test, logReg_Predictions, average='micro'))
        # Weighted
        parameters["Weighted"].append(precision_recall_fscore_support(y_test, logReg_Predictions, average='weighted'))
        
        
        parameters["Features"].append(indexes)
       
        # getting confusion matrix
        confusion.append(confusion_matrix(y_test,logReg_Predictions))
    parameters["Scores"].append(scores)
    parameters["Average"] = np.average(scores)
    return (scores,confusion,parameters)

def train_logR_k_fold_pca(matrix,target, nsplits= 10,penalty="l2",C=1,multi_class = "ovr"):
    scores = []
    confusion =[]
    features = []
    parameters = {
        "C":C,
        "Penalty":penalty,
        "Average":[],
        "Scores":[],
        "Features":[],
        "PCA_Param":[],
        "Macro":[],
        "Micro":[],
        "Weighted":[]
    }
   
    if(penalty =="elasticnet" or penalty=="l1"):
        solver = 'saga'
    else:
        solver = 'newton-cg'
    if(penalty == 'elasticnet'):
        
        logReg = LogisticRegression(penalty = penalty,C=C,multi_class= multi_class,solver = solver,l1_ratio =0.5)
    else:
        logReg = LogisticRegression(penalty = penalty,C=C,multi_class= multi_class,solver = solver)
        
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
        #print("Somma explained variance: ",sum(explained_variance))
            
        
        parameters["PCA_Param"].append(pca.get_params())
        # --------------- TRAINING ------------------------------
        # Training the model

        logReg.fit(X_train_pca, y_train)
        #--------------- TESTING -------------------------------
        # Getting the scores of the model on the test set
        logReg_Predictions = logReg.predict(X_test_pca)
        # getting accuracy
        scores.append(logReg.score(X_test_pca, y_test))
        # Macro
        parameters["Macro"].append(precision_recall_fscore_support(y_test, logReg_Predictions, average='macro'))
        # Micro
        parameters["Micro"].append(precision_recall_fscore_support(y_test, logReg_Predictions, average='micro'))
        # Weighted
        parameters["Weighted"].append(precision_recall_fscore_support(y_test, logReg_Predictions, average='weighted'))
        
        #parameters["Features"].append(indexes)
       
        # getting confusion matrix
        #confusion.append(confusion_matrix(y_test,svc_predictions))
    parameters["Scores"].append(scores)
    parameters["Average"] = np.average(scores)
    return (scores,confusion,parameters)

def train_logR_k_fold_rfe(matrix,target, nsplits= 10,penalty="l2",C=1,multi_class = "ovr",nfeatures=15):
    scores = []
    confusion =[]
    features = []
    parameters = {
        "C":C,
        "Penalty":penalty,
        "Average":[],
        "Scores":[],
        "Features":[],
        "Macro":[],
        "Micro":[],
        "Weighted":[]
    }
   
    if(penalty =="elasticnet" or penalty=="l1"):
        solver = 'saga'
    else:
        solver = 'newton-cg'
    if(penalty == 'elasticnet'):
        
        logReg = LogisticRegression(penalty = penalty,C=C,multi_class= multi_class,solver = solver,l1_ratio =0.5)
    else:
        logReg = LogisticRegression(penalty = penalty,C=C,multi_class= multi_class,solver = solver)
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
        logReg.fit(x_train_fs, y_train)

        #--------------- TESTING -------------------------------
        # Getting the scores of the model on the test set
        logReg_Predictions = logReg.predict(X_test[:,indexes])
        # getting accuracy
        scores.append(logReg.score(X_test[:,indexes], y_test))
        # Macro
        parameters["Macro"].append(precision_recall_fscore_support(y_test, logReg_Predictions, average='macro'))
        # Micro
        parameters["Micro"].append(precision_recall_fscore_support(y_test, logReg_Predictions, average='micro'))
        # Weighted
        parameters["Weighted"].append(precision_recall_fscore_support(y_test, logReg_Predictions, average='weighted'))
        
        
        parameters["Features"].append(indexes)
        
        # getting confusion matrix
        confusion.append(confusion_matrix(y_test,logReg_Predictions))
    parameters["Scores"].append(scores)
    parameters["Average"] = np.average(scores)
    return (scores,confusion,parameters)


def train_logR_k_fold(matrix,target, nsplits= 10,penalty="l2",C=1,multi_class = "ovr"):
    scores = []
    confusion =[]
    features = []
    parameters = {
    "C":C,
    "Penalty":penalty,
    "Average":[],
    "Scores":[],
    "Features":[],
    "Macro":[],
    "Micro":[],
    "Weighted":[]
    }
    if(penalty =="elasticnet" or penalty=="l1"):
        solver = 'saga'
    else:
        solver = 'newton-cg'
    if(penalty == 'elasticnet'):

        logReg = LogisticRegression(penalty = penalty,C=C,multi_class= multi_class,solver = solver,l1_ratio =0.5)
    else:
        logReg = LogisticRegression(penalty = penalty,C=C,multi_class= multi_class,solver = solver)
    cv = KFold(n_splits=nsplits, random_state=42, shuffle=False)
    for train_index, test_index in cv.split(matrix):
        #print("Train Index: ", train_index, "\n")
        #print("Test Index: ", test_index)
        X_train, X_test, y_train, y_test = matrix[train_index], matrix[test_index], target[train_index], target[test_index]

        # --------------- TRAINING ------------------------------
       # Training the model
        logReg.fit(X_train, y_train)

        #--------------- TESTING -------------------------------
        # Getting the scores of the model on the test set
        logReg_Predictions = logReg.predict(X_test)
        # getting accuracy
        scores.append(logReg.score(X_test, y_test))
        # Macro
        parameters["Macro"].append(precision_recall_fscore_support(y_test, logReg_Predictions, average='macro'))
        # Micro
        parameters["Micro"].append(precision_recall_fscore_support(y_test, logReg_Predictions, average='micro'))
        # Weighted
        parameters["Weighted"].append(precision_recall_fscore_support(y_test, logReg_Predictions, average='weighted'))



        # getting confusion matrix
        confusion.append(confusion_matrix(y_test,logReg_Predictions))
    parameters["Scores"].append(scores)
    parameters["Average"] = np.average(scores)
    return (scores,confusion,parameters)

def train_logistic_regression(Datasets,C,penalty,imputations,path):
    best_scores_pear = []
    summary_pear = []
    otherresults_pear = []

    best_scores_nofs = []
    summary_nofs = []
    otherresults_nofs = []

    best_scores_rfe = []
    summary_rfe = []
    otherresults_rfe = []

    best_scores_pca = []
    summary_pca = []
    otherresults_pca = []


    u = 0
    for dataset in Datasets:
        print("\n\n ----------- INIZIO DATASET :",imputations[u])
        esecuzioni_pear = []
        esecuzioni_rfe = []
        esecuzioni_nofs = []
        esecuzioni_pca = []

        for i in penalty:
            for j in C:

                scores_pear,confusion_pear,executions_pear = train_logR_k_fold_pear(dataset[0],dataset[1],penalty=i,C=j)
                esecuzioni_pear.append(executions_pear)
                otherresults_pear.append([np.average(scores_pear),confusion_pear,executions_pear])


                scores_rfe,confusion_rfe,executions_rfe = train_logR_k_fold_rfe(dataset[0],dataset[1],penalty=i,C=j)
                esecuzioni_rfe.append(executions_rfe)
                otherresults_rfe.append([np.average(scores_rfe),confusion_rfe,executions_rfe])

                scores_pca,confusion_pca,executions_pca = train_logR_k_fold_pca(dataset[0],dataset[1],penalty=i,C=j)
                esecuzioni_pca.append(executions_pca)
                otherresults_pca.append([np.average(scores_pca),confusion_pca,executions_pca])

                scores_nofs,confusion_nofs,executions_nofs = train_logR_k_fold(dataset[0],dataset[1],penalty=i,C=j)
                esecuzioni_nofs.append(executions_nofs)
                otherresults_nofs.append([np.average(scores_nofs),confusion_nofs,executions_nofs])



                print("FINE ESECUZIONE Logistic Regression ")
                print("Penalty ",i)
                print("C",j)

                print(" RISULTATI ACCURACY ")

                print("-----PEAR-------")
                print(np.average(scores_pear))
                print("-----RFE-------")
                print(np.average(scores_rfe))
                print("-----PCA-------")
                print(np.average(scores_pca))
                print("-----NOFS-------")
                print(np.average(scores_nofs))



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



    np.save(path+"/otherresults_pear.npy", otherresults_pear)
    np.save(path+"/otherresults_rfe.npy", otherresults_rfe)
    np.save(path+"/otherresults_pca.npy", otherresults_pca)
    np.save(path+"/otherresults_nofs.npy", otherresults_nofs)



    np.save(path+"/best_scores_pear.npy", best_scores_pear)
    np.save(path+"/best_scores_rfe.npy", best_scores_rfe)
    np.save(path+"/best_scores_pca.npy", best_scores_pca)
    np.save(path+"/best_scores_nofs.npy", best_scores_nofs)


    




def test_lreg(Datasets,best_scores,imputations,title,fs=True,pca=False):
    u= 0 
    test_accuracy_list = []
    test_precision_list = []
    test_recall_list = []
    multi_class = "ovr"
    for dataset in Datasets:
        #print(phases[u])
        # Getting Parameters for the specific dataset
        C = best_scores[u][2]['C']
        penalty = best_scores[u][2]['Penalty']
        
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
        
        if(penalty =="elasticnet" or penalty=="l1"):
            solver = 'saga'
        else:
            solver = 'newton-cg'
        if(penalty == 'elasticnet'):

            logReg = LogisticRegression(penalty = penalty,C=C,multi_class= multi_class,solver = solver,l1_ratio =0.5)
        else:
            logReg = LogisticRegression(penalty = penalty,C=C,multi_class= multi_class,solver = solver)
        
        logReg.fit(x_train, dataset[1])
        # Getting accuracy of the model
        logReg_predictions = logReg.predict(x_test)

    

        # getting accuracy
        test_recall_list.append(recall_score(logReg_predictions,dataset[3],average = "macro"))
        test_precision_list.append(precision_score(logReg_predictions,dataset[3],average = "macro"))
        test_accuracy_list.append(logReg.score(x_test, dataset[3]))
        #print (SVC_MODEL.score(dataset[2][:,best_feats], dataset[3]))
        #print(svc_predictions)
        print("RECALL "+imputations[u],recall_score(logReg_predictions,dataset[3],average = "macro"))
        print("PRECISION "+imputations[u],precision_score(logReg_predictions,dataset[3],average = "macro"))
        
        #precision_recall_classes(confusion_matrix(dataset[3],svc_predictions))
        
        
        
        
        
        u+=1
    
    
    
    logRego_scores_df = pd.DataFrame(test_accuracy_list,columns=["Accuracy"],index=imputations)
    fig, ax = plt.subplots(figsize=(15,7))
    logRego_scores_df.plot(kind="bar",ax=ax)
    ax.set_title("Plot Test Accuracy of the Logisti Regression Model on the different kind of datasets with "+title)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Datasets')
    fig.tight_layout()
    plt.show()
    print(logRego_scores_df)
    return(test_accuracy_list,test_recall_list,test_precision_list)



