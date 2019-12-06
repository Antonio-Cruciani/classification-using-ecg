import numpy as np
def precision_recall_classes(confusion_matrix,target):
    precisions=[]
    recalls = []
    print(confusion_matrix)
    print("SHAPE",confusion_matrix.shape)
    for i in range(0,confusion_matrix.shape[0]):
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
        recalls.append(recalls)

    single_precision_recall={
        "Precision":precisions,
        "Recall":recalls
    }

    precision_recall = pd.DataFrame(single_precision_recall,index=target)
