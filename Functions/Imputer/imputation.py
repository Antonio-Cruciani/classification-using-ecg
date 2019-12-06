import numpy as np
import scipy as sp

# Function that returns a list containing indexes of coloumns with ? values
def get_indices_of_missing_param(data):
    n,m = np.shape((data))
    l = []
    for i in range(0,m):
        if("?" in data[:,i]):
             l.append(i)
    return l
# Fun unzione che ritorna una lista di valori medi di colonne di una matrice
def get_means(indexes,data,character = "?"):
    output_l =[]
    output_c = []
    n,m = np.shape((data))
    for j in indexes:
        l =[0 for i in range(0,16)]
        c = [0 for i in range(0,16)]
      
        for i in range(0,n):
            if data[i,j] != character:
                l[data[i,-1]-1] += float(data[i,j])
                c[data[i,-1]-1] += 1
        output_l.append(l)
        output_c.append(c)
    
    return output_l,output_c
# Funzione che cerca le features con "missing values" e calcola i valori medi di tali features
def get_means_list(X):
    means = []
    m_param = get_indices_of_missing_param(X)
    k,h = get_means(m_param,X,"?")
    p = 0
    for i in k:
        l = [0 for j in range(0,16)]
    
        for j in range(0,16):
            if i[j]!=0:
                l[j] = i[j]/h[p][j]
            else:
                l[j]=0
        p+=1
        means.append(l)
    return means,m_param

# Funzione che sostituisce le features mancanti con i valori calcolati
def replacing_missing(X,params,indexes):
    n,m = np.shape((X))
    K = X
    k = 0 
    for i in indexes:
        for j in range(0,n):
            if (K[j,i]=="?"):
                K[j,i] = params[k][K[j,-1]-1] 
        k=+1
    return K

#funzione che calcola i mediani 
def get_median(X,index):
    n,m = np.shape((X))
    mediani = [0.0 for i in range(0,16)]
    liste = [[] for i in range(0,16)]
    sorted_list = []
    
    for i in range(0,n):
        if(X[i,index]!='?'):

            liste[X[i,-1]-1].append(X[i,index])
    h=0
    
        
    for i in liste:
        k = []
        for j in i:
            k.append(float(j))
        
        sorted_list = sorted(k)
        if sorted_list != []:
            a = np.array(sorted_list)
            #print a
            
            mediani[h] = np.median(a)
        else: 
            mediani[h]=0
        
        h+=1
        
    return mediani
#funzione che calcola i mediani per ogni classe per ogni indice dei valori mancanti
def get_median_list(X):
    median_list = []
    m_param = get_indices_of_missing_param(X)
    
    for i in m_param:
         median_list.append(get_median(X,i))
    
    return median_list,m_param
    
def get_mode(X,index):
    n,m = np.shape((X))
    mode = [0.0 for i in range(0,16)]
    liste = [[] for i in range(0,16)]
    sorted_list = []
    
    for i in range(0,n):
        if(X[i,index]!='?'):

            liste[X[i,-1]-1].append(X[i,index])
    h=0
    
    for i in liste:
        count_list = []
        for j in i:
            count_list.append(i.count(j))
        if(count_list != [ ]):
            mode.append(max(count_list))
        else:
            mode.append(0)
        
    return mode
    
def get_mode_list(X):
    mode_list = []
    m_param = get_indices_of_missing_param(X)
    
    for i in m_param: 
        mode_list.append(get_mode(X,i))

    
    return mode_list,m_param
    
    
def get_lists_of_values_for_each_class(index,matrix):
    lista = [[] for i in range(1,17)]
    j = 0
    n,m = np.shape((matrix))

    for i in range(1,16):
        #print matrix[j,m-1],i
        while(j<n and matrix[j,m-1]==i):
            if matrix[j,index]!='?':
                lista[i].append(float(matrix[j,index]))
            else:
                lista[i].append('NaN')
            j+=1
    return lista
    

    
    
    
  