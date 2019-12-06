import numpy as np

#def delete_column(matrix,col):
#	return (np.delete(matrix,col,axis =1))

def delete_row(matrix,row,n,m):

	matrice = []
	for i in range(0,n):
		if(i!= row):
			riga = []
			for j in range(0,m):
				riga.append(matrix[i,j])
			matrice.append(riga)
	return(np.array(matrice))




def delete_column(matrix,col,n,m):
	
	matrice = []
	for i in range(0,n):
		riga = []
		for j in range(0,m):
			if(j != col):
				riga.append(matrix[i,j])
		matrice.append(riga)
	return(np.array(matrice))