import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import colors as c
import sklearn as sk
import numpy as np


from sklearn.decomposition import PCA,FactorAnalysis
# Manifold
from sklearn.manifold import TSNE,Isomap,LocallyLinearEmbedding,SpectralEmbedding

import seaborn as sns
sns.set(color_codes=True)
from scipy import stats
import math as mt
import scipy as sp
import statsmodels as sm
import seaborn as sb

def fit_pca_tsne_spectral_isomap(rndperm,scaled_matrix_del,scaled_matrix_means,scaled_matrix_medians,scaled_matrix_mode,scaled_matrix_det_regression,scaled_matrix_stoc_regression):
	pca =  PCA(n_components = 3,svd_solver='randomized')
	DelPca = pca.fit_transform(scaled_matrix_del)
	MeansPca = pca.fit_transform(scaled_matrix_means)
	MediansPca = pca.fit_transform(scaled_matrix_medians)
	ModePca = pca.fit_transform(scaled_matrix_mode)
	DetRPca = pca.fit_transform(scaled_matrix_det_regression)
	StocRPca = pca.fit_transform(scaled_matrix_stoc_regression)
	DelPrincipalDf = pd.DataFrame(data = DelPca
             , columns = ['pca-one', 'pca-two','pca-three'])
	MeansPrincipalDf = pd.DataFrame(data = MeansPca
	             , columns = ['pca-one', 'pca-two','pca-three'])
	MediansPrincipalDf = pd.DataFrame(data = MediansPca
	             , columns = ['pca-one', 'pca-two','pca-three'])
	ModePrincipalDf = pd.DataFrame(data = ModePca
	             , columns = ['pca-one', 'pca-two','pca-three'])
	DetRPrincipalDf = pd.DataFrame(data = DetRPca
	             , columns = ['pca-one', 'pca-two','pca-three'])
	StocRPrincipalDf = pd.DataFrame(data = StocRPca
	             , columns = ['pca-one', 'pca-two','pca-three'])


	tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)

	DelTsne = tsne.fit_transform(scaled_matrix_del)
	MeansTsne = tsne.fit_transform(scaled_matrix_means)
	MediansTsne = tsne.fit_transform(scaled_matrix_medians)
	ModeTsne = tsne.fit_transform(scaled_matrix_mode)
	DetRTsne = tsne.fit_transform(scaled_matrix_det_regression)
	StocRTsne = tsne.fit_transform(scaled_matrix_stoc_regression)

	DelPrincipalDf['tsne-one'] = DelTsne[:,0]
	DelPrincipalDf['tsne-two'] = DelTsne[:,1]
	DelPrincipalDf['tsne-three'] = DelTsne[:,2]

	MeansPrincipalDf['tsne-one'] = MeansTsne[:,0]
	MeansPrincipalDf['tsne-two'] = MeansTsne[:,1]
	MeansPrincipalDf['tsne-three'] = MeansTsne[:,2]

	MediansPrincipalDf['tsne-one'] = MediansTsne[:,0]
	MediansPrincipalDf['tsne-two'] = MediansTsne[:,1]
	MediansPrincipalDf['tsne-three'] = MediansTsne[:,2]

	ModePrincipalDf['tsne-one'] = ModeTsne[:,0]
	ModePrincipalDf['tsne-two'] = ModeTsne[:,1]
	ModePrincipalDf['tsne-three'] = ModeTsne[:,2]

	DetRPrincipalDf['tsne-one'] = DetRTsne[:,0]
	DetRPrincipalDf['tsne-two'] = DetRTsne[:,1]
	DetRPrincipalDf['tsne-three'] = DetRTsne[:,2]


	StocRPrincipalDf['tsne-one'] = StocRTsne[:,0]
	StocRPrincipalDf['tsne-two'] = StocRTsne[:,1]
	StocRPrincipalDf['tsne-three'] = StocRTsne[:,2]

	spec = SpectralEmbedding(n_components=3, affinity='nearest_neighbors' ) 
	DelSpec = spec.fit_transform(scaled_matrix_del)
	MeansSpec = spec.fit_transform(scaled_matrix_means)
	MediansSpec = spec.fit_transform(scaled_matrix_medians)
	ModeSpec = spec.fit_transform(scaled_matrix_mode)
	DetRSpec = spec.fit_transform(scaled_matrix_det_regression)
	StocRSpec = spec.fit_transform(scaled_matrix_stoc_regression)

	DelPrincipalDf['spec-one'] = DelSpec[:,0]
	DelPrincipalDf['spec-two'] = DelSpec[:,1]
	DelPrincipalDf['spec-three'] = DelSpec[:,2]

	MeansPrincipalDf['spec-one'] = MeansSpec[:,0]
	MeansPrincipalDf['spec-two'] = MeansSpec[:,1]
	MeansPrincipalDf['spec-three'] = MeansSpec[:,2]

	MediansPrincipalDf['spec-one'] = MediansSpec[:,0]
	MediansPrincipalDf['spec-two'] = MediansSpec[:,1]
	MediansPrincipalDf['spec-three'] = MediansSpec[:,2]

	ModePrincipalDf['spec-one'] = ModeSpec[:,0]
	ModePrincipalDf['spec-two'] = ModeSpec[:,1]
	ModePrincipalDf['spec-three'] = ModeSpec[:,2]

	DetRPrincipalDf['spec-one'] = DetRSpec[:,0]
	DetRPrincipalDf['spec-two'] = DetRSpec[:,1]
	DetRPrincipalDf['spec-three'] = DetRSpec[:,2]


	StocRPrincipalDf['spec-one'] = StocRSpec[:,0]
	StocRPrincipalDf['spec-two'] = StocRSpec[:,1]
	StocRPrincipalDf['spec-three'] = StocRSpec[:,2]


	iso = Isomap(n_neighbors=5, n_components=3)
	DelIso = iso.fit_transform(scaled_matrix_del)
	MeansIso = iso.fit_transform(scaled_matrix_means)
	MediansIso = iso.fit_transform(scaled_matrix_medians)
	ModeIso = iso.fit_transform(scaled_matrix_mode)
	DetRIso = iso.fit_transform(scaled_matrix_det_regression)
	StocRIso = iso.fit_transform(scaled_matrix_stoc_regression)


	DelPrincipalDf['iso-one'] = DelIso[:,0]
	DelPrincipalDf['iso-two'] = DelIso[:,1]
	DelPrincipalDf['iso-three'] = DelIso[:,2]

	MeansPrincipalDf['iso-one'] = MeansIso[:,0]
	MeansPrincipalDf['iso-two'] = MeansIso[:,1]
	MeansPrincipalDf['iso-three'] = MeansIso[:,2]

	MediansPrincipalDf['iso-one'] = MediansIso[:,0]
	MediansPrincipalDf['iso-two'] = MediansIso[:,1]
	MediansPrincipalDf['iso-three'] = MediansIso[:,2]

	ModePrincipalDf['iso-one'] = ModeIso[:,0]
	ModePrincipalDf['iso-two'] = ModeIso[:,1]
	ModePrincipalDf['iso-three'] = ModeIso[:,2]

	DetRPrincipalDf['iso-one'] = DetRIso[:,0]
	DetRPrincipalDf['iso-two'] = DetRIso[:,1]
	DetRPrincipalDf['iso-three'] = DetRIso[:,2]


	StocRPrincipalDf['iso-one'] = StocRIso[:,0]
	StocRPrincipalDf['iso-two'] = StocRIso[:,1]
	StocRPrincipalDf['iso-three'] = StocRIso[:,2]

	return(DelPrincipalDf,MeansPrincipalDf,MediansPrincipalDf,ModePrincipalDf,DetRPrincipalDf,StocRPrincipalDf)


def PlotProjection(lista,datasets,data,rndperm):
    j= 1
    k = 0
    for i in lista:
        
        plt.figure(figsize=(16,7))
        plt.subplots_adjust(hspace=0.4)
        ax1 = plt.subplot(2, 2, 1)
        
        sns.scatterplot(
            
            x="pca-one", y="pca-two",
            hue=data['t'],
            palette=sns.color_palette("hls", 13),
            data=i,

            #legend="full",
            alpha=0.7,
            ax=ax1
        ).set_title('PCA '+ datasets[k])
        ax1.get_legend().remove()
        ax2 = plt.subplot(2, 2, 2)
        
        sns.scatterplot(
            
            x="tsne-one", y="tsne-two",
            hue=data['t'],
            palette=sns.color_palette("hls", 13),
            data=i,
            #legend="full",
            alpha=0.7,
            ax=ax2
        ).set_title('TSNE '+ datasets[k])
        ax2.get_legend().remove()
        
        ax3 = plt.subplot(2,2,3)
        sns.scatterplot(
            
            x="spec-one", y="spec-two",
            hue=data['t'],
            palette=sns.color_palette("hls", 13),
            data=i,

            #legend="full",
            alpha=0.7,
            ax=ax3
        ).set_title('Spectral '+ datasets[k])
        ax3.get_legend().remove()
        
        ax4 = plt.subplot(2,2,4)
        sns.scatterplot(

            x="iso-one", y="iso-two",
            hue=data['t'],
            palette=sns.color_palette("hls", 13),
            data=i,

            #legend="full",
            alpha=0.7,
            ax=ax4
        ).set_title('Isomap '+ datasets[k])
        ax4.get_legend().remove()
        plt.show()
        
        ax5 = plt.figure(figsize=(16,10)).gca(projection='3d')
        ax5.set_title("3D Plotting PCA")
        ax5.scatter(
            xs=i.loc[rndperm,:]["pca-one"], 
            ys=i.loc[rndperm,:]["pca-two"], 
            zs=i.loc[rndperm,:]["pca-three"], 
            c=data.loc[rndperm,:]["t"], 
            cmap='tab10'

        )
        ax5.set_xlabel('pca-one')
        ax5.set_ylabel('pca-two')
        ax5.set_zlabel('pca-three')
        
        ax6 = plt.figure(figsize=(16,10)).gca(projection='3d')
        ax6.set_title("3D Plotting TSNE")
        ax6.scatter(
            xs=i.loc[rndperm,:]["tsne-one"], 
            ys=i.loc[rndperm,:]["tsne-two"], 
            zs=i.loc[rndperm,:]["tsne-three"], 
            c=data.loc[rndperm,:]["t"], 
            cmap='tab10'

        )
        ax6.set_xlabel('tsne-one')
        ax6.set_ylabel('tsne-two')
        ax6.set_zlabel('tsne-three')
        
        ax7 = plt.figure(figsize=(16,10)).gca(projection='3d')
        ax7.set_title("3D Plotting Spectral")
        ax7.scatter(
            xs=i.loc[rndperm,:]["spec-one"], 
            ys=i.loc[rndperm,:]["spec-two"], 
            zs=i.loc[rndperm,:]["spec-three"], 
            c=data.loc[rndperm,:]["t"], 
            cmap='tab10'

        )
        ax7.set_xlabel('spec-one')
        ax7.set_ylabel('spec-two')
        ax7.set_zlabel('spec-three')
        
        ax8 = plt.figure(figsize=(16,10)).gca(projection='3d')
        ax8.set_title("3D Plotting Isomap")
        ax8.scatter(
            xs=i.loc[rndperm,:]["iso-one"], 
            ys=i.loc[rndperm,:]["iso-two"], 
            zs=i.loc[rndperm,:]["iso-three"], 
            c=data.loc[rndperm,:]["t"], 
            cmap='tab10'

        )
        ax8.set_xlabel('iso-one')
        ax8.set_ylabel('iso-two')
        ax8.set_zlabel('iso-three')
        k+=1
        plt.show()

