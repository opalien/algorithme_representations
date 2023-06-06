import sklearn.datasets
from scipy.sparse.linalg import svds 
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import ortho_group
from scipy.linalg import subspace_angles    
import matplotlib as mpl  
import PCA

from sklearn.metrics import mean_squared_error
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['legend.fontsize'] = 17
plt.rcParams['xtick.labelsize'] = 19  
plt.rcParams['ytick.labelsize'] = 19 



def distance_angle_vector(vecteurs_propres1,vecteurs_propres2):
    '''
    Cette fonction calcule la distance angulaire entre deux ensembles de vecteurs propres.
    Arguments :
        vecteurs_propres1 : Premier ensemble de vecteurs propres.
        vecteurs_propres2 : Deuxième ensemble de vecteurs propres.'''
    distance_vecteurs=[]
    i=0
    vecteurs_propres2=np.array(vecteurs_propres2)
    vecteurs_propres1=np.array(vecteurs_propres1)
    for i in range(vecteurs_propres2.shape[1]):
        if np.dot(vecteurs_propres1[:,i], vecteurs_propres2[:,i]) < 0:
            vecteurs_propres2[:,i] *= -1
        
    for vect in vecteurs_propres1:
        angle_minimal=np.pi/2
        
        indice_minimal=0
        for j in range(len(vecteurs_propres2)):
            angle=np.arccos(np.dot(vect,vecteurs_propres2[j])/(np.linalg.norm(vect)*np.linalg.norm(vecteurs_propres2[j])))
            if(angle<=np.pi/2):
                if(angle<angle_minimal):
                    angle_minimal=angle
                    indice_minimal=j
            else:
                if(np.pi-angle<angle_minimal):
                    angle_minimal=np.pi-angle
                    indice_minimal=j
        distance_vecteurs.append(angle_minimal)
        vecteurs_propres2 = np.delete(vecteurs_propres2, indice_minimal, axis=0)  
    return distance_vecteurs

def mse_projected_real_data(X,X_projected):
    '''
    Cette fonction calcule l'erreur quadratique moyenne entre les données réelles et les données projetées.
    Arguments :
        X : Données réelles.
        X_projected : Données projetées.
    '''
    return np.mean((X-X_projected))**2



def generate_eigen_vectors_values(dimension, strong_variance_number):
    '''
    Cette fonction génère des valeurs propres et des vecteurs propres pour une dimension donnée.
    Arguments :
        dimension : Dimension de l'espace des données.
        strong_variance_number : Nombre de valeurs propres avec une forte variance.
    '''
    weak_variance_number = dimension - strong_variance_number
    eigen_vectors = ortho_group.rvs(dim=dimension)
    strong_eigen_values = 0.9
    weak_eigen_values = 0.1
    eigen_values = np.zeros(dimension)
    previous_eigen_values = strong_eigen_values
    for i in range(strong_variance_number):
        eigen_values[i] = previous_eigen_values /(1.5)
        previous_eigen_values = eigen_values[i]
    
    previous_eigen_values = weak_eigen_values
    for j in range(weak_variance_number):
        eigen_values[j+strong_variance_number] = previous_eigen_values /2
        previous_eigen_values = eigen_values[j+strong_variance_number]
    
    return eigen_values, eigen_vectors



def generate_data(n_samples,eigen_values, eigen_vectors):
    '''
    Cette fonction génère des données à l'aide de valeurs propres et d'une matrice orthogonale aléatoire.

    Arguments :
        n_samples : Nombre d'échantillons à générer.
        eigen_values : Valeurs propres utilisées pour générer les données.
        eigen_vectors : Vecteurs propres utilisés pour générer les données.
    '''
    eigen_values=np.array(eigen_values)
    cov=np.dot(eigen_vectors,np.dot(np.diag(eigen_values),eigen_vectors.T))
    mean = np.zeros(eigen_values.shape[0])
    data = np.random.multivariate_normal(mean, cov, size=n_samples).T
    return data

def fill_df_examples(df, dimension, exempleMax, eigen_values, eigen_vectors):
    '''
    Cette fonction remplit un DataFrame avec des informations sur la PCA appliquée à des données générées aléatoirement. 
    Les informations incluent les valeurs propres moyennes, les vecteurs propres moyens, les erreurs des valeurs propres, 
    les erreurs des vecteurs propres, l'erreur quadratique moyenne de reconstruction pour les données d'entraînement et de test, 
    le nombre de composantes principales conservées, le nombre d'exemples et la variance expliquée.

    Arguments :
        df : DataFrame à remplir.
        dimension : Dimension de l'espace des données.
        exempleMax : Nombre maximal d'exemples à générer.
        eigen_values : Valeurs propres utilisées pour générer les données.
        eigen_vectors : Vecteurs propres utilisés pour générer les données.
    '''
    pca = PCA.PCA()
    nb_exemples = 5
    X_test=generate_data(5000,eigen_values,eigen_vectors)
    while nb_exemples < exempleMax:
        distance_angle_eigen_vectors = 0
        distance_angle_each_eigen_vectors = []
        error_each_eigen_values = []
        mse_projected_train = []
        mse_projected_test = []
        eigen_values_array = np.zeros(dimension)
        eigen_vectors_array = np.zeros((dimension, dimension))
        explained_variance = np.zeros(dimension)
        nb_tirages = 1000
        
        for j in range(nb_tirages):
            X = generate_data(nb_exemples, eigen_values, eigen_vectors)
            pca = PCA.PCA()
            pca.fit(X)
    
            eigen_values_array += pca.get_eigen_values()
            eigen_vectors_array += pca.get_eigen_vectors()
            distance_angle_each_eigen_vectors.append(distance_angle_vector(eigen_vectors, pca.get_eigen_vectors()))
            error_each_eigen_values.append(eigen_values - pca.get_eigen_values())
            explained_variance += pca.explained_variance() * 100
            mse_projected_train.append( mean_squared_error(X.T, pca.reconstruct_projected_data(X.T, dimension//2)))
            mse_projected_test.append(mean_squared_error(X_test.T, pca.reconstruct_projected_data(X_test.T,dimension//2 )))

        explained_variance = explained_variance / nb_tirages
        distance_angle_each_eigen_vectors =np.array(distance_angle_each_eigen_vectors)
        error_each_eigen_values=np.array(error_each_eigen_values)
        error_each_eigen_values_inter=[]
        error_each_eigen_vectors_inter=[]
        for i in range(dimension):
            error_each_eigen_values_inter.append(np.mean(error_each_eigen_values[:,i])**2)
            error_each_eigen_vectors_inter.append(np.mean(distance_angle_each_eigen_vectors[:,i])**2)
            
            
        error_each_eigen_values=np.array(error_each_eigen_values_inter)
        distance_angle_each_eigen_vectors=np.array(error_each_eigen_vectors_inter)
        
        mse_projected_test =np.mean(mse_projected_test)
        mse_projected_train =np.mean(mse_projected_train)
        eigen_values_array /= nb_tirages
        eigen_vectors_array /= nb_tirages
        distance_angle_eigen_vectors = np.mean(distance_angle_each_eigen_vectors)
        error_eigen_values = np.mean(error_each_eigen_values)
        
        new_row = {'eigen values mean': eigen_values_array, 'eigen vectors mean': eigen_vectors_array, 'eigenvalues error': error_eigen_values,
                   'eigenvectors error': distance_angle_eigen_vectors, 'error by eigenvalue': error_each_eigen_values, 'error by eigenvector': distance_angle_each_eigen_vectors,
                   'mse reconstruction train': mse_projected_train, 'mse reconstruction test': mse_projected_test,
                   'nb pc kept': dimension // 2, 'number of examples': nb_exemples, 'explained variance': explained_variance}
        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)
        nb_exemples *= 2
        print(nb_exemples)
    return df



def generate_eigen_vectors_values_2(dimension, strong_variance_number):
    '''
    Cette fonction génère des valeurs propres et des vecteurs propres pour une dimension donnée.
    Arguments :
        dimension : Dimension de l'espace des données.
        strong_variance_number : Nombre de valeurs propres avec une forte variance.
    '''
    weak_variance_number = dimension - strong_variance_number
    eigen_vectors = ortho_group.rvs(dim=dimension)
    strong_eigen_values = 0.9
    weak_eigen_values = 0.1
    eigen_values = np.zeros(dimension)
    for i in range(strong_variance_number):
        eigen_values[i] = strong_eigen_values/strong_variance_number
    for j in range(weak_variance_number):
        eigen_values[j+strong_variance_number] = weak_eigen_values/weak_variance_number
    return eigen_values[::-1], eigen_vectors


def fill_df_dimension(df, dimMax, nb_exemples):
    '''
    Cette fonction remplit un DataFrame avec des informations sur la PCA appliquée à des données générées aléatoirement. 
    Les informations incluent l'erreur quadratique moyenne de reconstruction pour les données d'entraînement et de test et la dimension des données.
    Arguments :
        df : DataFrame à remplir.
        dimMax : Dimension maximale des données.
        nb_exemples : Nombre d'échantillons à générer.
    '''
    data = []
    dimension = 10
    while dimension < dimMax:
        mse_projected_train = 0
        mse_projected_test = 0
        
        for j in range(100):
            pca = PCA.PCA()
            eigen_values, eigen_vectors = generate_eigen_vectors_values_2(dimension, 6)
            X = generate_data(nb_exemples, eigen_values, eigen_vectors)
            X_train, X_test = train_test_split(X, test_size=0.2, random_state=j)
            X_train = X_train.T
            X_test = X_test.T
            pca.fit(X)
            pca.fit(X_train)
            mse_projected_train = mse_projected_real_data(X_train, pca.reconstruct_projected_data(X_train, 6)).T
            mse_projected_test = mse_projected_real_data(X_test, pca.reconstruct_projected_data(X_test, 6)).T
            
            row = [mse_projected_train, mse_projected_test, dimension]
            data.append(row)

        dimension *= 2
        print(dimension)

    np_data = np.array(data)
    new_df = pd.DataFrame(np_data, columns=['mse reconstruction train', 'mse reconstruction test', 'dimension'])
    df = pd.concat([df, new_df], ignore_index=True)
    
    return df