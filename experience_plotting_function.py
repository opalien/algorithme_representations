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
import experience_pca_functions
import PCA
from experience_pca_functions import *
import seaborn as sns

def affichage_plan_orthogonal():
    '''Cette fonction génère un graphique 3D montrant deux plans orthogonaux: un plan horizontal (Z1, rouge) et
    un plan vertical (X2, bleu). Les plans sont créés à partir d'une grille de points générée par la fonction numpy.meshgrid.
    La fonction ne prend pas d'arguments en entrée et ne renvoie aucune valeur. 
    Elle enregistre une image du graphique sous le nom "images/plan_ortho".'''
    x = np.linspace(-10, 10, 20)
    y = np.linspace(-10, 10, 20)
    X, Y = np.meshgrid(x, y)

    # Plan horizontal
    Z1 = np.zeros_like(X)

    # Plan vertical
    X2 = np.zeros_like(X)

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z1, color='r', alpha=0.5, rstride=100, cstride=100)  # Plan horizontal
    ax.plot_surface(X2, Y, X, color='b', alpha=0.5, rstride=100, cstride=100)  # Plan vertical

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig("images/plan_ortho")
    plt.show()
    
def affichage_plan_proche():
    '''Cette fonction génère un graphique 3D illustrant deux plans: un plan à z=0 (Z1, rouge) 
    et un plan qui croise le premier (Z2, bleu). Les plans sont créés à partir d'une grille de points générée par la fonction numpy.meshgrid.
    La fonction ne prend pas d'arguments en entrée et ne renvoie aucune valeur. 
    Elle enregistre une image du graphique sous le nom "images/plans_proches".'''
    x = np.linspace(-10, 10, 20)
    y = np.linspace(-10, 10, 20)
    X, Y = np.meshgrid(x, y)

    # Plan à z=0
    Z1 = np.zeros_like(X)

    # Plan qui croise le premier
    Z2 = ((X+Y)/20) * 6 - 3

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z1, color='r', alpha=0.5, rstride=100, cstride=100)  # Plan à z=0
    ax.plot_surface(X, Y, Z2, color='b', alpha=0.5, rstride=100, cstride=100)  # Plan qui croise le premier

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_zlim(-30, 30)
    plt.savefig("images/plans_proches")
    plt.show()
    
def plot_by_eigen_value_biais(df):
    '''Cette fonction génère un graphique montrant le biais au carré pour les valeurs propres en fonction du nombre d'exemples.
    Les courbes sont tracées avec différentes couleurs pour distinguer les différentes valeurs propres.
    La fonction prend en entrée un DataFrame contenant les erreurs par valeur propre et le nombre d'exemples. 
    Elle enregistre une image du graphique sous le nom "images/biais_valeurs_propres.png".
    Arguments : df : DataFrame qui contient le biais par valeur propre et le nombre d'exemples.'''
    error_data = df['error by eigenvalue'].to_numpy()
    error_data = np.vstack(error_data)
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta']
    for i in range(5):
        plt.plot(df['number of examples'], error_data[:, i], color=colors[i], label='CP'+str(i+1))
    #for i in range(len(error_data[0])-1, len(error_data[0])-2, -1):
    #plt.plot(df['number of examples'], error_data[:, i], color=colors[len(colors)-1], label=str(i+1) +'ième valeur propre')
    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Nombre d'exemples")
    plt.ylabel("Biais au carré")
    plt.ylim(0)
    plt.legend()
    plt.savefig('images/biais_valeurs_propres.png',bbox_inches='tight')
    plt.show()
    

def plot_by_eigen_vector_biais(df):
    '''Cette fonction trace un graphique du biais par vecteur propre en fonction du nombre d'exemples. 
    Pour chaque composante principale (CP), elle trace une courbe distincte.
    Elle utilise une échelle logarithmique pour les deux axes et sauvegarde le graphique sous forme d'image PNG.

    Arguments : df : DataFrame qui contient le biais par vecteur propre et le nombre d'exemples.'''
    error_data = df['error by eigenvector'].to_numpy()
    error_data = np.vstack(error_data)
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta']
    for i in range(5):
        plt.plot(df['number of examples'], error_data[:, i], color=colors[i], label='CP'+str(i+1))
    #for i in range(len(error_data[0])-1, len(error_data[0])-2, -1):
    #plt.plot(df['number of examples'], error_data[:, i], color=colors[len(colors)-1], label=str(i+1)+'ième vecteur propre')
        
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Nombre d\'exemples')
    plt.ylabel("Biais au carré")
    plt.ylim(0)
    plt.legend()
    plt.savefig('images/biais_vecteurs_propres.png',bbox_inches='tight')
    plt.show()


    




def generate_plots(df, eigen_values=None, eigen_vectors=None):
    '''
    Génère des graphiques pour analyser les erreurs de valeurs propres, les erreurs de vecteurs propres
    et les erreurs de reconstruction à l'aide de la méthode des composantes principales.
    Arguments : df : DataFrame qui contient les données à afficher.
                eigen_values : array variance théorique
                eigen_vectors : array direction théorique
    '''
    intervalle_confiance = 1.96 * np.std(df['eigenvalues error'].to_numpy(), axis=0) / np.sqrt(df['number of examples'].to_numpy().astype(float))
    plt.errorbar(df['number of examples'], df['eigenvalues error'], yerr=intervalle_confiance, color='red', label='erreurs valeurs propre')
    plt.xscale('log')
    plt.xlabel("Nombre d'exemples")
    plt.ylabel('error eigen values')
    plt.ylim(0)
    plt.savefig('images/erreur_valeurs_propres.png',bbox_inches='tight')
    plt.show()
    
    intervalle_confiance = 1.96 * np.std(df['eigenvectors error'].to_numpy(), axis=0) / np.sqrt(df['number of examples'].to_numpy().astype(float))
    plt.errorbar(df['number of examples'], df['eigenvectors error'], yerr=intervalle_confiance, color='blue', label='erreurs vecteurs propres')
    plt.xscale('log')
    plt.xlabel("Nombre d'exemples")
    plt.ylabel('distance angle eigen vectors')
    plt.ylim(0)
    plt.savefig('images/distance_angle_vecteurs_propres.png',bbox_inches='tight')
    plt.show()
    plt.plot(df['number of examples'], df['mse reconstruction train'], color='red', label='MSE reconstruction train')
    plt.plot(df['number of examples'], df['mse reconstruction test'], color='blue', label='MSE reconstruction test')
    plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel("Nombre d'exemples")
    plt.ylabel('MSE reconstruction')
    plt.ylim(0)
    plt.legend()
    plt.savefig('images/mse_reconstruction.png',bbox_inches='tight')
    plt.show()
    
    #affiche la variance expliquée par les composantes principales avec eigen values
    variance_expliqué_pourcentage= []
    eigen_values=eigen_values[::-1]
    for i in range(len(eigen_values)-1,-1,-1):
        variance_expliqué_pourcentage.append(eigen_values[i]/np.sum(eigen_values)*100)
        
    plt.bar(range(1, len(variance_expliqué_pourcentage)+1), variance_expliqué_pourcentage)
    plt.xlabel('Composante principale')
    plt.ylabel('Variance expliquée (%)')
    plt.savefig('images/variance_theorique.png',bbox_inches='tight')
    plt.show()
    plot_by_eigen_value_biais(df)
    plot_by_eigen_vector_biais(df)
        
        
def variance_eigen_value(eigen_values,variance_total_values,nombre_exemples):
    
    '''
    Cette fonction trace la variance des erreurs sur les valeurs propres en fonction du nombre d'exemples.
    Arguments :
        eigen_values : Valeurs propres utilisées pour générer les données.
        variance_total_values : Liste de la variance des erreurs pour chaque valeur propre.
        nombre_exemples : Liste du nombre d'exemples utilisé pour chaque simulation.'''
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta']
    for i in range(5):
        variances_for_eigenvalue_values = [variances[i] for variances in variance_total_values]  
        plt.plot(nombre_exemples, variances_for_eigenvalue_values,color=colors[i])
        
    plt.xlabel('Nombre d\'exemples')
    plt.ylabel('Variance')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('images/variance_biais_valeurs_propres.png',bbox_inches='tight')
    plt.show()
    
def variance_eigen_vectors(eigen_vectors,variance_total_vectors,nombre_exemples):
    ''' 
    Cette fonction trace la variance des erreurs sur les vecteurs propres en fonction du nombre d'exemples.
    Arguments :
        eigen_vectors : Vecteurs propres utilisés pour générer les données.
        variance_total_vectors : Liste de la variance des erreurs pour chaque vecteur propre.
        nombre_exemples : Liste du nombre d'exemples utilisé pour chaque simulation.'''
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta']
    for i in range(5):
        variances_for_eigenvalue_vectors = [variances[i] for variances in variance_total_vectors]
        plt.plot(nombre_exemples, variances_for_eigenvalue_vectors,color=colors[i])
        
    plt.xlabel('Nombre d\'exemples')
    plt.ylabel('Variance')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('images/variance_biais_vecteurs_propres.png',bbox_inches='tight')
    plt.show()
    
    
def variance_biais_vectors_values(dimension, exempleMax, eigen_values=None, eigen_vectors=None):   
    '''
    Cette fonction calcule et trace la variance des biais sur les valeurs propres et les vecteurs propres pour un nombre donné d'exemples
    et de simulations. Les valeurs propres et les vecteurs propres utilisés pour générer les données peuvent être fournis, 
    sinon ils seront générés aléatoirement.
    Arguments :
        dimension : Dimension de l'espace des données.
        exempleMax : Nombre maximum d'exemples à utiliser pour chaque simulation.
        eigen_values : Valeurs propres utilisées pour générer les données. Par défaut, ce paramètre est None.
        eigen_vectors : Vecteurs propres utilisés pour générer les données. Par défaut, ce paramètre est None.
    ''' 
    pca = PCA.PCA()
    nb_exemples = 30
    nombre_exemples = []  
    nb_tirages = 1000
    variance_total_values = []
    variance_total_vectors = []
    print(eigen_values)
    while nb_exemples < exempleMax:
        error_each_eigen_values = []
        error_each_eigen_vectors = []
        for j in range(nb_tirages):
            X = generate_data(nb_exemples, eigen_values, eigen_vectors)
            pca.fit(X)
            errors_values = (eigen_values - pca.get_eigen_values())
            errors_vectors = distance_angle_vector(eigen_vectors, pca.get_eigen_vectors())
            error_each_eigen_values.append(errors_values)
            error_each_eigen_vectors.append(errors_vectors)
        variance_partiel_values=[]
        variance_partiel_vectors=[]
        error_each_eigen_values = np.array(error_each_eigen_values)
        error_each_eigen_vectors = np.array(error_each_eigen_vectors)
        for i in range(dimension):
            variance_partiel_values.append(np.var(error_each_eigen_values[:,i]))
            variance_partiel_vectors.append(np.var(error_each_eigen_vectors[:,i]))
            
        variance_total_values.append(variance_partiel_values)
        variance_total_vectors.append(variance_partiel_vectors)
        
        nombre_exemples.append(nb_exemples)  
        nb_exemples *= 2
        

    variance_eigen_value(eigen_values,variance_total_values,nombre_exemples)
    variance_eigen_vectors(eigen_vectors,variance_total_vectors,nombre_exemples)
    
    
def mse_dimension(df):
    ''' Cette fonction génère un graphique qui affiche l'erreur quadratique moyenne de reconstruction (MSE) en fonction de la dimension de l'espace des données.
        Arguments :
        df : DataFrame contenant les erreurs de reconstruction et la dimension.'''
    plt.figure(figsize=(10,6))
    sns.lineplot(x='dimension', y='mse reconstruction train',color="b", data=df, ci='sd')
    #sns.lineplot(x='dimension', y='mse reconstruction test',color="r",data=df, ci='sd')
    plt.xlabel('Dimension')
    plt.ylabel('MSE reconstruction')
    plt.savefig('images/mse_dimension.png',bbox_inches='tight')
    plt.show()