import scipy
import numpy as np
from sklearn.decomposition import PCA as sklearnPCA
class PCA:
    """
    A class for performing Principal Component Analysis (PCA) on a given dataset.
    """    
    def __init__(self, n_components=None):
        """
        Initialize the PCA object with the given number of components.

        Parameters:
        -----------
        n_components: int
            The number of principal components to use in the PCA. If None, then all components will be used.
            Specify n_components only for singular method. All components are anyway computed with eigen method.
        """
        self.n_components = n_components
        self.method='eigen'
        self.eigen_values = None
        self.eigen_vectors=None
        self.X=None
        
    def fit(self, X):
        """
        Fit the PCA model to the input data.

        Parameters:
         ----------
        X: numpy.ndarray
            The input data matrix to perform PCA on.
        """
        if self.method=="eigen":
            S=np.cov(X)
            self.X=X
            self.eigen_values,self.eigen_vectors=np.linalg.eigh(S)
            self.eigen_vectors = self.eigen_vectors[:,::-1]
            self.eigen_values = self.eigen_values[::-1]
                
        
        
    def project_data(self,n_components,X_data):
        """
        Project the input data matrix onto the principal components using eigen values and eigen vectors. 

        Parameters:
        -----------
        n_components: int

        Returns:
        --------
        numpy.ndarray:
            The projected data matrix.
        """
             
        return np.dot(X_data, self.eigen_vectors[:, :n_components])
    
    def reconstruct_projected_data(self,X_data,n_components=None):
        """
        Approximation of X_data by the selected number of principal components.

        Parameters:
        -----------
        n_components: int
            the number of components to use for the reconstruction.

        Returns:
        --------
        numpy.ndarray:
            The projected data matrix.
        """
        if(n_components==None and self.n_components==None):
            n_components=self.X.shape[1] 
        elif(n_components==None and self.n_components!=None):
            n_components=self.n_components
        X_projected=self.project_data(n_components,X_data)
        return np.dot(X_projected,self.eigen_vectors[:, :n_components].T)

        
    def explained_variance(self):
        """
        Calculate the explained variance for each principal component.
        
        Returns:
        --------
        numpy.ndarray:
            The explained variance for each principal component.
        """
        if self.eigen_values is None:
            raise ValueError("PCA has not been fitted yet")
        if(self.method=='eigen'):
            total_variance = np.sum(self.eigen_values)
            explained_variance = self.eigen_values / total_variance
            return explained_variance[::-1]

    
    
    def get_eigen_values(self):
        """
        Returns the eigen values of the input data.

        Returns:
        --------
         numpy.ndarray:
        The eigen values of the input data.
        """
        return self.eigen_values
    
    def get_eigen_vectors(self):
        """
        Returns the eigen vectors of the input data.

        Returns:
        --------
         numpy.ndarray:
        The eigen vectors of the    input data.
        """
        return self.eigen_vectors