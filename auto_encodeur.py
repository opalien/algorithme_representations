import numpy as np

class Autoencoder:
    def __init__(self, input_dim, hidden_layer_dim, learning_rate=0.01, epochs=100):
        '''
        Initialize an Autoencoder object
        
        Parameters:
        input_dim: int, the dimension of the input data.
        hidden_layer_dim: list, dimensions of the hidden layers.
        learning_rate: float, the learning rate for training the autoencoder. Default is 0.01.
        epochs: int, the number of epochs for training the autoencoder. Default is 100.
        '''
        self.input_dim = input_dim
        self.batch_size = 30
        self.W = []
        self.b = []
        
        for layer in range(len(hidden_layer_dim)):
            if layer == 0:
                self.W.append(np.random.randn(input_dim, hidden_layer_dim[layer]))
                self.initial_w = self.W[layer]
            else:
                self.W.append(np.random.randn(hidden_layer_dim[layer-1], hidden_layer_dim[layer]))
            self.b.append(np.zeros(hidden_layer_dim[layer]))
            self.W[layer] /= np.max(self.W[layer])
            
        self.W.append(np.random.randn(hidden_layer_dim[-1], input_dim))
        self.W[-1] /= np.max(self.W[-1])
        self.b.append(np.zeros(input_dim))
        self.learning_rate = learning_rate
        self.epochs = epochs

    def sigmoid(self, x):
        '''
        Sigmoid activation function.
        '''
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        '''
        Derivative of the sigmoid function.
        '''
        return x * (1 - x)
    
    def relu(self, x):
        '''
        ReLU (Rectified Linear Unit) activation function.
        '''
        return np.maximum(0, x)

    def relu_derivative(self, x):
        '''
        Derivative of the ReLU function.
        '''
        return np.where(x > 0, 1.0, 0.0)
    
    def linear(self, x):
        '''
        Linear activation function.
        '''
        M = np.eye(x.shape[0])
        M = 0.01 * M
        return np.dot(M, x)
    
    def linear_derivative(self, x):
        '''
        Derivative of the linear function.
        '''
        return 1
    
    def error(self, X, X_reconstructed):
        '''
        Compute the reconstruction error.
        '''
        return X_reconstructed - X
    
    def forward_propagation(self, x):
        '''
        Perform the forward propagation.
        '''
        h = []
        for layer in range(len(self.W)):
            if layer == 0:
                z = np.dot(x, self.W[layer]) + self.b[layer]
                h.append(self.linear(z))
            else:
                z = np.dot(h[layer-1], self.W[layer]) + self.b[layer]
                h.append(self.linear(z))
        l = self.error(x, h[-1])
        return h, l

    def gradient(self, x):
        '''
        Compute the gradient of the loss function with respect to the parameters.
        '''
        h, l = self.forward_propagation(x)
        l_derivative = 2 * l
        mse = np.mean(np.power(l, 2))

        dW = [0] * len(self.W)
        db = [0] * len(self.b)
        dh_next = l_derivative

        for layer in range(len(self.W)-1, -1, -1):
            dh = dh_next * self.linear_derivative(h[layer])

            if layer == 0:
                dW[layer] = np.dot(x.T, dh)
            else:
                        dh_next = l_derivative

        for layer in range(len(self.W)-1, -1, -1):
            dh = dh_next * self.linear_derivative(h[layer])

            if layer == 0:
                dW[layer] = np.dot(x.T, dh)
            else:
                dW[layer] = np.dot(h[layer-1].T, dh)

            db[layer] = np.sum(dh, axis=0)

            if layer > 0: 
                dh_next = np.dot(dh, self.W[layer].T)

        return dW, db, mse

    def fit(self, X,X_test=None):
        '''
        Entraîne l'autoencodeur sur un ensemble de données X.
        
        Paramètres:
        X: array, les données sur lesquelles entraîner l'autoencodeur.
        X_test: array, les données de test pour évaluer l'autoencodeur. Par défaut, c'est None.
        '''
        mse_train=[]
        mse_test=[]
        for epoch in range(self.epochs):
            X_mixed=X.copy()
            np.random.shuffle(X_mixed)
            mean_mse=[]
            mse=0
            indice_min=0
            indice_max=self.batch_size
            while indice_max+self.batch_size<=X_mixed.shape[0]:
                dW,db, mse = self.gradient(X_mixed[indice_min:indice_max])
                self.learning_rate=1/(np.linalg.norm(X_mixed[indice_min:indice_max])**2)
                for layer in range(len(self.W)):
                    self.W[layer] -= self.learning_rate * dW[layer]
                indice_min=indice_max
                indice_max+=self.batch_size
                mean_mse.append(mse)
            if(indice_max<X_mixed.shape[0]):
                dW,db, mse = self.gradient(X_mixed[indice_min:indice_max])
                self.learning_rate=1/(np.linalg.norm(X_mixed[indice_min:indice_max])**2)
                for layer in range(len(self.W)):
                    self.W[layer] -= self.learning_rate * dW[layer]
                indice_min=indice_max
                indice_max+=self.batch_size
                mean_mse.append(mse)
            
            mse_train.append(np.mean(mean_mse))
            mse_test.append(np.mean(np.power(X_test-self.reconstruct(X_test),2))) 
            
        return mse_train,mse_test

    def reconstruct(self, X):
        '''
        Reconstruit l'entrée à partir de la sortie de l'autoencodeur.
        
        Paramètres:
        X: array, les données à reconstruire.
        '''
        h,l= self.forward_propagation(X)
        return h[-1]
    
    def get_w(self):
        '''
        Renvoie les poids de l'autoencodeur.
        '''
        return self.W
    
    def get_ini(self):
        '''
        Renvoie les poids initiaux de l'autoencodeur.
        '''
        return self.initial_w

               
