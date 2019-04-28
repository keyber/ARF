from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    @abstractmethod
    def forward(self, y, yhat):
        pass

    @abstractmethod
    def backward(self, y, yhat):
        pass


class Module(ABC):
    def __init__(self, param):
        self.__parameters = param
        self.__gradient = np.zeros_like(param)

    def update_parameters(self, gradient_step):
        """màj paramètres du module selon le gradient actuel
        avec un pas de la taille de gradient_step"""
        self.__parameters += self.__gradient * gradient_step

    def zero_grad(self):
        """annule le gradient actuellement stocké"""
        self.__gradient = np.zeros_like(self.__gradient)
    
    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward_update_gradient(self, input, delta):
        """calcule le gradient du coût par rapport aux paramètres
        et l'ajoute au gradient actuel"""
        pass

    @abstractmethod
    def backward_delta(self, input, delta):
        """calcule le gradient du coût par rapport aux entrées
        et l'ajoute au gradient actuel"""
        pass
        

class ModuleLineaire(Module):
    def __init__(self, n_in, n_out):
        # rajoute une case pour le biais
        super().__init__(np.zeros(n_in + 1, n_out))
        self.n_in = n_in
        self.n_out = n_out
    
    def forward(self, X):
        """(1, in)  *  (in, out)  ->  (1, out)"""
        assert X.shape == (1, self.n_in)
        
        xw = np.dot(X, self.__parameters[:, 1:])
        
        # biais
        xw += self.__parameters[:, 0]
        
        assert xw.shape == (1, self.n_out)
        return xw
    
    def backward_update_gradient(self, input, delta):
        """input: (1, in)
        delta: (1, n)
        """
        assert input.shape == self.n_in
        
        gradient = np.dot(delta, input)
        
        self.__gradient += gradient
    
    def backward_delta(self, input, delta):
        ## Calcul la dérivée de l'erreur
        input = input.reshape(-1, 1)
        sortie = np.dot(self.__parameters.T, input)
        return sortie * delta
        #return self._loss.backward(sortie, delta).T


class FASigmoide(Module):
    def __init__(self):
        pass
    
    def sigmoid(self, input):
        return 1.0 / (1.0 + np.exp(-input))
    
    def sigmoid_prime(self, input):
        return self.sigmoid(input) * (1 - self.sigmoid(input))
    
    def forward(self, X):
        ### Calcule la passe forward
        return self.sigmoid(X)
    
    def backward_delta(self, input, delta):
        ## Calcul la dérivée de l'erreur
        sp = self.sigmoid_prime(input)
        return delta * sp


class MSELoss(Loss):
    def forward(self, y, yhat):
        return ((y - yhat) ** 2).mean()
    
    def backward(self, y, yhat):
        return (yhat - y)  #.mean()