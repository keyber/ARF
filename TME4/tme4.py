import matplotlib.pyplot as plt
import numpy as np
from tME3 import tme3
import sys
sys.path.append('./utils/')
from utils import arftools


def mse(datax, datay, w):
    """ retourne la moyenne de l'erreur aux moindres carres """
    return np.mean(np.square(datax.dot(w.T) - datay))


def mse_g(datax, datay, w, biais=0.0):
    """ retourne le gradient moyen de l'erreur aux moindres carres """
    return - 2 * np.mean(datax.T.dot(datay - datax.dot(w.T) + biais), axis=0)


def hinge(datax, datay, w):
    """ retourne la moyenne de l'erreur hinge """
    return np.mean(np.maximum(0, datay - datax.dot(w) + 1))


def hinge_g(datax, datay, w, biais=0.0):
    """ retourne le gradient moyen de l'erreur hinge """
    x = datax.dot(w.T) + biais
    x = np.multiply(datay, x)
    return np.mean(np.where(x < 1, - np.multiply(datay, datax), 0), axis=0)


class Lineaire(object):
    def __init__(self, loss=hinge, loss_g=hinge_g, use_biais=False, max_iter=1000, eps=1e-3):
        """ :loss: fonction de cout
            :loss_g: gradient de la fonction de cout
            :max_iter: nombre d'iterations
            :eps: pas de gradient
        """
        self.max_iter = max_iter
        self.eps = eps
        self.loss, self.loss_g = loss, loss_g
        self.w = None
        self.use_biais = use_biais
        self.b = None if use_biais else 0.0
    
    def fit(self, datax, datay, log=None, testx=None, testy=None):
        """ :datax: donnees de train
            :datay: label de train
            :testx: donnees de test
            :testy: label de test
        """
        # on transforme datay en vecteur colonne
        datay = datay.reshape(-1, 1)
        N = len(datay)
        datax = datax.reshape(N, -1)
        D = datax.shape[1]
        
        if testx is not None:
            testy = testy.reshape(-1, 1)
            testx = testx.reshape(len(testy), D)
        
        self.w = np.random.random((1, D))
        
        if self.use_biais:
            self.b = np.random.random()
        
        list_score_train, list_score_test = None, None
        if log:
            list_score_train = []
            if testx is not None:
                list_score_test = []
            
        #todo dérivée du biais sans l'intégrer dans w ?
            
        for epoch in range(self.max_iter):
            grad = self.loss_g(datax, datay, self.w, biais=self.b)
            self.w -= self.eps * grad
            
            if log and epoch in log:
                if list_score_train is not None:
                    list_score_train.append(self.score(datax, datay))
                if list_score_test is not None:
                    list_score_test.append(self.score(testx, testy))
                    
            #if self.b:
            #    self.b -= grad[1]
        
        return list_score_train, list_score_test
    
    def predict(self, datax):
        if len(datax.shape) == 1:
            datax = datax.reshape(1, -1)
        return np.sign(datax.dot(self.w.T)) # dot renvoie shape (1,1)
    
    
    def score(self, datax, datay):
        return sum(self.predict(x)[0][0] == y for (x, y) in zip(datax, datay.reshape(-1))) / len(datay)


def show_usps(data):
    plt.imshow(data.reshape((16, 16)), interpolation="nearest", cmap="gray")
    plt.colorbar()
    plt.show()


def plot_error(datax, datay, f):
    grid, x1list, x2list = arftools.make_grid(xmin=-4, xmax=4, ymin=-4, ymax=4)
    plt.contourf(x1list, x2list, np.array([f(datax, datay, w) for w in grid]).reshape(x1list.shape), 25)
    plt.colorbar()
    plt.show()


def _main_presque_sep():
    trainx, trainy = arftools.gen_arti(nbex=1000, data_type=0, epsilon=1)
    testx, testy = arftools.gen_arti(nbex=1000, data_type=0, epsilon=1)
    print("MSE (pas fait pour la classification)")
    plt.figure()
    plot_error(trainx, trainy, mse)
    
    epochs = 1000
    plotted = range(0, epochs, epochs//10)
    carres = Lineaire(mse, mse_g, max_iter=epochs, eps=0.1)
    err_train, err_test = carres.fit(trainx, trainy, plotted, testx, testy)
    # print(err_train)
    # print(err_test)
    print("w (diverge) : ", carres.w)
    print("Score : train %f, test %f" % (carres.score(trainx, trainy), carres.score(testx, testy)))
    plt.plot(plotted, err_train)
    plt.plot(plotted, err_test)
    plt.close('all')
    
    
    print("Hinge")
    plt.figure()
    plot_error(trainx, trainy, hinge)
    
    epochs = 1000
    plotted = range(0, epochs, epochs//10)
    perceptron = Lineaire(hinge, hinge_g, max_iter=epochs, eps=0.1)
    err_train, err_test = perceptron.fit(trainx, trainy, plotted, testx, testy)
    print("w:", perceptron.w)
    print("Score : train %f, test %f" % (perceptron.score(trainx, trainy), perceptron.score(testx, testy)))
    plt.plot(plotted, err_train)
    plt.plot(plotted, err_test)
    
    plt.figure()
    arftools.plot_frontiere(trainx, perceptron.predict, step=200)
    arftools.plot_data(trainx, trainy)
    plt.show()

def _main_usps():
    print("\nUSPS (hinge) 0 vs 1")
    train = tme3.load_usps("../tME3/USPS_train.txt")
    test  = tme3.load_usps("../tME3/USPS_test.txt")
    
    # extrait seulement deux classes
    id_c0, id_c1 = 0, 1
    kept_train = np.where((train[1] == id_c0) | (train[1] == id_c1))[0]
    kept_test  = np.where((test [1] == id_c0) | (test [1] == id_c1))[0]
    train = [train[0][kept_train], train[1][kept_train]]
    test  = [test [0][kept_test ], test [1][kept_test ]]
    # met les étiquettes -1/1
    train[1] = np.where(train[1] == id_c0, -1, 1)
    test [1] = np.where(test [1] == id_c0, -1, 1)
    
    epochs = 100
    plotted = range(0, epochs, epochs//10)
    perceptron = Lineaire(hinge, hinge_g, max_iter=epochs, eps=1e0)
    err_train, err_test = perceptron.fit(train[0], train[1], log=plotted)
    # print("err_train:", err_train)
    plt.plot(plotted, err_train)
    plt.show()
    print("Score : train %f, test %f" % (perceptron.score(train[0], train[1]), perceptron.score(test[0], test[1])))
    
def _main():
    _main_presque_sep()
    
    # _main_usps()
    
if __name__ == "__main__":
    _main()

