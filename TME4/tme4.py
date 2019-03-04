import arftools
import matplotlib.pyplot as plt
import numpy as np
import tME3.tme3 as tme3


def mse(datax, datay, w):
    """ retourne la moyenne de l'erreur aux moindres carres """
    return np.mean(np.square(datax.dot(w.T) - datay))


def mse_g(datax, datay, w, biais=0.0):
    """ retourne le gradient moyen de l'erreur aux moindres carres """
    return - 2 * np.mean(datax.T.dot(datay - datax.dot(w.T) + biais), axis=0)


def hinge(datax, datay, w):
    """ retourn la moyenne de l'erreur hinge """
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
    
    def fit(self, datax, datay, testx=None, testy=None):
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
        
        self.w = np.random.random((1, D))
        
        if self.use_biais:
            self.b = np.random.random()
            
        #todo dérivée du biais sans l'intégrer dans w ?
            
        for epoch in range(self.max_iter):
            grad = self.loss_g(datax, datay, self.w, biais=self.b)
            self.w -= self.eps * grad
            
            #if self.b:
            #    self.b -= grad[1]
    
    def predict(self, datax):
        if len(datax.shape) == 1:
            datax = datax.reshape(1, -1)
        return np.sign(datax.dot(self.w.T))
    
    def score(self, datax, datay):
        return sum(self.predict(x) == y for (x, y) in zip(datax, datay)) / len(datay)


def load_usps(fn):
    with open(fn, "r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split()) > 2]
    tmp = np.array(data)
    return tmp[:, 1:], tmp[:, 0].astype(int)


def show_usps(data):
    plt.imshow(data.reshape((16, 16)), interpolation="nearest", cmap="gray")
    plt.colorbar()
    plt.show()


def plot_error(datax, datay, f):
    grid, x1list, x2list = arftools.make_grid(xmin=-4, xmax=4, ymin=-4, ymax=4)
    plt.contourf(x1list, x2list, np.array([f(datax, datay, w) for w in grid]).reshape(x1list.shape), 25)
    plt.colorbar()
    plt.show()


def aff(trainx, trainy):
    plt.figure()
    plot_error(trainx, trainy, mse)
    plt.figure()
    plot_error(trainx, trainy, hinge)

def _main_presque_sep():
    trainx, trainy = arftools.gen_arti(nbex=1000, data_type=0, epsilon=1)
    testx, testy = arftools.gen_arti(nbex=1000, data_type=0, epsilon=1)
    
    #aff(trainx, trainy)
    
    carres = Lineaire(mse, mse_g, max_iter=1000, eps=0.1)
    carres.fit(trainx, trainy)
    print("MSE (pas fait pour la classification)")
    print("w (diverge) : ", carres.w)
    print("Score : train %f, test %f" % (carres.score(trainx, trainy), carres.score(testx, testy)))
    
    perceptron = Lineaire(hinge, hinge_g, max_iter=1000, eps=0.1)
    perceptron.fit(trainx, trainy)
    print("Hinge")
    print("w:", perceptron.w)
    print("Score : train %f, test %f" % (perceptron.score(trainx, trainy), perceptron.score(testx, testy)))
    plt.figure()
    
    arftools.plot_frontiere(trainx, perceptron.predict, 200)
    arftools.plot_data(trainx, trainy)
    plt.show()

def _main_usps():
    images, classes = tme3.load_usps("../tME3/USPS_train.txt")
    id_c0, id_c1 = 0, 1
    images_c0 = np.array([images[i] for i in range(len(images)) if classes[i]==id_c0])
    images_c1 = np.array([images[i] for i in range(len(images)) if classes[i]==id_c1])
    ratio_train = .8
    n_train0 = int(len(images_c0) * ratio_train)
    n_train1 = int(len(images_c1) * ratio_train)
    print(images_c0.shape)
    print(images_c1.shape)
    trainx, trainy = images_c0[:n_train0], images_c1[:n_train1]
    
def _main():
    #_main_presque_sep()
    
    _main_usps()
    
if __name__ == "__main__":
    _main()
    