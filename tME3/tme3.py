import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def make_grid(xmin=-5, xmax=5, ymin=-5, ymax=5, step=20, data=None):
    """ Cree une grille sous forme de matrice 2d de la liste des points
    :return: une matrice 2d contenant les points de la grille, la liste x, la liste y"""
    if data is not None:
        xmax, xmin, ymax, ymin = np.max(data[:, 0]), np.min(data[:, 0]), \
                                 np.max(data[:, 1]), np.min(data[:, 1])
    x, y = np.meshgrid(np.arange(xmin, xmax, (xmax - xmin) * 1. / step),
                       np.arange(ymin, ymax, (ymax - ymin) * 1. / step))
    grid = np.c_[x.ravel(), y.ravel()]
    return grid, x, y


def draw_2D(f):
    grid, xx, yy = make_grid(-1, 3, -1, 3, 20)
    plt.figure()
    fgrid = np.array([f(x) for x in grid])
    plt.contourf(xx, yy, fgrid.reshape(xx.shape))
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx, yy, fgrid.reshape(xx.shape), rstride=1, cstride=1,
                           cmap=cm.gist_rainbow, linewidth=0, antialiased=False)
    fig.colorbar(surf)
    plt.show()


def draw_1D(f, x_ini, n=100, eps=1e-4):
    x, f, grad = optimize(f[0], f[1], x_ini, eps, n)
    plt.figure("g")
    plt.plot(grad)
    plt.figure("xi & f(xi)")
    plt.plot(range(n), x, f)
    plt.figure("log(err)")
    plt.plot([np.log(np.linalg.norm(x[i] - x[-1])) for i in range(n)])
    plt.show()


def optimize(fonc, dfonc, xinit, eps, max_iter):
    """return (x_list, f_list, grad_list)"""
    xinit = np.array(xinit)
    x_list, f_list, grad_list = [], [], []
    x = xinit
    for _ in range(max_iter):
        x_list.append(x.copy())
        
        f_list.append(fonc(x))
        
        grad = np.array(dfonc(x))
        
        grad_list.append(grad)
        
        x -= eps * grad
    
    return np.array(x_list), np.array(f_list), np.array(grad_list)


def xcosx():
    """non convexe"""
    
    def f1d_1_val(x):
        return [x[0] * np.cos(x[0])]
    
    def f1d_1_grad(x):
        return [np.cos(x[0]) - np.sin(x[0]) * x[0]]
    
    return f1d_1_val, f1d_1_grad


def x2_minus_logx():
    """convexe"""
    
    def f1d_2_val(x):
        return [-np.log(x[0]) + x[0] ** 2]
    
    def f1d_2_grad(x):
        return [- 1 / x[0] + 2 * x[0]]
    
    return f1d_2_val, f1d_2_grad


def fRosenbrock():
    """Rosenbrock performance test problem for optimization algorithms"""
    
    def f2d_1_val(x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    
    def f2d_1_grad(x):
        return [100 * (-2 * x[0] * (2 * (x[1] - x[0] ** 2))) - 2 * (1 - x[0]),
                100 * 2 * (x[1] - x[0] ** 2)]
    
    return [f2d_1_val, f2d_1_grad]


def load_usps(filename):
    with open(filename, "r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split()) > 2]
    tmp = np.array(data)
    return tmp[:, 1:], tmp[:, 0].astype(int)


class Logistic:
    def __init__(self):
        # w0 stocke b
        self.W = None
    
    @staticmethod
    def _loss(W, X, Y):
        W, b = W[1:], W[0]
        return sum(np.log(1 + np.exp(-(2 * Y[i] - 1) * (X[i].dot(W) + b)))
                   for i in range(len(Y))) / len(X)
    
    @staticmethod
    def _grad_loss(W, X, Y):
        W, b = W[1:], W[0]
        
        cache = np.array([-(2 * Y[j] - 1) / (1 + np.exp((2 * Y[j] - 1) * (X[j].dot(W) + b))) for j in range(len(X))])
        
        grad_w = [np.sum(cache * X[:, i]) for i in range(len(W))]
        
        grad_b = np.sum(cache)

        grad = np.array([grad_b] + grad_w) / len(X)

        return grad
    
    def fit(self, datax, datay, eps, max_iter):
        w, list_f, grad = optimize(fonc=lambda x: self._loss(x, datax, datay),
                          dfonc=lambda x: self._grad_loss(x, datax, datay),
                          # zeros ou random normalement, tout marche ici
                          xinit=np.random.random(datax.shape[1] + 1) * 2 - 1, # rajoute un nombre pour représenter b
                          eps=eps, max_iter=max_iter)
        plt.figure(); plt.title("w"); plt.plot(w[2:])
        plt.figure(); plt.title("f"); plt.plot(list_f[2:])
        plt.figure(); plt.title("g"); plt.plot(grad[2:])
        plt.show()
        self.W = w[-1]
        return list_f
    
    def predict(self, datax):
        """maximum de vraisemblance, compare
        P(y==1|x) = 1/1+e-wx+b    à   1 - // """
        W, b = self.W[1:], self.W[0]
        pred = 1 / (1 + np.exp(-(datax.dot(W) + b)))
        return np.where(pred < .5, 0, 1)
    
    def score(self, datax, datay):
        """return nb de bonne classifications"""
        return np.sum(self.predict(datax) == datay)


def main():
    # draw_1D(x2_minus_logx(), x_ini=[10.], eps=1e-1, n=100)
    # draw_1D(fRosenbrock(), x_ini=[0.,0.], eps=1e-3, n=10000)
    # draw_2D(fRosenbrock()[0])
    
    c0, c1 = 0, 1
    trainx, trainy = load_usps("USPS_train.txt")
    ind_kept = np.where((trainy==c0) | (trainy==c1))[0]
    trainx = trainx[ind_kept]
    trainy = trainy[ind_kept]
    
    l = Logistic()
    print("fitting")
    losses = l.fit(trainx, trainy, eps=1e0, max_iter=1000)
    print("fitted")
    print("losses:", losses) # log(1 + e(x)) donne des inf au lieu de x
    plt.show()
    
    testx, testy = load_usps("USPS_test.txt")
    ind_kept = np.where((testy==c0) | (testy==c1))[0]
    testx = testx[ind_kept]
    testy = testy[ind_kept]
    score = l.score(testx, testy)
    print("score", score, "/", len(testy), "=", score/len(testy))


if __name__ == '__main__':
    main()
