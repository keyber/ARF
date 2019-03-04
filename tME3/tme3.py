import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def make_grid(xmin=-5,xmax=5,ymin=-5,ymax=5,step=20,data=None):
    """ Cree une grille sous forme de matrice 2d de la liste des points
    :return: une matrice 2d contenant les points de la grille, la liste x, la liste y
    """
    if data is not None:
        xmax,xmin,ymax,ymin = np.max(data[:,0]),np.min(data[:,0]),\
                              np.max(data[:,1]),np.min(data[:,1])
    x,y = np.meshgrid(np.arange(xmin,xmax,(xmax-xmin)*1./step),
                      np.arange(ymin,ymax,(ymax-ymin)*1./step))
    grid=np.c_[x.ravel(),y.ravel()]
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
    plt.plot(range(n), x,f)
    plt.figure("log(err)")
    plt.plot([np.log(np.linalg.norm(x[i]-x[-1])) for i in range(n)])
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
        return [x[0]*np.cos(x[0])]

    def f1d_1_grad(x):
        return [np.cos(x[0]) - np.sin(x[0]) * x[0]]
    
    return f1d_1_val, f1d_1_grad


def x2_minus_logx():
    """convexe"""
    def f1d_2_val(x):
        return [-np.log(x[0]) + x[0]**2]
    
    def f1d_2_grad(x):
        return [- 1 / x[0] + 2*x[0]]
    
    return f1d_2_val, f1d_2_grad


def fRosenbrock():
    """Rosenbrock performance test problem for optimization algorithms"""
    def f2d_1_val(x):
        return 100 * (x[1] - x[0]**2)**2  +  (1 - x[0])**2
    
    def f2d_1_grad(x):
        return [100 * (-2*x[0] * (2 * (x[1] - x[0]**2)))  - 2 * (1 - x[0]),
                100 * 2 * (x[1] - x[0]**2)]
    
    return [f2d_1_val, f2d_1_grad]



def load_usps(filename):
    with open(filename,"r") as f:
        f.readline()
        data =[ [float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp = np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)


class logistic:
    def __init__(self, n, eps):
        self.n = n
        self.eps = eps
    
    @staticmethod
    def _fw(w, x):
        return x.dot(w)
    
    @staticmethod
    def _cout(w, x, y):
        return sum(np.log(1+np.exp(-(2*y[i] - 1) * logistic._fw(w, x[i]))) for i in range(len(y)))
    
    def fit(self, datax, datay):
        """cf cours 3 slide 14/44"""
        optimize()
        pass
    
    def predict(self, datax):
        """maximum de vraisemblance, compare
        P(y==1|x) = 1/1+e-wx+b
        à
        1 - //
        """
        pass
    
    def score(self, datax, datay):
        """return nb de bonne classifications"""
        pass
    
def logistique():
    """
    - somme des log(P(Yi|Xi)
    = - somme des II(yi==1) log 1/1+e(-wxi+b) + II(yi==0) log 1+e(-wxi+b)
    :return:
    """
    
def main():
    #draw_1D(x2_minus_logx(), x_ini=[10.], eps=1e-1, n=100)
    draw_1D(fRosenbrock(), x_ini=[0.,0.], eps=1e-3, n=10000)
    draw_2D(fRosenbrock()[0])

if __name__ == '__main__':
    main()
