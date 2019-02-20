import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


parismap = mpimg.imread('data/paris-48.806-2.23--48.916-2.48.jpg')

## coordonnees GPS de la carte
xmin,xmax = 2.23,2.48   ## coord_x min et max
ymin,ymax = 48.806,48.916 ## coord_y min et max

def show_map():
    #extent pour controler l'echelle du plan
    plt.imshow(parismap,extent=[xmin,xmax,ymin,ymax],aspect=1.5)

poidata = pickle.load(open("data/poi-paris.pkl","rb"))
## liste des types de point of interest (poi)
print("Liste des types de POI" , ", ".join(poidata.keys()))

## Choix d'un poi
typepoi = "night_club"

## Creation de la matrice des coordonnees des POI
geo_mat = np.zeros((len(poidata[typepoi]),2))
for i,(k,v) in enumerate(poidata[typepoi].items()):
    geo_mat[i,:]=v[0]

## Affichage brut des poi
show_map()
## alpha permet de regler la transparence, s la taille
plt.scatter(geo_mat[:,1], geo_mat[:,0],alpha=0.8,s=3)
###################################################

# discretisation pour l'affichage des modeles d'estimation de densite
steps = 10
xx,yy = np.meshgrid(np.linspace(xmin,xmax,steps),np.linspace(ymin,ymax,steps))
grid = np.c_[xx.ravel(),yy.ravel()]

plt.close('all')

class ModelHisto:
    def fit(self, points, step):
        self.points = points
        self.step = step
        self.n = len(self.points)
    
    def predict(self, _grid):
        #ne se sert pas de la grille pour simplifier
        plt.figure("figureHistoTmp")
        h = plt.hist2d(self.points[:, 0], self.points[:, 1], bins=self.step)[0]
        h = h / self.n
        plt.close("figureHistoTmp")
        return h

def distance(x1, x2):
    return

class ModelParzen:
    def fit(self, points, distancewx, wy):
        self.points = points
        self.n = len(self.points)
        this.distance = distance
        self.wx = wx
        self.wy = wy
    
    def predict(self, grid):
        #x et y inversés
        return np.array([self.predictPoint(y,x) for (x,y) in grid])
    
    def predictPoint(self, x, y):
        #parcourt tous les points de la base d'apprentissage
        cpt = 0
        for (px, py) in self.points:
            if abs(x - px)<self.wx and abs(y - py) < self.wy:
                cpt += 1
        cpt /= self.wx * self.wy * self.n
        return cpt


#res = np.random.random((steps,steps))
modelHisto = ModelHisto()
modelHisto.fit(geo_mat, steps)

import math
modelParzen = ModelParzen()
modelParzen.fit(geo_mat, (xmax-xmin)/steps, (ymax-ymin)/steps)

for model in [modelHisto, modelParzen]:
    res = model.predict(grid).reshape(steps,steps)
    plt.figure()
    plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',
                   alpha=0.3,origin = "lower")
    plt.colorbar()
plt.show()
