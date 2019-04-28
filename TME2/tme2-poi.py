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

# plt.show()

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

def similarite_parzen(x1, x2, width):
    diff = x1[0] - x2[0], x1[1] - x2[1]
    norm = diff[0] ** 2 + diff[1] ** 2
    return 1/width**2 if norm < width**2 else 0

def similarite_gauss(x1, x2, sigma_2):
    diff = x1[0] - x2[0], x1[1] - x2[1]
    norm_2 = diff[0] ** 2 + diff[1] ** 2
    return 1/np.sqrt(2*np.pi*sigma_2) * np.exp(-norm_2/(2*sigma_2))

class ModelNoyau:
    def fit(self, points, distance):
        self.points = points
        self.n = len(self.points)
        self.similarite = distance
    
    def predict(self, grid):
        #x et y inversés
        res = np.array([self.predictPoint(y,x) for (x,y) in grid])
        return res / np.sum(res)
    
    def predictPoint(self, x, y):
        # parcourt tous les points de la base d'apprentissage
        s = sum(self.similarite((x, y), (px, py)) for (px, py) in self.points)
        return s / self.n


def _main():
    modelHisto = ModelHisto()
    modelHisto.fit(geo_mat, steps)
    
    modelParzen = ModelNoyau()
    modelParzen.fit(geo_mat, lambda x1, x2: similarite_parzen(x1, x2, .05))
    
    modelGauss = ModelNoyau()
    modelGauss.fit(geo_mat, lambda x1,x2: similarite_gauss(x1, x2, .0001))
    
    for model in [modelHisto, modelParzen, modelGauss]:
        res = model.predict(grid).reshape(steps,steps)
        plt.figure()
        plt.imshow(res, extent=[xmin,xmax,ymin,ymax], interpolation='none', alpha=0.3, origin="lower")
        plt.colorbar()
    plt.show()

if __name__ == '__main__':
    _main()