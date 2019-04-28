import matplotlib.pyplot as plt
import numpy as np

def load(filename):
    return plt.imread(filename)[ : , : , : 3 ] #on  garde  que  les  3  premieres  composantes,  la transparence  est  inutile

def cout(elements, labels, n_cluster):
    """cout de reconstruction: somme des distances intra-cluster"""
    clusters = mean_clusters(elements, labels, n_cluster)
    dist = dist_matrix(elements, clusters)
    return np.sum(np.min(dist, axis=1))
    
def mean_clusters(elements, labels, n_cluster):
    """les nouveaux clusters sont la moyenne des éléments qui le composent"""
    res = np.array([np.mean(elements[labels == i], axis=0) for i in range(n_cluster)])
    
    if np.any(np.isnan(res)):
        # on replace les clusters vides au centre de tous les points de l'ensemble
        centre = np.mean(elements, axis=0)
        for x in np.where(np.isnan(res)):
            res[x] = centre
    
    return res
    
def dist_matrix(elements, clusters):
    elements = elements.reshape((-1, 1, 3))
    return np.sum(np.power(elements - clusters, 2), axis=2)

def associate(elements, clusters):
    """associe à chaque element le cluster le plus proche de lui"""
    dist = dist_matrix(elements, clusters)
    return np.argmin(dist, axis=1)
    
def main():
    image = load("montagne.jpeg")
    # image = load("route.jpeg")
    elements = image.reshape((-1, 3))
    elements = elements/np.max(elements)
    plt.imshow(image)
    plt.show()
    
    for n_color in range(5, 30, 5):
        # le nombre de cluster correspond au nombre de couleurs différentes
        
        # on associe un cluster à chaque pixel aléatoirement
        curr_labels = np.random.randint(0, n_color, len(elements))
        clusters = None
        
        for _ in range(5):
            clusters = mean_clusters(elements, curr_labels, n_color)
            curr_labels = associate(elements, clusters)
            print(n_color, "cout", cout(elements, curr_labels, n_color))
        
        new_image = np.array(clusters[curr_labels]).reshape(image.shape)
        # plt.imshow(image)
        plt.imshow(new_image)
        plt.show()
    

if __name__ == '__main__':
    main()