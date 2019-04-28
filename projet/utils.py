import sklearn.linear_model
import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
import itertools


REMOVED = -100

def read_im(filename):
    im = plt.imread(filename)[:,:,:3]
    if np.any(im>1):
        im = im / 255
    return col.rgb_to_hsv(im)

def print_im(image):
    plt.figure()
    plt.imshow(col.hsv_to_rgb(image))

def get_patch(image, i, j, h):
    """i, j are the upper left coordinate"""
    return image[i:i+h, j:j+h]

def patch_to_vec(patch):
    return patch.reshape((-1,))

def vec_to_patch(patch, h):
    return patch.reshape((h, h, 3))

def noise(image, p, marge=10):
    """ne bruite pas les bords"""
    res = image.copy()
    
    l = np.array(list(itertools.product(range(res.shape[0]-2*marge), range(res.shape[1]-2*marge))))
    indexes = l[np.random.choice(range(len(l)), int(image.shape[0] * image.shape[1] * p))]
    for x, y in indexes:
        res[x+marge, y+marge, :] = (REMOVED, REMOVED, REMOVED)
    
    return res

def delete_rec(image, i, j, w, h):
    res = image.copy()
    res[i:i+w, j:j+h] = np.full((w,h), REMOVED)
    return res

def split_complete_incomplete(image, h):
    """retourne
    complete:"dictionnaire",
    incomplete: à compléter"""
    coo_patches = [[(i,j)
                    for j in range(0, image.shape[1] - h, h)]
                   for i in range(0, image.shape[0] - h, h)]
    coo_patches = np.array(coo_patches).reshape(-1, 2)

    complete = np.array([not np.any(get_patch(image, x, y, h) == REMOVED) for x,y in coo_patches])
    
    return coo_patches, complete


def undo_noise(image, w=3, alpha=.01):
    coo, complete = split_complete_incomplete(image, w)
    data = np.array([get_patch(image, x, y, w) for (x,y) in coo])
    data_complete = data[complete==True]
    
    for x,y in coo[complete==False]:
        patch = get_patch(image, x, y, w)
        connus   = patch != REMOVED
        inconnus = patch == REMOVED
        
        m = sklearn.linear_model.Lasso(alpha=alpha, tol=1e-6, max_iter=1e5)
        train = np.array([d[connus] for d in data_complete])
        train = train.T
        m.fit(train, patch[connus])
        print("loss train", np.mean(np.power(train.dot(m.coef_) + m.intercept_ - patch[connus], 2)),
              "nb poids", np.count_nonzero(m.coef_), "/", len(m.coef_))
        
        patch[inconnus] = np.array([d[inconnus] for d in data_complete]).T.dot(m.coef_) + m.intercept_

def undo_rect(image, rect, w=3, alpha=.01):
    pass

def _main():
    im = read_im("dégradé2.jpeg")
    # print_im(im)
    p = .001
    bruitee = noise(im, p)
    print("image", im.shape, "nb pixels", im.size//3)
    print("nb à reconstruire", int(im.size//3 * p))
    print_im(bruitee)
    undo_noise(bruitee)
    print_im(bruitee)
    print("loss", np.mean(np.power(im - bruitee, 2)))
    plt.show()
    
if __name__ == '__main__':
    _main()
