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

def delete_rec(image, rec):
    res = image.copy()
    i, j, w, h = rec
    res[i:i+w, j:j+h] = np.full((w, h, 3), REMOVED)
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


def _undo_pixel_exact(patch, data_complete, eps):
    """on cherche un patch identique au notre à eps près"""
    if eps <= 0:
        return
    connus = patch != REMOVED
    inconnus = patch == REMOVED
    
    diff = data_complete[:, connus] - np.expand_dims(patch[connus], 0)
    dist = np.mean(np.power(diff, 2), axis=1)
    best = np.argmin(dist)
    
    if dist[best] < eps:
        patch[inconnus] = data_complete[best][inconnus]
        return True
    
    return False


def _undo_pixel_lasso(patch, data_complete, alpha, max_iter=1000):
    """on cherche une combinaison de pas trop de patchs"""
    connus = patch != REMOVED
    inconnus = patch == REMOVED
    m = sklearn.linear_model.Lasso(alpha=alpha, max_iter=max_iter)
    train = np.array([d[connus] for d in data_complete])
    train = train.T
    m.fit(train, patch[connus])
    
    patch[inconnus] = np.array([d[inconnus] for d in data_complete]).T.dot(m.coef_) + m.intercept_


def undo_noise(image, w=3, threshOS=0.0, alpha=.1, max_iter=1000):
    coo, complete = split_complete_incomplete(image, w)
    data = np.array([get_patch(image, x, y, w) for (x,y) in coo])
    data_complete = data[complete==True]
    
    for x,y in coo[complete==False]:
        patch = get_patch(image, x, y, w)
        if not _undo_pixel_exact(patch, data_complete, threshOS):
            _undo_pixel_lasso(patch, data_complete, alpha, max_iter)


def _get_spiral_coo(rect, w):
    x0, y0, x1, y1 = rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]
    
    # shift pour que le patch ne contienne que le point courrant
    # sens de parcourt: gauche bas droite haut
    sx = [-w + 1, 0, 0, -w + 1]
    sy = [-w + 1, -w + 1, 0, 0]
    points = []
    # côté actuel
    c = 0
    while x0 != x1 and y0 != y1:
        if c == 0:
            points += [(x + sx[c], y0 + sy[c]) for x in range(x0, x1, 1)]
            y0 += 1
        elif c == 1:
            points += [(x1 - 1 + sx[c], y + sy[c]) for y in range(y0, y1, 1)]
            x1 -= 1
        elif c == 2:
            points += [(x + sx[c], y1 - 1 + sy[c]) for x in range(x1 - 1, x0 - 1, -1)]
            y1 -= 1
        elif c == 3:
            points += [(x0 + sx[c], y + sy[c]) for y in range(y1 - 1, y0 - 1, -1)]
            x0 += 1
        c = (c + 1) % 4
    
    return points

def undo_rect(image, rect, w=3, threshOS=0.0, alpha=.01, max_iter=1000, verbose=False):
    coo, complete = split_complete_incomplete(image, w)
    
    data = np.array([get_patch(image, x, y, w) for (x, y) in coo])
    data_complete = data[complete == True]
    for x, y in _get_spiral_coo(rect, w):
        patch = get_patch(image, x, y, w)
        if not _undo_pixel_exact(patch, data_complete, threshOS):
            _undo_pixel_lasso(patch, data_complete, alpha=alpha, max_iter=max_iter)
    
def _main1():
    im = read_im("degrade2_petit.png")
    # print_im(im)
    p = .001
    bruitee = noise(im, p)
    print("image", im.shape, "nb pixels", im.size // 3)
    print("nb à reconstruire", int(im.size // 3 * p))
    print_im(bruitee)
    undo_noise(bruitee)
    print_im(bruitee)
    print("loss", np.mean(np.power(im - bruitee, 2)))
    plt.show()

def _main2():
    im = read_im("degrade2_petit.png")
    rectangles = [
        (20,20,2,10),
        (50,20,8,16),
        (25,50,8,8),
        (50,60,3,10),
    ]
    bruitee = im
    for r in rectangles:
        bruitee = delete_rec(bruitee,r)
    print("image", im.shape, "nb pixels", im.size // 3)
    print("nb à reconstruire", sum(r[2]*r[3] for r in rectangles))
    # print_im(bruitee)
    for i, r in enumerate(rectangles):
        print(i, "/", len(rectangles))
        undo_rect(bruitee,r, w=3, threshOS=.01, alpha=.001, max_iter=int(1e4), verbose=True)
    print_im(bruitee)
    print("loss", np.mean(np.power(im - bruitee, 2)))
    plt.show()

if __name__ == '__main__':
    # _main1()
    _main2()
