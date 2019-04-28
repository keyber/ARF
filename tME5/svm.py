import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt
import tme3
import sys
sys.path.append('./utils/')
from utils import arftools

all_train = tme3.load_usps("../tME3/USPS_train.txt")
all_test = tme3.load_usps("../tME3/USPS_test.txt")

def _main_biclass():
    print("SVM (grid search - cross val)")

    # extrait seulement deux classes
    id_c0, id_c1 = 1, 7
    kept_train = np.where((all_train[1] == id_c0) | (all_train[1] == id_c1))[0]
    kept_test = np.where((all_test[1] == id_c0) | (all_test[1] == id_c1))[0]
    train = [all_train[0][kept_train], all_train[1][kept_train]]
    test = [all_test[0][kept_test], all_test[1][kept_test]]
    # met les étiquettes -1/1
    train[1] = np.where(train[1] == id_c0, -1, 1)
    test[1] = np.where(test[1] == id_c0, -1, 1)

    svc = SVC()
    
    # x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=.2)
    # svc.fit(x_train,y_train)
    # score = svc.score(x_test, y_test)
    #
    # skf = model_selection.StratifiedKFold(n_splits=3)
    # for train_ind, test_ind in skf.split(x,y):
    #     x_train, x_test = x[train_ind], y[test_ind]
    #     y_train, y_test = y[train_ind], y[test_ind]
    #
    # scores = model_selection.cross_val_score(svc, x, y, cv=3)
    
    parameters = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]
    clf = model_selection.GridSearchCV(svc, parameters, cv=3)
    clf.fit(train[0], train[1])
    print("best params", clf.best_params_, "score validation", clf.best_score_)
    score_test = clf.score(test[0], test[1])
    print("score test", score_test)

class PredicteurOneVsOne:
    def __init__(self, predicteurs):
        self.predicteurs = predicteurs
    
    def predict(self, X):
        res = []
        for x in X:
            mat_scores = np.array([f.predict(x.reshape((1,-1))) if f is not None else 0
                                   for f in self.predicteurs.reshape(-1)]).reshape((len(self.predicteurs), -1))
            list_class_score = []
            for i in range(len(self.predicteurs)):
                # la matrice est triangulaire supérieure sans diagonale
                # compte le nombre de vote de notre classe contre les autres
                class_score = - np.sum(mat_scores[i,i+1:])
                # compte le nombre de vote des autres classes contre la notre
                class_score += np.sum(mat_scores[:i,i])
                
                list_class_score.append(class_score)
            res.append(np.argmax(list_class_score))
        return np.array(res)
        
        
def _main_one_vs_one_perso():
    print("one vs one perso")
    predicteurs_simples = np.full((10, 10), None, dtype=object)
    predicteurs_simples_score = np.full((10, 10), np.nan)
    for id_c0 in range(10):
        for id_c1 in range(id_c0 + 1, 10):
            kept_train = np.where((all_train[1] == id_c0) | (all_train[1] == id_c1))[0]
            kept_test = np.where((all_test[1] == id_c0) | (all_test[1] == id_c1))[0]
            train = [all_train[0][kept_train], all_train[1][kept_train]]
            test = [all_test[0][kept_test], all_test[1][kept_test]]
            # met les étiquettes -1/1
            train[1] = np.where(train[1] == id_c0, -1, 1)
            test[1] = np.where(test[1] == id_c0, -1, 1)
        
            svc = LinearSVC(C=1, max_iter=int(1e5))
            svc.fit(train[0], train[1])
            predicteurs_simples[id_c0][id_c1] = svc
            predicteurs_simples_score[id_c0][id_c1] = svc.score(test[0], test[1])
            
    plt.imshow(predicteurs_simples_score)
    plt.colorbar()
    plt.title("One vs One: performances de chaque classifieur simple")
    
    predicteur = PredicteurOneVsOne(predicteurs_simples)
    predicted = predicteur.predict(all_test[0])
    conf = metrics.confusion_matrix(all_test[1], predicted)
    precision = np.sum(np.diagonal(conf))/len(all_test[0])
    conf = conf / np.sum(conf, axis=1)
    print("    linearSVC précision", precision)
    
    plt.figure()
    plt.imshow(conf)
    plt.colorbar()
    plt.title("matrice de confusion du classifieur multiclasse (précision :"+str(round(precision, 3))+")")
    plt.show()

def _main_scikit_learn():
    print("Scikit Learn")
    print("one vs one")
    
    svc = SVC(kernel="linear", gamma="auto").fit(all_train[0], all_train[1])
    print("    SVC linear kernel précision", svc.score(all_test[0], all_test[1]))
    
    svc = SVC(gamma="auto").fit(all_train[0], all_train[1])
    print("    SVC rbf kernel    précision", svc.score(all_test[0], all_test[1]))
    
    print("one vs rest")
    
    svc = LinearSVC(max_iter=int(1e6), multi_class="ovr").fit(all_train[0], all_train[1])
    print("    LinearSVC précision", svc.score(all_test[0], all_test[1]))


def _main_contourf():
    for data_type in [0,1,2]:
        trainx, trainy = arftools.gen_arti(nbex=1000, data_type=data_type, epsilon=1)
        # testx, testy = arftools.gen_arti(nbex=1000, data_type=data_type, epsilon=1)
        param_list = [{'kernel':"linear", 'gamma':"auto"}, {'kernel':"rbf", 'gamma':"auto"}]
        for param in param_list:
            svm = SVC(**param)
            svm.fit(trainx, trainy)
            plt.figure()
            arftools.plot_frontiere(trainx, svm.predict, step=200)
            arftools.plot_data(trainx, trainy)
            plt.title(param["kernel"])
    plt.show()
    
if __name__ == '__main__':
    print("SVM BI CLASSE")
    # _main_biclass()
    print("\nSVM MULTI CLASSE")
    # _main_one_vs_one_perso()
    # _main_scikit_learn()
    _main_contourf()
