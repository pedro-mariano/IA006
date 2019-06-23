import numpy as np
from collections import Counter

# k-nearest neighbors

# Entradas:

# k: numero de vizinhos proximos
# X_train: atributos dos dados de treinamento
# Y_train: classe associada a cada amostra de treinamento (valor inteiro, comecando em 0)
# X_test: atributos dos dados de teste

# Retorna a predicao da classe para cada amostra de teste

def k_nearest_neighbors(k,X_train,Y_train, X_test):
# Calculo das distancias

    N_train = len(X_train)
    N_test = len(X_test)
    Nclasses = int(Y_train.max()) + 1

    # dist(i,j) = distancia entre a i-esima amostra de teste e a j-esima amostra de treinamento
    dist = np.zeros((N_test,N_train)) 

    for i in np.arange(N_test):
        for j in np.arange(N_train):
            # Distancia euclidiana
            dist[i,j] = np.sqrt(np.sum((X_test[i] - X_train[j])**2))

    ind_neigh = np.argsort(dist,axis=1) # indices dos vizinhos em ordem crescente de distancia
    dist_neigh = np.array([dist[i,ind_neigh[i]] for i in range(N_test)]) # distancias ordenadas

    classes_neigh = Y_train[ind_neigh] # classes dos vizinhos mais proximos
    
    N_test = classes_neigh.shape[0]
    k_classes = classes_neigh[:,:k] # classes dos k-vizinhos mais proximos a cada amostra
    k_dist = dist_neigh[:,:k] # distancias dos k-vizinhos mais proximos a cada amostra
    
    votes = np.zeros((N_test,Nclasses)) # votos das classes para cada amostra
    
    # soma de votos para cada classe
    for c in np.arange(Nclasses):
        votes[:,c] = votes[:,c] + ((k_classes == c)/(k_dist+1)).sum(axis=1) # votos ponderados pelo inverso da distancia+1
    
    # retorna classe com maior voto para cada amostra
    return votes.argmax(axis=1)

# Funcao de avaliacao do classificador

# Entradas:

# decision: predicao do classificador para cada amostra de teste (valor inteiro, comecando em 0)
# Y_test: classe associada a cada amostra de teste (valor inteiro, comecando em 0)
# Nclasses: numero de classes

# Retorna matriz de confusao, precision, recall e f1-medida

def eval_classif(decision,Y_test,Nclasses):

    conf = np.zeros((Nclasses,Nclasses),dtype=int) # Matriz de confusao

    for i in np.arange(Nclasses):
        c = Counter(decision[Y_test == i].tolist())
        for j in np.arange(Nclasses):
            conf[i,j] = c[j]

    # Avaliacao global

    precision = 0.0
    recall = 0.0
    for i in np.arange(Nclasses):
        TP = conf[i,i]
        FN = sum(conf[i,:]) - TP
        FP = sum(conf[:,i]) - TP
        precision = precision + float(TP) / (TP + FP)
        recall = recall + float(TP) / (TP + FN)

    precision = precision / Nclasses
    recall = recall / Nclasses

    f1 = 2*precision*recall/(precision+recall)
    
    return conf, precision, recall, f1
