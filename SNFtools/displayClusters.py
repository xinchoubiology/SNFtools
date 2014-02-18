__author__ = 'xinchou'

import numpy as np
from numpy.linalg import eig
from numpy.linalg import svd
from matplotlib.pyplot import imshow


def eigDiscresation(eigenM):

    """ for each eigvector's norm-2 == 1
        find the max inner product ~ cos -> 1
    """

    (nrow, ncol) = eigenM.shape
    eigenDiscrete = np.zeros((nrow, ncol))   # M.shape : n sample and k center n * k -> k * n

    for i in range(nrow):
        x = np.array(eigenM[i, :])[0]
        k = np.where(x == np.max(x))[0][0]   # max index

        eigenDiscrete[i, k] = 1

    return np.asmatrix(eigenDiscrete)


def discresation(eigenU, K):

    """ dicresation of top-K eigen-vector of sub-matrix
    """

    eigrow, eigcol = eigenU.shape

    # for center equal K
    R = np.zeros((K, K))

    # random initial status
    R[:, 0] = eigenU[np.random.randint(eigcol), :]

    c = np.zeros((eigrow, 1))
    for i in range(1, K):
        c = c + np.asarray(np.abs(eigenU * np.asmatrix(R[:, i-1]).transpose()))
        inner = np.argsort(np.array(c.transpose()[0]))
        R[:, i] = eigenU[np.where(inner == 0)[0], :]

    # iteration

    iteration = 20
    lastObjectValue = 0

    for i in range(iteration):
        eigDiscrete = eigDiscresation(np.abs(eigenU * R))

        (u, s, v) = svd(eigDiscrete.transpose() * eigenU)

        Ncutval = 2 * (eigrow - np.sum(s))

        if np.abs(Ncutval - lastObjectValue) < np.finfo(float).eps:
            break

        lastObjectValue = Ncutval

        R = v * u.transpose()

    return eigDiscrete



def spectralClustering(W, type=1, K=10):

    """ spectral clustering of matrix
        W -- similarity matrix
        type -- Laplacian Matrix
        K -- number of cluster
    """

    D = np.array(np.add.reduce(np.array(W), axis=1))

    D[np.where(D == 0)] = np.finfo(float).eps
    D = np.diag(D)

    if type == 1:
        DL = D - W
    elif type == 2:
        Di = np.asmatrix(D).I
        DL = np.inner(Di, D - W)
    else:
        Di = np.asmatrix(np.sqrt(D)).I
        DL = np.inner(np.inner(Di, D - W), Di)

    # pre-defied parameter
    # nrow, ncol = DL.shape

    value, vector = eig(np.asmatrix(DL))    # matrix consistance

    U = vector[:, np.argsort(value)[:K]]

    eigDiscrete = discresation(U, K)

    labels = np.array(np.where(eigDiscrete == np.max(eigDiscrete))[1])[0]

    return labels, eigDiscrete



def displayClusters(W, type, K):

    """ input is W is unified similarity network after SNF
    """

    # defined Laplacian Matrix cliusteirng
    res, eigDiscrete = spectralClustering(W, type, K)


    index = np.argsort(res)
    nrow, ncol = W.shape

    disp_W = np.zeros(W.shape)

    for i in range(nrow):
        for j in range(ncol):
            x, y = index[i], index[j]
            disp_W[i, j] = W[x, y]

    disp_W = disp_W - np.diag(np.diag(disp_W)) + np.eye(W.shape[0]) * np.max(disp_W - np.diag(np.diag(disp_W)))

    imshow(disp_W)

    return [disp_W, index, res, eigDiscrete]

