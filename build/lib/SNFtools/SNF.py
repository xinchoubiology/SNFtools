from __future__ import division
import matrixOperate as mO
import numpy as np

__author__ = 'xinchou'


def normalized(W):

    """ normalized calculated P
    """

    W = W / np.add.reduce(W, axis=1).transpose()

    return W


def SNF(Ws, param):

    """ main function in SNFtools
        input : param -- K : k nearest neighbour
                      +- T : iterations
        return unified similariy matrix

    """

    T = param.iterate

    # Ps, and Ss used for store P^{v} and S^{v} -- initialized
    Ps = []
    Ss = []

    for W in Ws:
        Ss.append(mO.kernelMatrix(W, param))
        Ps.append(mO.statusMatrix(W))

    Ps = np.array(Ps)
    Ss = np.array(Ss)

    for iterate in range(T):
        for idx in range(Ps.__len__()):
            Ps_diag = np.diag(np.diag(Ps[idx]))

            Ps[idx] = np.matrix(Ss[idx]) * np.matrix(np.add.reduce(Ps) - Ps[idx]) / (Ps.__len__() - 1) * np.matrix(Ss[idx]).transpose()
            Ps[idx] = normalized(Ps[idx])

            Ps[idx] = Ps[idx] - np.diag(np.diag(Ps[idx])) + Ps_diag

    P = np.matrix(np.add.reduce(Ps)) / Ps.__len__()

    return (P + P.transpose()) / 2      # ensure affinity matrix is symmetrical

























