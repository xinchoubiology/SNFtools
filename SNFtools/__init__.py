from __future__ import division
import numpy as np

__version__ = '0.0.1'
__author__ = 'xinchou'
__all__ = ["affinityMatrix", "matrixOperate"]


def read_network(network):

    """ read the different data type's network in
        matrix : col -- different feature
                 row -- different patient
    """
    f = open(network, "r")

    feature_list = [x.replace('"', '') for x in f.readline().strip("\n").split(" ")]
    print feature_list

    feature_specture = []
    for line in f.readlines():
        line = line.strip("\n").replace('"', '')            # not necessary
        feature_specture.append([float(x) for x in line.split(" ")])

    return feature_list, np.array(feature_specture)


class OriginData(object):

    """ initialize similarity network fusion method's parameter
        ** parameters **:
            k : mandatory, integer
                number of neighbours, usually [10, 30]
            alpha : mandatory, integer
                    hyper-parameter, usually [0.3, 0.8], for similarity matrix
            t : mandatory, integer
                number of iterations, usually [10, 100]
    """

    def __init__(self, k=10, a=0.5, t=15):
        self.knn = k
        self.alpha = a
        self.iteration = t
        self.origin_dat = []

    def set_network(self, *args):

        """ *args is list of network to be fused
            for item in args:
            read in all network such as genome expression, methylation, epigenetic
        """

        for item in args:
            feature_type, exp_dat = read_network(item)
            self.origin_dat.append(exp_dat)

    @property
    def K(self):
        return self.knn

    @property
    def eps(self):
        return self.alpha

    @property
    def iterate(self):
        return self.iteration


def entropy(x):

    """ calculated entropy of numpy vector x
    """

    classes = np.unique(x)      # the labels number of vector x
    nx = np.shape(x)[0]         # labels length
    nc = np.shape(classes)[0]   # number of classes

    result = 0

    for i in classes:
        try:
            result += (0 - np.size(np.where(x == i))/nx) * np.log10(np.size(np.where(x == i))/nx)
        except ZeroDivisionError:
            print("x has no member!")

    return result, nc





