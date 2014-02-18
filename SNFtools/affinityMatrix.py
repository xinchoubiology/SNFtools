__author__ = 'xinchou'
import numpy as np

Machine_Epsilon = np.finfo(float).eps

def standardNormalization(origin_d):

    """ normalization each column of origin_d to
        have mean = 0 and standard deviation = 1
        standn = (x - mean / std)
    """
    normal_dat = []
    for data in origin_d:
        mean = np.mean(data, axis=0)
        sd = np.std(data, axis=0)
        sd[sd == 0] = 1.0
        normal_dat.append(np.asmatrix((data - mean)/sd))

    return np.array(normal_dat)


# use decorator to defined different distance function

def dist2matrix(func):

    """ from origin_data to pair-wised disatance matrix
        n sample x n sample
    """
    def __wrapper(normal_dat):

        w = []
        for data in normal_dat:
            dist_matrix = func(np.asmatrix(data))
            w.append(dist_matrix)

        return np.array(w)

    return __wrapper


@dist2matrix
def euclidDist(x):

    """ calculate \sum(x_{i}{k}-x_{j}{k})^2
    """

    sumsqX = np.add.reduce(np.multiply(x, x), axis=1)
    square_x = sumsqX + sumsqX.transpose()

    product_x = 2 * x * x.transpose()

    return square_x - product_x


@dist2matrix
def manhatDist(x):

    """ calculated \sum(x_{i}{k}-x_{j}{k})
    """
    sumsqX = np.add.reduce(x, axis=1)

    return np.abs(sumsqX - sumsqX.transpose())


def affinityMatrix(Diff, param):

    """ similarity matrix generated
        w(i,j) = exp(-dist(xi, xj) / mu / eps)
        eps = (mean(D(xi, Ni)) + mean(D(xj, Nj)) + Dist(xi, xj)) / 3

    """

    Diff_mat = (Diff + Diff.transpose()) / 2            # the distance pair-wised matrix


    Diff_mat_sort = Diff_mat - np.diag(np.diag(Diff_mat))    # set Diff's diagonal to 0
    Diff_mat_sort = np.sort(Diff_mat_sort, axis=0)

    # calculated the epsilon
    K_dist = np.mean(Diff_mat_sort[:param.K], axis=0)
    epsilon = (K_dist + K_dist.transpose()) / 3 * 2 + Diff_mat / 3 + Machine_Epsilon

    W = np.exp(-(Diff_mat / (param.eps * epsilon)))

    return (W + W.transpose()) / 2


















