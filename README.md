# SNFtools

similarity network fusion tools transplanted from nmeth

Similarity Networks Fusion Tools

SNFtools is a Library for SNF and samples' labels prediction

import SNFtools

K = 20
alpha = 0.5
t = 20

SNFmodel = SNFtools.OriginData(K, alpha, t)

SNFmodel.set_network('Data1.dat', 'Data2.dat')

import SNFtools.affinityMatrix as affinityMatrix

dist2_dat = affinityMatrix.euclidDist(SNFmodel.origin_dat)  # calculate out sample-sample distance

Walls = [ affinityMatrix.affinityMatrix(Diff, SNFmodel) for Diff in dist2_dat ]

import SNFtools.matrixOperate as matrixOperate
import SNFtools.displayClusters as dispCluster
import SNFtools.SNF as SNF

P = SNF.SNF(Walls, SNFmodel)

disp_W, index, cluster, eigDiscrete = dispCluster.displayClusters(P, 2, 4)

imshow(disp_W)

#####################
K = 10
W = P
D = np.array(np.add.reduce(np.array(W), axis=1))
D[np.where(D == 0)] = np.finfo(float).eps
D = np.diag(D)
Di = np.asmatrix(D).I
DL = np.inner(Di, D - W)
value, vector = eig(np.asmatrix(DL))
P
eigenU = U
eigrow, eigcol = eigenU.shape
K = 10
R = np.zeros((K, K))
R[:, 0] = eigenU[np.random.randint(eigcol), :]
c = np.zeros((eigrow, 1))
for i in range(1, K):
        c = c + np.asarray(np.abs(eigenU * np.asmatrix(R[:, i-1]).transpose()))
        inner = np.argsort(np.array(c.transpose()[0]))
        R[:, i] = eigenU[np.where(inner == 0)[0], :]


eigDiscrete = dispCluster.eigDiscresation(np.abs(eigenU * R))

(u, s, v) = svd(eigDiscrete.transpose() * eigenU)






