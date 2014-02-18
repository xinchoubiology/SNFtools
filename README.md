Similarity Network Fusion method By Python --- nmeth

# Example

---
import SDFtools

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

disp_W, index, cluster, eigDiscrete = dispCluster.displayClusters(P, 2, 2)

imshow(P)

---
#Install

python setup.py build

python setup.py install

