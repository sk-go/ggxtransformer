"""
chroma/messung/verschachtelung.py
Zweitmaß (c): Lässt sich aus der Gleichartigkeit der Anlage eine Ordnung von Gattung und Art
(ein porphyrischer Baum) rekonstruieren? Gemessen als kophenetische Korrelation.
"""

import numpy as np
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import pdist


def kophenetische_korrelation(E):
    d = pdist(np.asarray(E, dtype=np.float64), metric="cosine")
    c, _ = cophenet(linkage(d, method="average"), d)
    return float(c)


def cluster_labels(E, n_cluster):
    d = pdist(np.asarray(E, dtype=np.float64), metric="cosine")
    return fcluster(linkage(d, method="average"), n_cluster, criterion="maxclust")