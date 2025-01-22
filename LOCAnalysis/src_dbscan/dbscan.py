import os
import inspect
import ctypes
import numpy as np
import numpy.ctypeslib as ctl


def dbscan(locs, epsilon, minpts):
    """
    dbscan cluster analysis of the input locs
    dependent on dbscan.dll in the same pacakge
    INPUT:
        locs:               [[locx], [locy], [locz]], localizations of the detections 
        epsilon:            the epsilon radius that defines the cluster
        minpts:             the number of minimal points that defines the cluster                      
    RETURN:
        classification:     [nspots] int, the labels of the cluster_id of each localization in locs
                            0:      UNCLASSIFIED
                            -1:     NOISE
                            1~n:    cluster_idx
        iscore:             [nspots] bool, the labels of the core points of the locs
                            False:  Non-core points
                            True:   core points
    """
    ndim = len(locs)
    nspots = len(locs[0])

    locs = np.ascontiguousarray(np.array(locs).ravel(), dtype=np.float32)
    classification = np.ascontiguousarray(np.zeros(nspots), dtype=np.int32)
    iscore = np.ascontiguousarray(np.zeros(nspots), dtype=bool)
    
    # call c libraries
    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fname = 'dbscan.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    dbscan_kernel = lib.dbscan
    dbscan_kernel.argtypes = [ctypes.c_int32, ctypes.c_int32,
                                ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                                ctypes.c_float, ctypes.c_int32,
                                ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),
                                ctl.ndpointer(bool, flags="aligned, c_contiguous")]
    dbscan_kernel(ndim, nspots, locs, epsilon, minpts, classification, iscore)
    return classification, iscore



if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler
    
    from sklearn.cluster import DBSCAN
    from sklearn import metrics

    from matplotlib import pyplot as plt
    import time

    # simulations
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=131072, centers=centers, cluster_std=0.3, random_state=0)
    X = StandardScaler().fit_transform(X)

    epsilon = 0.2
    minpts = 10

    # sklearn dbscan
    tas = time.time() 
    db = DBSCAN(eps=epsilon, min_samples=minpts).fit(X)
    labels = db.labels_
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    print(time.time()-tas)
    n_clusters_sklearn = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_sklearn = list(labels).count(-1)

    
    # cpp
    tas = time.time()
    classification, iscore = dbscan(X.T, epsilon, minpts)
    print(time.time()-tas)
    n_clusters_cpp = len(set(classification)) - (1 if -1 in labels else 0)
    n_noise_cpp = list(classification).count(-1)
    

    # compare
    print("Estimated number of clusters: sklearn-{n1:d}, mycpp-{n2:d}".format(n1=n_clusters_sklearn, n2=n_clusters_cpp))
    print("Estimated number of noise points: sklearn-{n1:d}, mycpp-{n2:d}".format(n1=n_noise_sklearn, n2=n_noise_cpp))

    fig, (ax0, ax1) = plt.subplots(1, 2)
    
    # sklearn
    unique_labels_sklearn = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels_sklearn))]
    for k, col in zip(unique_labels_sklearn, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask & core_samples_mask]
        ax0.plot(xy[:, 0], xy[:, 1], 'o', mfc=tuple(col), mec="k", markersize=6)

        xy = X[class_member_mask & ~core_samples_mask]
        ax0.plot(xy[:, 0], xy[:, 1], 'o', mfc=tuple(col), mec="k", markersize=2)
    ax0.set_title("Estimated number of clusters: {}".format(n_clusters_sklearn))

    # mycpp
    unique_labels_cpp = set(classification)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels_cpp))]
    for k, col in zip(unique_labels_cpp, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = classification == k

        xy = X[class_member_mask & iscore]
        ax1.plot(xy[:, 0], xy[:, 1], 'o', mfc=tuple(col), mec="k", markersize=6)

        xy = X[class_member_mask & ~iscore]
        ax1.plot(xy[:, 0], xy[:, 1], 'o', mfc=tuple(col), mec="k", markersize=2)
    ax1.set_title("Estimated number of clusters: {}".format(n_clusters_cpp))
    
    plt.show()