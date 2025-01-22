import os
import numpy as np
import pickle
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt



"""
cluster_statistics:     dictionary contains cluster statistical information as follows:
                        'nspots':           int, number of localizations in total
                        'nnoise':           int, number of unclustered localizations
                        'nClusters':        int, number of clusters (including both leaf and non-leaf clusters)
cluster_properties:     dictionary contains perperties of each one of the clusters as follows
                        'nspots':           (nClusters,) int ndarray, number of localizations in each cluster
                        'vol':              (nClusters,) float ndarray, volume of each cluster
                        'cir':              (nClusters,) float ndarray, circularity of each cluster (0 for perfect circle, 1 for line)
                        'mcenter':          (ndim, nClusters) float ndarray, mass centerx of each cluster
"""


            
def _dbscan_kernel(locs, max_epsilon, min_samples, volume, ellip_scale):
    """
    dbscan cluster analysis of the input locs
    INPUTS: 
        locs:               (nspots, ndim) float ndarray, [[locx, locy, locz],...], localizations of the detections 
    
    Clustering parameters:
        max_eps:            float, the maximum epsilon radius that defines the cluster
        min_samples:        int, the minpts for dbscan. Up and down steep regions can't have more than min_samples consecutive non-steep points
        xi_frac:            float, detgermines the minimum steepness on the reachability plot that constitutes a cluster boundary.
                            For example, an upwards point in the reachaility plot is defined by the ratio from one point to its successor being at 1-xi_frac
    Cluster analysis parameters:
        volume:             method to analyze the volume (area if ndim==2) of a cluster
                            'convex':       analyze the volume by the convex hull of a cluster
                            'ellipsoid':    analyze the volume by approximating a cluster to ellipsoid/ellipse
        ellip_scale:        scale for elliptical approximation, only used when volume == 'ellipsoid'
    
    RETURN:
        labels:                 (nspots,) int ndarray, labels of the clusterID of each localization, -1 for non-clustered
        cluster_statistics:     dictionary contains cluster statistical information as follows:
        cluster_properties:     dictionary contains perperties of each one of the clusters as follows
    """
    nspots, ndim = locs.shape
    if ndim not in {2, 3}:
        raise ValueError("only supports 2d or 3d for calculation of the ellipsoid volume") 

    # optics analysis
    db = DBSCAN(eps=max_epsilon, min_samples=min_samples).fit(locs)
    labels = db.labels_
    clusterIDs = set(labels)
    if -1 in clusterIDs:
        clusterIDs.remove(-1)

    nClusters = len(clusterIDs)
    nnoise = list(labels).count(-1)

    cluster_statistics = {  'nspots': nspots, 
                            'nnoise': nnoise, 
                            'nClusters': nClusters  }
    
    nspots_per_cluster = np.zeros(nClusters, dtype=np.int32)
    mcenters_per_cluster = np.zeros((nClusters, ndim))
    vol_per_cluster = np.zeros(nClusters)
    cir_per_cluster = np.zeros(nClusters)
    for i, clusterID in enumerate(clusterIDs):
        
        dumind = labels == clusterID
        nspots_per_cluster[i] = np.sum(dumind)

        loc_per_cluster = locs[dumind]
        pca = PCA(n_components=ndim).fit(loc_per_cluster)
        mcenters_per_cluster[i] = pca.mean_
        cir_per_cluster[i] = 1.0 - np.sqrt(pca.explained_variance_.min() / pca.explained_variance_.max())
        if volume == 'convex':
            hull = ConvexHull(loc_per_cluster)
            vol_per_cluster[i] = hull.volume
        elif volume == 'ellipsoid':
            radius = ellip_scale * np.sqrt(pca.explained_variance_)
            vol_per_cluster[i] = np.pi * radius.prod() if ndim == 2 else 4/3 * np.pi * radius.prod()

    cluster_properties = {  'nspots': nspots_per_cluster, 
                            'vol': vol_per_cluster, 
                            'cir': cir_per_cluster, 
                            'mcenterx': mcenters_per_cluster[:, 0],
                            'mcentery': mcenters_per_cluster[:, 1]  }
    if ndim == 3:
        cluster_properties['mcenterz'] = mcenters_per_cluster[:, 2]
            
    return labels, cluster_statistics, cluster_properties

    

def _dbscan_visual(locs, labels):
    """
    plot the dbscan_cluster analysis results
    INPUT:
        locs:                   [[locx, locy],...], localizations of the detections
        labels:                 dbscan labels for clusterID of each loc
    RETURN:
        dbscan_fig:             matplotlib pyplot object ploting optics cluster analysis
    """
    clusterIDs = set(labels)
    if -1 in clusterIDs:
        clusterIDs.remove(-1)
    
    fig = plt.figure(figsize=(8.5, 11), tight_layout=True)
    ax = fig.add_subplot()
    dumind = labels == -1
    ax.plot(locs[dumind, 0], locs[dumind, 1], 'k.', ms=2, mec='none', alpha=0.3)
    for clusterID in clusterIDs:
        dumind = labels == clusterID
        ax.plot(locs[dumind, 0], locs[dumind, 1], '.', ms=2, mec='none', alpha=0.3)
    
    return fig



def dbscan_batch(sample_fpath, ndim, epsilon=150.0, min_samples=5, volume='convex', ellip_scale=3.0, version='1.11'):
    """
    batch function for runing the _optics_kernel through all the rois in one spool folder
    INPUT:
        sample_fpath:       str, the sample path in wich the spool_fpath/smlm_results stores the _locsnm.pkl file of each of the rois 
        ndim:               int, number of dimensions
    RETURN:
        integrated dictionary for cluster_statistics and cluster_properties
    """
    
    cl_statistics = [{}, {}, {}, {}]
    for i in range(4):
        cl_statistics[i]['roiname']         = np.array([], dtype=str)
        cl_statistics[i]['nspots']          = np.array([], dtype=np.int32)
        cl_statistics[i]['nnoise']          = np.array([], dtype=np.int32)
        cl_statistics[i]['nClusters']       = np.array([], dtype=np.int32)
        
    cl_properties = [{}, {}, {}, {}]
    for i in range(4):
        cl_properties[i]['roiname']         = np.array([], dtype=str)
        cl_properties[i]['nspots']          = np.array([], dtype=np.int32)
        cl_properties[i]['vol']             = np.array([], dtype=np.float64)
        cl_properties[i]['cir']             = np.array([], dtype=np.float64)
        cl_properties[i]['mcenterx']        = np.array([], dtype=np.float64)
        cl_properties[i]['mcentery']        = np.array([], dtype=np.float64)
        if ndim == 3:
            cl_properties[i]['mcenterz']    = np.array([], dtype=np.float64)
    
    # file name collection
    spool_fpaths = [spool_fpath for spool_fpath in os.listdir(sample_fpath) if spool_fpath.startswith('spool_') and os.path.isdir(os.path.join(sample_fpath, spool_fpath))]
    if len(spool_fpaths) == 0:
        print("no spool folders found in {}".format(sample_fpath))
        return cl_statistics, cl_properties

    # dbscan kernels
    for spool_fpath in spool_fpaths:
        
        spool_fpath = os.path.join(sample_fpath, spool_fpath)
        if not os.path.isdir(os.path.join(spool_fpath, 'smlm_result')):
            print("the smlm_result folder is not found in {}".format(spool_fpath))
            continue
        
        spool_fpath = os.path.join(spool_fpath, 'smlm_result')
        smlm_loc_fnames = [smlm_loc_fname for smlm_loc_fname in os.listdir(spool_fpath) if smlm_loc_fname.endswith('_locsnm.pkl')]
        if len(smlm_loc_fnames) == 0:
            print("no _locsnm.pkl files found in {}".format(spool_fpath))
            continue

        for smlm_loc_fname in smlm_loc_fnames:
            
            with open(os.path.join(spool_fpath, smlm_loc_fname), 'rb') as fid:
                smlm_data = pickle.load(fid)
                assert ndim <= smlm_data['ndim'], "ndim mismatch, the input.ndim={}, the smlm_data.ndim={}".format(ndim, smlm_data['ndim'])
                creg = smlm_data['creg'] 
            
            fluorophores = set(creg)
            if -1 in fluorophores:
                fluorophores.remove(-1)
            
            for fluor in fluorophores:
                locs = smlm_data['xvec'][creg==fluor, :ndim] if version=='1.11' else smlm_data['xvec'][:ndim, creg==fluor].T
                    
                labels, cluster_statistics, cluster_properties = _dbscan_kernel(locs, epsilon, min_samples, volume, ellip_scale)
                fig = _dbscan_visual(locs, labels)
                plt.savefig(os.path.join(spool_fpath, smlm_loc_fname[:-11]+'_dbscan_fl{}.png'.format(fluor)), dpi=300)
                plt.close(fig)

                cl_statistics[fluor]['roiname'] = np.hstack([cl_statistics[fluor]['roiname'], smlm_loc_fname[:-11]])
                for key in cluster_statistics.keys():
                    cl_statistics[fluor][key] = np.hstack([cl_statistics[fluor][key], cluster_statistics[key]])
                
                cl_properties[fluor]['roiname'] = np.hstack([cl_properties[fluor]['roiname'], np.tile(smlm_loc_fname[:-11], cluster_statistics['nClusters'])])
                for key in cluster_properties.keys():    
                    cl_properties[fluor][key] = np.hstack([cl_properties[fluor][key], cluster_properties[key]])
    
    return cl_statistics, cl_properties




if __name__ == '__main__':
    
    rng = np.random.default_rng()
    
    nspots_per_cluster = 250
    
    xcenters = [-5.0, 4.0, 1.0, -2.0, 3.0, 5.0]
    ycenters = [-2.0, -1.0, -2.0, 3.0, -2.0, 6.0]
    scales = [0.8, 0.1, 0.2, 0.3, 1.6, 2.0]
    nclusters = len(scales)

    locx = np.zeros((nclusters, nspots_per_cluster))
    locy = np.zeros((nclusters, nspots_per_cluster))
    for i in range(nclusters):
        locx[i] = rng.normal(size=nspots_per_cluster, loc=xcenters[i], scale=scales[i])
        locy[i] = rng.normal(size=nspots_per_cluster, loc=ycenters[i], scale=scales[i])
    locx = locx.flatten()
    locy = locy.flatten()
    locs = np.vstack([locx, locy]).T
    
    labels, cluster_statistics, cluster_properties = _dbscan_kernel(locs, 2.0, 50, 'convex', 3.0)
    print(cluster_statistics)
    for key in cluster_properties.keys():
        print(key)
        print(cluster_properties[key])
    
    fig = _dbscan_visual(locs, labels)
    plt.show()
    
    #optics_spool_batch(2, 'sample', max_epsilon=150, min_samples=25, xi_frac=0.05, volume='convex', ellip_scale=3.0)