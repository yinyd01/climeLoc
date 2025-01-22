import os
import numpy as np
import pickle
from scipy.spatial import ConvexHull
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
from matplotlib import gridspec
from matplotlib import pyplot as plt



"""
cluster_statistics:     dictionary contains cluster statistical information as follows:
                        'nspots':           int, number of localizations in total
                        'nnoise':           int, number of unclustered localizations
                        'nClusters':        int, number of clusters (including both leaf and non-leaf clusters)
                        'nLeafClusters':    int, number of leaf clusters
cluster_properties:     dictionary contains perperties of each one of the clusters as follows
                        'isleaf':           (nClusters,) bool array, label if the cluster is a leaf cluster
                        'nspots':           (nClusters,) int ndarray, number of localizations in each cluster
                        'vol':              (nClusters,) float ndarray, volume of each cluster
                        'cir':              (nClusters,) float ndarray, circularity of each cluster (0 for perfect circle, 1 for line)
                        'mcenter':          (ndim, nClusters) float ndarray, mass centerx of each cluster
cluster_reachability:   dictionary contains the reachability information from optics analysis
                        'ordering':         (nspots,) int ndarray, the ordering of the optics
                        'reachability':     (nspots,) float ndarray, the reachability distance of each localization
                        'hierarchy_segs':   (nCluster, 2) int ndarray, [[start, end],...] the starts and ends of the ordered localization index of each cluster (inclusive for starts and ends)
                        'isleaf'"           (nCluster,) bool array, label if the the hierarchy seg is a leaf cluster (same as in cluster_properties)
"""


def _hierarchy_analysis(hierarchy_segs, nspots):
    """
    sort the hierarchy_segs according to the starts
    and label the hierarchy segs if the seg is a leaf cluster (seg not intersected by others)
    INPUT:  
        hierarchy_segs:         (nCluster, 2) int ndarray, [[start, end],...] the starts and ends of the ordered localization index of each cluster (inclusive for starts and ends)
        nspots:                 int, number of spots in total
    RETURN: 
        hierarchy_segs_sorted:  (nCluster, 2) int ndarray, sorted hierarchy_segs
        leaflabel:              (nCluster,) bool arra, Ture if a seg is leaf
        nnoise:                 int, number of noise points
    """
    hierarchy_segs_sorted = hierarchy_segs[hierarchy_segs[:, 0].argsort()]
    nsegs = len(hierarchy_segs_sorted)
    starts = hierarchy_segs_sorted[:, 0]
    ends = hierarchy_segs_sorted[:, 1]

    # hierarchy analysis
    previous_end_min = ends[0]
    previous_end_max = ends[0]

    labels = np.zeros(nsegs, dtype=bool)
    nnoise = starts[0]
    j = 0
    for i in range(nsegs):
        
        if starts[i] < previous_end_min:
            if ends[i] < previous_end_min:
                j = i
                previous_end_min = ends[i]
            if ends[i] >= previous_end_max:
                previous_end_max = ends[i]    
            continue
        
        labels[j] = True
        j = i
        previous_end_min = ends[i]
        
        if starts[i] > previous_end_max:
            nnoise += starts[i] - previous_end_max - 1
            previous_end_max = ends[i]
    
    labels[j] = True # mark the last one
    nnoise += nspots - 1 - previous_end_max
    
    return hierarchy_segs_sorted, labels, nnoise
            

            
def _optics_kernel(locs, max_epsilon, min_samples, xi_frac, volume, ellip_scale):
    """
    optics cluster analysis of the input locs
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
        cluster_statistics:     dictionary contains cluster statistical information as follows:
        cluster_properties:     dictionary contains perperties of each one of the clusters as follows
        cluster_reachability:   dictionary contains the reachability information from optics analysis
    """
    nspots, ndim = locs.shape
    if ndim not in {2, 3}:
        raise ValueError("only supports 2d or 3d for calculation of the ellipsoid volume")

    # optics analysis
    clust = OPTICS(min_samples=min_samples, max_eps=max_epsilon, cluster_method='xi', xi=xi_frac)
    clust.fit(locs)
    locs_sorted = locs[clust.ordering_]
    
    # cluster analysis
    hierarchy_segs, isleaf, nnoise = _hierarchy_analysis(clust.cluster_hierarchy_, nspots)
    nClusters = len(hierarchy_segs)
    cluster_statistics = {  'nspots': nspots, 
                            'nnoise': nnoise, 
                            'nClusters': nClusters, 
                            'nLeafClusters': np.sum(isleaf)  }
    
    nspots_per_cluster = np.zeros(nClusters, dtype=np.int32)
    mcenters_per_cluster = np.zeros((nClusters, ndim))
    vol_per_cluster = np.zeros(nClusters)
    cir_per_cluster = np.zeros(nClusters)
    for i in range(nClusters):
        
        nspots_per_cluster[i] = hierarchy_segs[i][1] - hierarchy_segs[i][0] + 1
        loc_per_cluster = locs_sorted[hierarchy_segs[i][0] : hierarchy_segs[i][1] + 1]
        
        pca = PCA(n_components=ndim).fit(loc_per_cluster)
        mcenters_per_cluster[i] = pca.mean_
        cir_per_cluster[i] = 1.0 - np.sqrt(pca.explained_variance_.min() / pca.explained_variance_.max())
        if volume == 'convex':
            hull = ConvexHull(loc_per_cluster)
            vol_per_cluster[i] = hull.volume
        elif volume == 'ellipsoid':
            radius = ellip_scale * np.sqrt(pca.explained_variance_)
            vol_per_cluster[i] = np.pi * radius.prod() if ndim == 2 else 4/3 * np.pi * radius.prod()

    cluster_properties = {  'isleaf': isleaf, 
                            'nspots': nspots_per_cluster, 
                            'vol': vol_per_cluster, 
                            'cir': cir_per_cluster, 
                            'mcenterx': mcenters_per_cluster[:, 0],
                            'mcentery': mcenters_per_cluster[:, 1]}
    if ndim == 3:
        cluster_properties['mcenterz'] = mcenters_per_cluster[:, 2]
    
    cluster_reachability = {'ordering': clust.ordering_, 
                            'reachability': clust.reachability_, 
                            'hierarchy': hierarchy_segs, 
                            'isleaf': isleaf }
            
    return cluster_statistics, cluster_properties, cluster_reachability

    

def _optics_visual(locs, cluster_reachability):
    """
    plot the optics_cluster analysis results
    INPUT:
        locs:                   [[locx, locy],...], localizations of the detections
                                only supports for 2d localizations
        cluster_reachability:   dictionary contains the reachability information from optics analysis
    RETURN:
        optics_fig:             matplotlib pyplot object ploting optics cluster analysis
    """
    
    ordering = cluster_reachability['ordering']
    reachability = cluster_reachability['reachability'][ordering]
    cluster_segs = cluster_reachability['hierarchy']
    isleaf = cluster_reachability['isleaf']
    
    nspots = len(locs)
    nClusters = len(cluster_segs)
    locx = locs[ordering, 0]
    locy = locs[ordering, 1]

    xspace = np.arange(nspots)
    if np.all(isleaf):
        
        fig = plt.figure(figsize=(6, 8), tight_layout=True)
        gs = gridspec.GridSpec(4, 1)
        ax0 = fig.add_subplot(gs[0,  0])
        ax1 = fig.add_subplot(gs[1:4, 0])

        inactive_labels = np.ones(nspots, dtype=bool)
        for i in range(nClusters):
            inactive_labels[cluster_segs[i][0] : cluster_segs[i][1] + 1] = False
        ax0.plot(xspace[inactive_labels], reachability[inactive_labels], 'k.', ms=4, mec='none', alpha=0.3)
        ax1.plot(locx[inactive_labels], locy[inactive_labels], 'k.', ms=4, mec='none', alpha=0.1)
        
        for i in range(nClusters):
            ax0.plot(xspace[cluster_segs[i][0]:cluster_segs[i][1]+1], reachability[cluster_segs[i][0]:cluster_segs[i][1]+1], '.', ms=4, mec='none', alpha=0.3)
            ax1.plot(locx[cluster_segs[i][0]:cluster_segs[i][1]+1], locy[cluster_segs[i][0]:cluster_segs[i][1]+1], '.', ms=4, mec='none', alpha=0.3) 
        ax0.set_ylabel("Reachability (epsilon distance)")
        ax0.set_title("Reachability Plot")
        ax1.set_title("Automatic Clustering")

    else:
        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        gs = gridspec.GridSpec(4, 2)
        ax0 = fig.add_subplot(gs[0,  0])
        ax1 = fig.add_subplot(gs[1:4, 0])
        ax2 = fig.add_subplot(gs[0,  1])
        ax3 = fig.add_subplot(gs[1:4, 1])
        
        # plot leaf clusters
        inactive_labels = np.ones(nspots, dtype=bool)
        for i in range(nClusters):
            if isleaf[i]:
                inactive_labels[cluster_segs[i][0] : cluster_segs[i][1] + 1] = False
        ax0.plot(xspace[inactive_labels], reachability[inactive_labels], 'k.', ms=4, mec='none', alpha=0.3)
        ax1.plot(locx[inactive_labels], locy[inactive_labels], 'k.', ms=4, mec='none', alpha=0.1)
        
        for i in range(nClusters):
            if isleaf[i]: 
                ax0.plot(xspace[cluster_segs[i][0]:cluster_segs[i][1]+1], reachability[cluster_segs[i][0]:cluster_segs[i][1]+1], '.', ms=4, mec='none', alpha=0.3)
                ax1.plot(locx[cluster_segs[i][0]:cluster_segs[i][1]+1], locy[cluster_segs[i][0]:cluster_segs[i][1]+1], '.', ms=4, mec='none', alpha=0.3)
        ax0.set_ylabel("Reachability (nm)")
        ax0.set_title("Reachability Plot")
        ax1.set_title("Automatic Clustering (Leaf Clusters)")

        # plot non-leaf clusters
        inactive_labels = np.ones(nspots, dtype=bool)
        for i in range(nClusters):
            if not isleaf[i]:
                inactive_labels[cluster_segs[i][0] : cluster_segs[i][1] + 1] = False
        ax2.plot(xspace[inactive_labels], reachability[inactive_labels], 'k.', ms=4, mec='none', alpha=0.3)
        ax3.plot(locx[inactive_labels], locy[inactive_labels], 'k.', ms=4, mec='none', alpha=0.1)

        for i in range(nClusters):
            if not isleaf[i]:
                ax2.plot(xspace[cluster_segs[i][0]:cluster_segs[i][1]+1], reachability[cluster_segs[i][0]:cluster_segs[i][1]+1], '.', ms=4, mec='none', alpha=0.3)
                ax3.plot(locx[cluster_segs[i][0]:cluster_segs[i][1]+1], locy[cluster_segs[i][0]:cluster_segs[i][1]+1], '.', ms=4, mec='none', alpha=0.3)
        ax2.set_ylabel("Reachability (nm)")
        ax2.set_title("Reachability Plot")
        ax3.set_title("Automatic Clustering (non-Leaf Clusters)")
    
    return fig



def optics_batch(sample_fpath, ndim, max_epsilon=np.inf, min_samples=5, xi_frac=0.05, volume='convex', ellip_scale=3.0, version='1.11'):
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
        cl_statistics[i]['nLeafClusters']   = np.array([], dtype=np.int32)
    
    cl_properties = [{}, {}, {}, {}]
    for i in range(4):
        cl_properties[i]['roiname']         = np.array([], dtype=str)
        cl_properties[i]['isleaf']          = np.array([], dtype=bool)
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

    # optic kernels
    for spool_fpath in spool_fpaths:
        
        spool_fpath = os.path.join(sample_fpath, spool_fpath)
        if not os.path.isdir(os.path.join(spool_fpath, 'smlm_result')):
            print("the smlm_result folder is not found in {}".format(spool_fpath))
            continue
        
        spool_fpath = os.path.join(spool_fpath, 'smlm_result')
        smlm_loc_fnames = [smlm_loc_fname for smlm_loc_fname in os.listdir(spool_fpath) if smlm_loc_fname.endswith('_locsnm.pkl')]
        if len(smlm_loc_fnames) == 0:
            print("no _locs.pkl files found in {}".format(spool_fpath))
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
                    
                cluster_statistics, cluster_properties, cluster_reachability = _optics_kernel(locs, max_epsilon, min_samples, xi_frac, volume, ellip_scale)
                fig = _optics_visual(locs[:, :2], cluster_reachability)
                plt.savefig(os.path.join(spool_fpath, smlm_loc_fname[:-11]+'_optics_fl{}.png'.format(fluor)), dpi=300)
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
    
    cluster_statistics, cluster_properties, cluster_reachability = _optics_kernel(locs, 2.0, 50, 0.05, 'convex', 3.0)
    print(cluster_statistics)
    for key in cluster_properties.keys():
        print(key)
        print(cluster_properties[key])
    
    fig = _optics_visual(locs, cluster_reachability)
    plt.show()
    
    #optics_spool_batch(2, 'sample', max_epsilon=150, min_samples=25, xi_frac=0.05, volume='convex', ellip_scale=3.0)