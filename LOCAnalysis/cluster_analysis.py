import os
import pickle
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import gridspec
from matplotlib import pyplot as plt


suffix_0 = '_properties.csv'
suffix_1 = '_statistics.csv'
nbins = 64


def _textPos(ax, relativePos):
    """
    Return the absolute position of putting the text into a matplotlib figure
    INPUT:
        ax:                 the aixs handle
        relativePos:        (float, float) the relative position where to put the text for x- and y- axis
                            (i.e. 0.0 on the left(bottom), 0.5 in the middle, 1.0 on the right(top))
    RETURN:
        posx, posy:         the absolut coordinate in the axis
    """
    posx = relativePos[0]*(ax.get_xlim()[1]-ax.get_xlim()[0]) + ax.get_xlim()[0]
    posy = relativePos[1]*(ax.get_ylim()[1]-ax.get_ylim()[0]) + ax.get_ylim()[0]
    return posx, posy



def _exp1func(xdata, Amp, xc):
    return Amp * np.exp(-xdata / xc)



def _exp1fit(xdata, ydata):
    """
    fit ydata = Amp * exp((xdata - x0)/xc)
    xdata should be sorted
    RETURN:
        xc:                     the xdata value that ydata_predicted decreased to 1/e
        xc_std:                 std of the fitted xc
        x0:                     x-offset for exp1 fit
    """
    ind, = np.where(ydata > 0)
    ind_offset = ind[0]
    xdata_offset = xdata[ind_offset]
    
    zdata = xdata[ind_offset:] - xdata_offset
    p0 = [ydata[ind_offset], np.sum(xdata*ydata)/np.sum(ydata)]
    popt, pcov = curve_fit(_exp1func, zdata, ydata[ind_offset:], p0=p0)
    pstd = np.sqrt(np.diag(pcov))

    ydata_predicted = np.zeros(len(ydata))
    ydata_predicted[:ind_offset] = 0.0
    ydata_predicted[ind_offset:] = _exp1func(zdata, popt[0], popt[1])

    return popt[1], pstd[1], xdata_offset, ydata_predicted



def cluster_properties_plot(fpath, kws):
    
    fnames = [fname for fname in os.listdir(fpath) if fname.endswith('_properties.csv')]
    fnames_sort = []
    kws_sort = []
    for kw in kws:
        for fname in fnames:
            if kw in fname:
                fnames_sort.append(fname)
                kws_sort.append(kw) 

    nspots_bins = np.linspace(0, 200, nbins+1, endpoint=True)
    vol_bins = np.linspace(0, 360000, nbins+1, endpoint=True)
    cir_bins = np.linspace(0.0, 1.0, nbins+1, endpoint=True)

    nspots_hist_result = {'bins':nspots_bins[:-1]}
    nspots_hist_predicted = {'bins':nspots_bins[:-1]}
    nspots_fit_results = np.zeros((len(fnames_sort), 3))

    vol_hist_result = {'bins':vol_bins[:-1]}
    vol_hist_predicted = {'bins':vol_bins[:-1]}
    vol_fit_results = np.zeros((len(fnames_sort), 3))
    
    fig, axs = plt.subplots(nrows=len(fnames_sort), ncols=3, figsize=(8.5, 11))
    for i, fname in enumerate(fnames_sort):
        
        fname = os.path.join(fpath, fname)
        cluster_properties = pd.read_csv(fname)
        isleaf = np.array(cluster_properties['isleaf'], dtype=bool)
        
        nspots = np.array(cluster_properties['nspots'])
        nspots_counts = np.histogram(nspots, nspots_bins)[0]
        nspots_avg, nspots_std, nspots_offset, nspots_counts_predicted = _exp1fit(nspots_bins[:-1], nspots_counts)
        nspots_hist_result[kws_sort[i]] = nspots_counts
        nspots_hist_predicted[kws_sort[i]] = nspots_counts_predicted
        nspots_fit_results[i] = np.array([nspots_avg, nspots_std, nspots_offset])

        vol = np.array(cluster_properties['vol'])
        vol_counts = np.histogram(vol, vol_bins)[0]
        vol_avg, vol_std, vol_offset, vol_counts_predicted = _exp1fit(vol_bins[:-1], vol_counts)
        vol_hist_result[kws_sort[i]] = vol_counts
        vol_hist_predicted[kws_sort[i]] = vol_counts_predicted
        vol_fit_results[i] = np.array([vol_avg, vol_std, vol_offset])

        cir = np.array(cluster_properties['cir'])
        cir_counts = np.histogram(cir, cir_bins)[0]
        
        if len(fnames_sort) == 1:
            axs[0].bar(nspots_bins[:-1], nspots_counts, width=nspots_bins[1]-nspots_bins[0], align='edge')
            axs[0].plot(nspots_bins[:-1], nspots_counts_predicted, color='tab:red')
            txtPosx, txtPosy = _textPos(axs[0], (0.45, 0.75))
            axs[0].text(txtPosx, txtPosy, 'AVG={AVG:.4f}\nSTD={STD:.4f}'.format(AVG=nspots_avg, STD=nspots_std))
            axs[0].set_title('nspots_per_cluster')
            axs[0].set_ylabel(kws_sort[i])
        else:
            axs[i, 0].bar(nspots_bins[:-1], nspots_counts, width=nspots_bins[1]-nspots_bins[0], align='edge')
            axs[i, 0].plot(nspots_bins[:-1], nspots_counts_predicted, color='tab:red')
            txtPosx, txtPosy = _textPos(axs[i, 0], (0.45, 0.75))
            axs[i, 0].text(txtPosx, txtPosy, 'AVG={AVG:.4f}\nSTD={STD:.4f}'.format(AVG=nspots_avg, STD=nspots_std))
            axs[i, 0].set_ylabel(kws_sort[i])
            if i == 0:
                axs[i, 0].set_title('nspots_per_cluster')

        if len(fnames_sort) == 1:
            axs[1].bar(vol_bins[:-1], vol_counts, width=vol_bins[1]-vol_bins[0], align='edge')
            axs[1].plot(vol_bins[:-1], vol_counts_predicted, color='tab:red')
            txtPosx, txtPosy = _textPos(axs[1], (0.45, 0.75))
            axs[1].text(txtPosx, txtPosy, 'AVG={AVG:.4f}\nSTD={STD:.4f}'.format(AVG=vol_avg, STD=vol_std))
            axs[1].set_title('vol_per_cluster')
            axs[1].set_ylabel(kws_sort[i])
        else:
            axs[i, 1].bar(vol_bins[:-1], vol_counts, width=vol_bins[1]-vol_bins[0], align='edge')
            axs[i, 1].plot(vol_bins[:-1], vol_counts_predicted, color='tab:red')
            txtPosx, txtPosy = _textPos(axs[i, 1], (0.45, 0.75))
            axs[i, 1].text(txtPosx, txtPosy, 'AVG={AVG:.4f}\nSTD={STD:.4f}'.format(AVG=vol_avg, STD=vol_std))
            axs[i, 1].set_ylabel(kws_sort[i])
            if i == 0:
                axs[i, 1].set_title('vol_per_cluster')

        if len(fnames_sort) == 1:
            axs[2].bar(cir_bins[:-1], cir_counts, width=cir_bins[1]-cir_bins[0], align='edge')
            txtPosx, txtPosy = _textPos(axs[2], (0.65, 0.75))
            axs[2].text(txtPosx, txtPosy, 'AVG={AVG:.4f}\nSTD={STD:.4f}'.format(AVG=cir.mean(), STD=cir.std()))
            axs[2].set_title('cir_per_cluster')
            axs[2].set_ylabel(kws_sort[i])
        else:
            axs[i, 2].bar(cir_bins[:-1], cir_counts, width=cir_bins[1]-cir_bins[0], align='edge')
            txtPosx, txtPosy = _textPos(axs[i, 2], (0.65, 0.75))
            axs[i, 2].text(txtPosx, txtPosy, 'AVG={AVG:.4f}\nSTD={STD:.4f}'.format(AVG=cir.mean(), STD=cir.std()))
            axs[i, 2].set_ylabel(kws_sort[i])
            if i == 0:
                axs[i, 2].set_title('cir_per_cluster')
    
    plt.savefig(os.path.join(fpath, 'cluster_properties.png'), dpi=300)
    
    nspots_fit_results = nspots_fit_results.T
    vol_fit_results = vol_fit_results.T
    df = pd.DataFrame({ 'kw':               kws_sort, 
                        'nspots_avg':       nspots_fit_results[0], 
                        'nspots_std':       nspots_fit_results[1], 
                        'nspots_offset':    nspots_fit_results[2],
                        'vol_avg':          vol_fit_results[0],
                        'vol_std':          vol_fit_results[1],
                        'vol_offset':       vol_fit_results[2]      })
    df.to_csv(os.path.join(fpath, 'cluster_propsum.csv'))
        
    df = pd.DataFrame(nspots_hist_result)
    df.to_csv(os.path.join(fpath, 'nspots_per_cluster_hist.csv'))

    df = pd.DataFrame(nspots_hist_predicted)
    df.to_csv(os.path.join(fpath, 'nspots_per_cluster_histfit.csv'))

    df = pd.DataFrame(vol_hist_result)
    df.to_csv(os.path.join(fpath, 'vol_per_cluster_hist.csv'))

    df = pd.DataFrame(vol_hist_predicted)
    df.to_csv(os.path.join(fpath, 'vol_per_cluster_histfit.csv'))




def cluster_statistics_summ(fpath, kws):
    
    fnames = [fname for fname in os.listdir(fpath) if fname.endswith('_statistics.csv')]
    fnames_sort = []
    kws_sort = []
    for kw in kws:
        for fname in fnames:
            if kw in fname:
                fnames_sort.append(fname)
                kws_sort.append(kw) 
    
    nspots_summ = dict()
    nnoise_summ = dict()
    cluster_rate = dict()

    nClusters_summ = dict()
    nLeafClusters_summ = dict()
    hirearchy_lv = dict()
    
    for i, fname in enumerate(fnames_sort):
        
        fname = os.path.join(fpath, fname)
        cluster_statistics = pd.read_csv(fname)
        
        nspots = np.array(cluster_statistics['nspots'])
        nnoise = np.array(cluster_statistics['nnoise'])
        nClusters = np.array(cluster_statistics['nClusters'])
        nLeafClusters = np.array(cluster_statistics['nLeafClusters'])
        
        nspots_summ[kws_sort[i]] = nspots
        nnoise_summ[kws_sort[i]] = nnoise
        cluster_rate[kws_sort[i]] = 1.0 - nnoise / nspots
        nClusters_summ[kws_sort[i]] = nClusters
        nLeafClusters_summ[kws_sort[i]] = nLeafClusters
        hirearchy_lv[kws_sort[i]] = 1.0 - nLeafClusters / nClusters

    df = pd.DataFrame.from_dict(nspots_summ, orient='index')
    df = df.transpose()
    df.to_csv(os.path.join(fpath, 'nspots_summ.csv'), index=False)
    
    df = pd.DataFrame.from_dict(nnoise_summ, orient='index')
    df = df.transpose()
    df.to_csv(os.path.join(fpath, 'nnoise_summ.csv'), index=False)
    
    df = pd.DataFrame.from_dict(cluster_rate, orient='index')
    df = df.transpose()
    df.to_csv(os.path.join(fpath, 'cluster_rate.csv'), index=False)
    
    df = pd.DataFrame.from_dict(nClusters_summ, orient='index')
    df = df.transpose()
    df.to_csv(os.path.join(fpath, 'nClusters_summ.csv'), index=False)

    df = pd.DataFrame.from_dict(nLeafClusters_summ, orient='index')
    df = df.transpose()
    df.to_csv(os.path.join(fpath, 'nLeafClusters_summ.csv'), index=False)

    df = pd.DataFrame.from_dict(hirearchy_lv, orient='index')
    df = df.transpose()
    df.to_csv(os.path.join(fpath, 'hirearchy_lv.csv'), index=False)




if __name__ == '__main__':
    
    fpath = 'Z:/Data/chengsy/Dataset/GlueTAC'
    kws = ['10mins', '20mins', '30mins', '40mins', '50mins', '60mins', 'Non-Treated']
    cluster_properties_plot(fpath, kws)
    cluster_statistics_summ(fpath, kws)


    