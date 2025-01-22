import os
import inspect
import ctypes
from math import prod
import numpy as np
import numpy.ctypeslib as ctl


def pcorr2d(locA, photonA, locB, photonB, Imsz, rbins, leafLim=32):
    """
    Calculate the pross-correlation between two sets of 2d-localizations
    dependent on pcorr2d.dll in the same pacakge
    INPUT:
        locsA:              [[locx], [locy]], localizations of detections in channel A  
        photonA:            [nspotsA] the photon number of each detection in channel A 
        locsB:              [[locx], [locy]], localizations of detections in channel B
        photonB:            [nspotsB] the photon number of each detection in channel B
        Imsz:               (Imszy, Imszx) images size
        rbins:              if tuple (nbins, binsz), number of bins and radial size of each bin
                            if array [nbins+1], user defined radial bins for radial profile
        leafLim:            size of a leaf node for a kdtree                      
    RETURN:
        pc:                 [nbins] the AcorrB pair-correlation profile
    """
    
    Imszy, Imszx = Imsz
    nspotsA = len(locA[0])
    nspotsB = len(locB[0])
    
    posA = np.ascontiguousarray(np.array(locA).ravel(), dtype=np.float32)
    photonA = np.ascontiguousarray(photonA, dtype=np.float32)
    posB = np.ascontiguousarray(np.array(locB).ravel(), dtype=np.float32)
    photonB = np.ascontiguousarray(photonB, dtype=np.float32)

    # call c libraries
    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fname = 'pcorr2d.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    
    if type(rbins) is tuple:
        nbins, binsz = rbins
        pc = np.ascontiguousarray(np.zeros(nbins), dtype=np.float32)
        pcorr2d_kernel = lib.kernel_binsz
        pcorr2d_kernel.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_int32, ctypes.c_float,
                                    ctypes.c_int32,
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                                    ctypes.c_int32,
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                                    ctypes.c_int32,
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous")]
        pcorr2d_kernel(Imszy, Imszx, nbins, np.float32(binsz), nspotsA, posA, photonA, nspotsB, posB, photonB, leafLim, pc)
    
    else:
        nbins = len(rbins)-1
        rbins = np.ascontiguousarray(rbins, dtype=np.float32)
        pc = np.ascontiguousarray(np.zeros(nbins), dtype=np.float32)
        pcorr2d_kernel = lib.kernel_edge
        pcorr2d_kernel.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_int32, 
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                                    ctypes.c_int32,
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                                    ctypes.c_int32,
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                                    ctypes.c_int32,
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous")]
        pcorr2d_kernel(Imszy, Imszx, nbins, rbins, nspotsA, posA, photonA, nspotsB, posB, photonB, leafLim, pc)

    #pc *= prod(imsz) / nspotsA / nspotsB
    pc *= prod(Imsz) / photonA.sum() / photonB.sum()
    return pc



if __name__ == '__main__':
    
    from numpy.random import default_rng
    from matplotlib import pyplot as plt
    import time
    
    
    rbinEdges = np.hstack([np.arange(0, 10, 1), np.arange(10, 30, 2), np.arange(30, 80, 5), np.arange(80, 180, 10), 
                           np.arange(180, 380, 20), np.arange(380, 880, 50), np.arange(880, 1000, 100)]) # unit nm
    nbins = 500
    binsz = 1.0
    
    # simulation (unit = nm)
    nCluster = 100
    nMolPerCluster = 100
    nBlinking = 10

    Imszx, Imszy = 10000, 10000      
    rCluster = 100 
    rBlinking = 20

    rng = default_rng()
    xCluster = np.repeat(rng.random(nCluster)*Imszx, nMolPerCluster)
    yCluster = np.repeat(rng.random(nCluster)*Imszy, nMolPerCluster)
    xMol = np.repeat(rng.normal(loc=xCluster, scale=rCluster*np.ones(nCluster*nMolPerCluster)), nBlinking)
    yMol = np.repeat(rng.normal(loc=yCluster, scale=rCluster*np.ones(nCluster*nMolPerCluster)), nBlinking)
    locx = rng.normal(loc=xMol, scale=rBlinking*np.ones(nCluster*nMolPerCluster*nBlinking))
    locy = rng.normal(loc=yMol, scale=rBlinking*np.ones(nCluster*nMolPerCluster*nBlinking))
    loc = np.vstack([locx, locy])
    photons = rng.normal(loc=1000, scale=10, size=nCluster*nMolPerCluster*nBlinking)
    
    # correlation 2d
    tas = time.time()
    pc0 = pcorr2d(loc, photons, loc, photons, (Imszy, Imszx), rbinEdges, leafLim=32)
    print("{t:f} secs ellpased for autocorrelation of {n:d} spots".format(t=time.time()-tas, n=nCluster*nMolPerCluster*nBlinking))
    tas = time.time()
    pc1 = pcorr2d(loc, photons, loc, photons, (Imszy, Imszx), (nbins, binsz), leafLim=32)
    print("{t:f} secs ellpased for autocorrelation of {n:d} spots".format(t=time.time()-tas, n=nCluster*nMolPerCluster*nBlinking))

    fig, ax = plt.subplots()
    ax.plot(0.5*(rbinEdges[:-1]+rbinEdges[1:]), pc0)
    ax.plot(np.arange(0, nbins*binsz, binsz), pc1)
    xmin, xmax = ax.get_xlim()
    ax.hlines(y=1.0, xmin=xmin, xmax=xmax)
    plt.show()
    