import os
import inspect
import ctypes
from math import prod
import numpy as np
import numpy.ctypeslib as ctl


def ccorr(ndim, ccorrsz, locsA, photonsA, locsB, photonsB, leafLim=32):
    """
    Calculate the cross-correlation between two sets of localizations
    dependent on ccorr.dll in the same pacakge
    INPUT:  ndim:       int, number of dimention of the localizations (upto 3)
            ccorrsz:    (ndim,) int ndarray, (ccorrszx, ccorrszy, corrszz) windowsz for correlation computation
            locsA:      (nspotsA, ndim) float ndarray, [[locx, locy, locz],...] coordinates of detections in channel A
            photonsA:   (nspotsA,) float ndarray, the phot_avg number of detections in channel A
            locsB:      (nspotsB, ndim) float ndarray, [[locx, locy, locz],...] coordinates of detections in channel B
            photonsB:   (nspotsB,) float ndarray, the phot_avg number of detections in channel B
            leafLim:    int maximum number of points stored in a leaf node of a kd-tree
    RETURN: cc:         (prod(ccorrsz)) float ndarray, the cross-correlation profile
    """

    nspotsA = len(locsA)
    nspotsB = len(locsB)
    ccorrsz = np.ascontiguousarray(ccorrsz, dtype=np.int32)
    locsA = np.ascontiguousarray(np.array(locsA).flatten(), dtype=np.float32)
    photonsA = np.ascontiguousarray(photonsA, dtype=np.float32)
    locsB = np.ascontiguousarray(np.array(locsB).flatten(), dtype=np.float32)
    photonsB = np.ascontiguousarray(photonsB, dtype=np.float32)
    cc = np.ascontiguousarray(np.zeros(np.prod(ccorrsz)), dtype=np.float32)

    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fname = 'ccorr.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    ccorr_kernel = lib.kernel
    ccorr_kernel.argtypes = [ctypes.c_int32, 
                            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),
                            ctypes.c_int32,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_int32,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctypes.c_int32,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous")]
    ccorr_kernel(ndim, ccorrsz, nspotsA, locsA, photonsA, nspotsB, locsB, photonsB, leafLim, cc)
    cc *= prod(ccorrsz) / np.sum(photonsA) / np.sum(photonsB)
    
    return cc.reshape(tuple(ccorrsz[ndim-1::-1]))




if __name__ == '__main__':
    
    from numpy.random import default_rng
    from matplotlib import pyplot as plt
    import time
    
    # camera parameters (unit = nm)
    TarPxszx, TarPxszy, Tarzstepsz = 5, 5, 2
     
    # simulation parameters (unit = nm)
    nspots = 50000
    phot_avg = 1000
    phot_std = 10

    fov_lateral = 20000
    fov_axial = 2000
    ccorrsz_lateral = 5000
    ccorrsz_axial = 200
    
    loc_errx = 1.0
    loc_erry = 1.0
    loc_errz = 3.0
    
    shiftx = 235
    shifty = -177
    shiftz = 48

    # tranlsate to Target units
    fov_width, fov_height, fov_depth = fov_lateral/TarPxszx, fov_lateral/TarPxszy, fov_axial/Tarzstepsz
    ccorrszx, ccorrszy, ccorrszz = np.int32((np.ceil(ccorrsz_lateral/TarPxszx), np.ceil(ccorrsz_lateral/TarPxszy), np.ceil(ccorrsz_axial/Tarzstepsz)))
    simu_shiftx, simu_shifty, simu_shiftz = shiftx/TarPxszx, shifty/TarPxszy, shiftz/Tarzstepsz
    
    # simulation
    rng = default_rng()
    center_x = rng.uniform(low=0.0, high=fov_width, size=nspots)        
    center_y = rng.uniform(low=0.0, high=fov_height, size=nspots)       
    center_z = rng.uniform(low=0.0, high=fov_depth, size=nspots)

    locx_A = rng.normal(loc=center_x, scale=loc_errx)
    locy_A = rng.normal(loc=center_y, scale=loc_erry)
    locz_A = rng.normal(loc=center_z, scale=loc_errz)
    msk_ind = (locx_A >= 0) & (locx_A < fov_width) & (locy_A >= 0) & (locy_A < fov_height) & (locz_A >= 0) & (locz_A < fov_depth) 
    locA = np.vstack((locx_A[msk_ind], locy_A[msk_ind], locz_A[msk_ind])).T
    nspotsA = len(locA)
    photonsA = rng.normal(loc=phot_avg, scale=phot_std, size=nspotsA)

    locx_B = rng.normal(loc=center_x+simu_shiftx, scale=loc_errx)
    locy_B = rng.normal(loc=center_y+simu_shifty, scale=loc_erry)
    locz_B = rng.normal(loc=center_z+simu_shiftz, scale=loc_errz)
    msk_ind = (locx_B >= 0) & (locx_B < fov_width) & (locy_B >= 0) & (locy_B < fov_height) & (locz_B >= 0) & (locz_B < fov_depth)
    locB = np.vstack((locx_B[msk_ind], locy_B[msk_ind], locz_B[msk_ind])).T
    nspotsB = len(locB)
    photonsB = rng.normal(loc=phot_avg, scale=phot_std, size=nspotsB)

    fig, axs = plt.subplots(1, 2)
    # correlation 2d
    tas = time.time()
    cc2d = ccorr(2, (ccorrszx, ccorrszy), locA[:,:2], photonsA, locB[:,:2], photonsB)
    indy, indx = np.array(np.unravel_index(np.argmax(cc2d), (ccorrszy, ccorrszx)))
    print("correction_x = {dx:f}, correction_y = {dy:f}".format(dx=(indx-ccorrszx/2.0)*TarPxszx, dy=(indy-ccorrszy/2.0)*TarPxszy))
    print(time.time()-tas)
    imobj0 = axs[0].imshow(cc2d)
    fig.colorbar(imobj0, ax=axs[0])
    xmin, xmax = axs[0].get_xlim()
    ymin, ymax = axs[0].get_ylim()
    axs[0].plot(indx, indy, 'rx')
    axs[0].plot(simu_shiftx+cc2d.shape[1]/2.0, simu_shifty+cc2d.shape[0]/2.0, 'wo', mfc='None')
    axs[0].hlines(y=cc2d.shape[0]/2.0, xmin=xmin, xmax=xmax)
    axs[0].vlines(x=cc2d.shape[1]/2.0, ymin=ymin, ymax=ymax)

    # correlation 3d
    tas = time.time()
    cc3d = ccorr(3, (ccorrszx, ccorrszy, ccorrszz), locA, photonsA, locB, photonsB)
    indz, indy, indx = np.array(np.unravel_index(np.argmax(cc3d), cc3d.shape))
    print("correction_x = {dx:f}, correction_y = {dy:f}, correction_z = {dz:f}".format(dx=(indx-cc3d.shape[2]/2.0)*TarPxszx, dy=(indy-cc3d.shape[1]/2.0)*TarPxszy, dz=(indz-cc3d.shape[0]/2.0)*Tarzstepsz))
    print(time.time()-tas)
    imobj1 = axs[1].imshow(cc3d[indz])
    fig.colorbar(imobj1, ax=axs[1])
    xmin, xmax = axs[0].get_xlim()
    ymin, ymax = axs[0].get_ylim()
    axs[1].plot(indx, indy, 'rx')
    axs[1].plot(simu_shiftx+cc2d.shape[1]/2.0, simu_shifty+cc2d.shape[0]/2.0, 'wo', mfc='None')
    axs[1].hlines(y=cc2d.shape[0]/2.0, xmin=xmin, xmax=xmax)
    axs[1].vlines(x=cc2d.shape[1]/2.0, ymin=ymin, ymax=ymax)

    plt.show()
    