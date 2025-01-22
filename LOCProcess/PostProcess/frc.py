import os
import inspect
import ctypes
import numpy as np
import numpy.ctypeslib as ctl 
from numpy.random import default_rng
import numpy.fft as ft
from scipy.signal import savgol_filter



def _tukeywin(im_in, nfrac=8):
    """
    apply tukey window to the input image
    INPUT:
        im_in:      (imThickness, imHeight, imWidth) or (imHeight, imWidth) float ndarray, the input image for tukey window filter
        nfrac:      int, images are divided into nfrac segs, and the edge seg on both sides are tukey windowed, for both axes
    RETURN:
        im_out:     (imThickness, imHeight, imWidth) or (imHeight, imWidth) float ndarray, the output image
    """
    im_in = np.array(im_in)
    assert im_in.ndim in (2, 3), "Only support for 2d or 3d im_in, im_in.ndim={}".format(im_in.ndim)
    
    masks = []
    for sz in im_in.shape:
        dum = (np.arange(sz) + 0.5 - sz / 2.0) / sz
        mask = 0.5 - 0.5 * np.cos(nfrac * np.pi * dum)
        mask[np.abs(dum) < (nfrac-2)/2/nfrac] = 1
        masks.append(mask)
    
    if len(masks) == 2:
        mask_dd = np.outer(masks[0], masks[1])
    elif len(masks) == 3:
        mask_dd = np.multiply.outer(masks[0], np.outer(masks[1], masks[2]))
    return mask_dd * im_in
    


def _radialsum(mat):
    """
    radial sum for a matrix  
    """
    mat = np.array(mat)
    ndim = mat.ndim
    maxR = (min(mat.shape)-1)//2+1
    if ndim == 3:
        imThickness, imHeight, imWidth = mat.shape
    elif ndim == 2:
        imHeight, imWidth = mat.shape

    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_radialsum')
    fname = 'radialsum.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))

    mat = np.ascontiguousarray(mat.ravel(), dtype=np.float64)
    radial = np.ascontiguousarray(np.zeros(maxR), dtype=np.float64)

    if ndim == 3:
        radialsum_kernel = lib.kernel_3d
        radialsum_kernel.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,                                              
                                ctl.ndpointer(np.float64, flags="aligned, c_contiguous"),  
                                ctl.ndpointer(np.float64, flags="aligned, c_contiguous")]
        radialsum_kernel(imThickness, imHeight, imWidth, mat, radial)
    elif ndim == 2:
        radialsum_kernel = lib.kernel_2d
        radialsum_kernel.argtypes = [ctypes.c_int32, ctypes.c_int32,                                              
                                    ctl.ndpointer(np.float64, flags="aligned, c_contiguous"),  
                                    ctl.ndpointer(np.float64, flags="aligned, c_contiguous")]
        radialsum_kernel(imHeight, imWidth, mat, radial)
    
    return radial



def frc_fft(locs_nm, tarpxsz=5.0, winfold=True):
    """
    get the Fourier Ring Correlation
    INPUT:
        locs_nm:        (nspots, ndim) float ndarray, the localizations in nanometer
        tarpxsz:        int, size of the rendering pixel
    RETURN: 
        freq:           float ndarray, the frequence space in 1/nm
        frc:            float ndarray, the frac radial profile corresponding to freq
    """
    
    # tranlsate from nanometer to tarpxsz
    locs_nm = np.array(locs_nm)
    nspots, ndim = locs_nm.shape
    locs = locs_nm / tarpxsz
    for dimID in range(ndim):
        locs[:, dimID] -= locs[:, dimID].min()

    # determine the windowsz and fold the localizations into the window
    Imsz = []
    for dimID in range(ndim):
        Imsz.append(np.int32(locs[:, dimID].max()))
    windowsz = np.min(Imsz)
    for dimID in range(ndim):
        if winfold:
            locs[:, dimID] %= windowsz
    maxR = (windowsz - 1) // 2 + 1
    freq = ft.fftfreq(windowsz, d=tarpxsz)[:maxR]

    # shuffle the localizations
    rng = default_rng()
    ind_rnd = np.arange(nspots)
    rng.shuffle(ind_rnd)
    indA = ind_rnd[:nspots//2]
    indB = ind_rnd[nspots//2:]

    # render and filter the image
    imA = np.histogramdd(locs[indA], bins=(np.tile(np.arange(windowsz+1), ndim).reshape(ndim, windowsz+1)))[0]
    imB = np.histogramdd(locs[indB], bins=(np.tile(np.arange(windowsz+1), ndim).reshape(ndim, windowsz+1)))[0]
    imA = _tukeywin(imA)
    imB = _tukeywin(imB)
    
    # get the frc profile
    fimA = ft.fftshift(ft.fftn(imA))
    fimB = ft.fftshift(ft.fftn(imB))    
    realfimAB = np.real(fimA * np.conj(fimB)) / windowsz / windowsz
    absfimAA = np.abs(fimA) * np.abs(fimA) / windowsz / windowsz
    absfimBB = np.abs(fimB) * np.abs(fimB) / windowsz / windowsz
    dum_AB = _radialsum(realfimAB)
    dum_AA = _radialsum(absfimAA)
    dum_BB = _radialsum(absfimBB)
    
    dum_ind = (dum_AA > 0) & (dum_BB > 0)
    frc = np.zeros(maxR)
    frc[dum_ind] = dum_AB[dum_ind] / np.sqrt(dum_AA[dum_ind] * dum_BB[dum_ind])

    # smooth the frc profile and calculate the resolution
    sspan = max(maxR // 20, 5)
    sspan += 1 - sspan % 2
    frc_smooth = savgol_filter(frc, sspan, 3)
    ind = np.argmin(np.abs(frc_smooth-1.0/7.0))
    frc_val = 1 / freq[ind]
    
    return np.vstack((freq, frc, frc_smooth)), frc_val











if __name__ == '__main__':
    
    import os
    import pickle
    from matplotlib import pyplot as plt
    from numpy.random import default_rng

    # Camera parameters (unit = nm)
    fname_smlm = 'D:/Data/samples/HYF_G1A50_20240111_SAMPLES_MICRO_647M_TOM20_660CR_samples-3d/spool_1000MW_3D_50ms_5_2/smlm_result_mulpa/spool_1000MW_3D_50ms_5_2_roi_0_locsnm.pkl'
    with open(fname_smlm, 'rb') as fid:
        smlm_data = pickle.load(fid)
    idx = (smlm_data['xvec'][:, 2] > 300) & (smlm_data['xvec'][:, 2] < 900)
    locs_nm = smlm_data['xvec'][:, :3]
    locx_nm = smlm_data['xvec'][:, 0]
    locy_nm = smlm_data['xvec'][:, 1]
    frc_2d, frc_2d_val = frc_fft(locs_nm[idx, :2])
    frc_3d, frc_3d_val = frc_fft(locs_nm[idx], tarpxsz=2.5)
    print("frc2d = {}, frc3d = {}".format(frc_2d_val, frc_3d_val))

    fig, ax = plt.subplots()
    ax.plot(frc_2d[0], frc_2d[1], color='tab:blue', alpha=0.25)
    ax.plot(frc_2d[0], frc_2d[2], color='tab:blue')

    ax.plot(frc_3d[0], frc_3d[1], color='tab:orange', alpha=0.25)
    ax.plot(frc_3d[0], frc_3d[2], color='tab:orange')

    xmin, xmax = ax.get_xlim()
    ax.hlines(y = 1.0/7.0, xmin=xmin, xmax=xmax, color='k')
    ax.set_xscale('log')
    plt.show()