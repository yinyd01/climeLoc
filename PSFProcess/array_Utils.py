import numpy as np
import numpy.fft as ft
from scipy.ndimage import map_coordinates



def _nextpow2(x):
    # NOTE: numpy build-in log2 and power function is faster than verctorizing the bit_length function for non-numpy type
    if isinstance(x, int):
        return 1 if x < 1 else 2**((x - 1).bit_length())
    elif isinstance(x, np.int_):
        return 1 if x < 1 else 2**np.ceil(np.log2(x))
    else:
        x = np.array(x)
        x[x < 1] = 1
        return 2**np.ceil(np.log2(x)).astype(np.int32)



def upsample(arr, upscalars, norder):
    """
    UPsampling an array by given upscalars via spline interpolation
    INPUT:
        arr:            (szz, szy, szx) ndarray, the input array for upsampling
        upscalars:      [upscalarz, upscalary, upscalarx] ndarray, the arr will be upsampled by upfactor times
        norder:         int, the spline order for upsampling, must be integer between 0 and 5
    RETURN:
        arr_up:         (szz*upscalarz, szy*upscalary, szx*upscalarx) ndarray, the upsampled arr
    NOTE:
        scipy.ndimage.map_coordinates sees arr locats at [0, 1, ... sz-1] other than [0.5, 1.5, ... sz-0.5]
        thus the qvecs should also moves 0.5 back before using map_coordinates
    """
    szs = arr.shape
    upszs = szs * upscalars
    if np.all(upscalars <= 1):
        arr_up = np.copy(arr)
    else:
        vecs = [np.linspace(0, sz, upsz, endpoint=False) + 0.5/upscalar - 0.5 for sz, upsz, upscalar in zip(szs, upszs, upscalars)]
        grids = np.meshgrid(*vecs, indexing='ij')
        arr_up = map_coordinates(arr, grids, order=norder, mode='nearest')
    return arr_up



def interp(arr, raw_step, tar_step, norder=3):
    """
    Interpolate an array from grids in raw_step to grids in tar_step
    INPUT:
        arr:            (szz, szy, szx) ndarray, the input array for interplotation
        raw_step:       [raw_stepz, raw_stepy, raw_stepx] ndarray, the step size of the original grid set
        tar_step:       [tar_stepz, tar_stepy, tar_stepx] ndarray, the step size of the target grid set
        norder:         int, the spline order for upsampling, must be integer between 0 and 5, default = 3
    OUTPUT:
        arr_interpped:  (_szz, _szy, _szx) ndarray, the interpolated array. shapes are determined by the original shape and the interploted step size
    """
    if np.any(arr.shape % 2 == 0):
        raise ValueError('not all odd: shape={}'.format(arr.shape))
    
    raw_hsz = np.array(arr.shape // 2)
    interp_hsz = raw_hsz * np.int32(raw_step // tar_step)
    
    vecs = [np.arange(-ihsz, ihsz+1) * dt / dr + rhsz for ihsz, dt, dr, rhsz in zip(interp_hsz, tar_step, raw_step, raw_hsz)]
    grids = np.meshgrid(*vecs, indexing='ij')
    arr_interped = map_coordinates(arr, grids, order=norder, mode='nearest')
    return arr_interped



def mass_center(arr, upscalars, norder=3):
    """
    Get the mass center, central Intensity, and bkg of an array
    INPUT:
        arr:            (szz, szy, szx) ndarray, the input array for mass centers
        upscalars:      [upscalarz, upscalary, upscalarx] ndarray, the arr will be upsampled by upfactor times
        norder:         int, the spline order for upsampling, must be integer between 0 and 5, default = 3
    RETURN:
        cntr_locs:      [zc, yc, xc] ndarray, the mass centers of the arr
        amp:            float, mean intensity of the central region of the arr
        bkg:            float, median of the most minimum 5*upscalars.prod()-th pixels (the 3*upscalars.prod()-th smallest pixel) of the upsampled arr
    """
    arr = upsample(arr, upscalars, norder)
    bkg = np.partition(arr.flatten(), 3*np.prod(upscalars)-1)[2]

    szs = arr.shape
    vecs = [np.arange(sz) + 0.5 for sz in szs]
    grids = np.meshgrid(*vecs, indexing='ij')
    wi = arr.sum()
    cntr_locs = np.array([np.sum(grid * arr) / wi for grid in grids])
    
    cntr_inds = np.int32(cntr_locs)
    slc = [slice(max(cntr_ind-upscalar, 0), min(cntr_ind+upscalar+1, sz), 1) for cntr_ind, upscalar, sz in zip(cntr_inds, upscalars, szs)]
    cntr_region = arr[tuple(slc)]
    amp = cntr_region.mean() - bkg

    return cntr_locs / upscalars, amp, bkg



def mass_var(arr, upscalars, norder=3):
    """
    Get the mass variance (standard diviation)
    INPUT:
        arr:            (szz, szy, szx) ndarray, the input array for mass centers
        upscalars:      [upscalarz, upscalary, upscalarx] ndarray, the arr will be upsampled by upfactor times
        norder:         int, the spline order for upsampling, must be integer between 0 and 5, default = 3
    RETURN:
        cntr_var:       [varz, vary, varx] ndarray, the mass variance of the arr
    """
    arr = upsample(arr, upscalars, norder)
    
    szs = arr.shape
    vecs = [np.arange(sz) + 0.5 for sz in szs]
    grids = np.meshgrid(*vecs, indexing='ij')
    wi = arr.sum()
    mass_mu = np.array([np.sum(grid * arr) / wi for grid in grids])
    mass_mu2 = np.array([np.sum(grid * grid * arr) / wi for grid in grids])
    mass_var = mass_mu2 - mass_mu * mass_mu
    
    return mass_var / upscalars / upscalars



def max_center(arr, upscalars, norder=3):
    """
    Get the max center, Intensity, and bkgd of an array
    INPUT:
        arr:            (szz, szy, szx) ndarray, the input array for max centers
        upscalars:      [upscalarz, upscalary, upscalarx] ndarray, the arr will be upsampled by upfactor times
        norder:         int, the spline order for upsampling, must be integer between 0 and 5, default = 3   
    RETURN:
        ind_max:        [zmax, ymax, xmax] ndarray, the max index of the arr (float, because of upscaling)
        I_max:          float, approximate to arr.max()
        bkg:            float, median of the most minimum 5*upscalars.prod()-th pixels (the 3*upscalars.prod()-th smallest pixel) of the upsampled arr
    """
    arr = upsample(arr, upscalars, norder)
    ind_maxs = np.unravel_index(np.argmax(arr), arr.shape)
    bkg = np.partition(arr.flatten(), 3*np.prod(upscalars)-1)[2]
    I_max = arr[ind_maxs] - bkg    
    return ind_maxs / upscalars, I_max, bkg



def translate(arr, svec, method='spline', norder=3):
    """
    Linearly translate an array along the svec
    INPUT:
        arr:            (szz, szy, szx) ndarray, the input array for translate
        svec:           [deltaz, deltay, deltax] ndarray, the translation steps along the 3 axis
        method:         'spline':   translate the arr via spline interpolation     
                        'FT':       translate the arr via Fourier Transfer
        norder:         int, the spline order for upsampling, must be integer between 0 and 5, default = 3 (only used if method='spline')
    RETURN:
        arr_trans:      (szz, szy, szx) ndarray, the translated arr 
    """
    szs = arr.shape
    if method == 'FT':
        spans = _nextpow2(szs)
        ext = [(0, span-sz) for span, sz in zip(spans, szs)]
        padded_arr3d = np.pad(arr, ext, 'edge')

        kvecs = [ft.fftfreq(span, d=1.0) for span in spans]
        kgrids = np.meshgrid(*kvecs, indexing='ij')
        for i in range(arr.ndim):
            kgrids[i] *= svec[i]
        ktrans = np.exp(2j * np.pi * np.sum(kgrids, axis=0))
        arr_trans = np.real( ft.ifftn( ft.fftn(padded_arr3d) * ktrans ) )
        
        slc = [slice(0, sz, 1) for sz in szs]
        arr_trans = arr_trans[tuple(slc)]
    
    else:
        vecs = [np.arange(sz) - s for sz, s in zip(szs, svec)]
        grids = np.meshgrid(*vecs, indexing='ij')
        arr_trans = map_coordinates(arr, grids, order=norder, mode='nearest')
    
    return arr_trans



def ccorr(arr_A, arr_B, upscalars, norder=3):
    """
    Calculate the correlation between arr_A and arr_B 
    INPUT:
        arr_A:          (szz, szy, szx) ndarray, the input array for correlation
        arr_B:          (szz, szy, szx) ndarray, the input array for correlation
        upscalars:      [upscalarz, upscalary, upscalarx] ndarray, the arr will be upsampled by upfactor times
    RETURN:
        Corr0:          float, the Correlation value at 0 (at span//2 after fftshift)
        CorrMax:        float, the maximum of the correlation profile
        shifts:         [shiftz, shifty, shiftx] ndarray, the shifts from arr_B to arr_A
        shiftDist:      float, np.sum(shifts**2) the distance of the shifts
    NOTE 1:
        span = nextpow2(sz), must be even
    NOTE 2:
        for odd  freq sampling, freq = ft.fftfreq(5) = [0, 0.2, 0.4. -0.4, -0.2] and ft.fftshift(freq) = [-0.4, -0.2, 0, 0.2, 0.4], '0' is at the center 5/2
        for even freq sampling, freq = ft.fftfreq(4) = [0, 0.25, -0.5, -0.25] and ft.fftshift(freq) = [-0.5, -0.25, 0, 0.25], '0' is at the center (4+1)/2
    """
    
    # Parse the input
    if np.any(arr_A.shape != arr_B.shape):
        raise ValueError("shape mismatch: arr_A.shape={}, arr_B.shape={}".format(arr_A.shape, arr_B.shape))
    
    # Upsample
    arr_r = upsample(arr_A, upscalars, norder)
    arr_r -= arr_r.min()
    arr_r /= arr_r.mean()

    arr_w = upsample(arr_B, upscalars, norder)
    arr_w -= arr_w.min()
    arr_w /= arr_w.mean()
    
    # Correlation via fft
    szs = arr_r.shape
    spans = _nextpow2(szs)
    ext = [(0, span-sz) for span, sz in zip(spans, szs)]
    padded_arr3dr = np.pad(arr_r, ext, 'edge')
    padded_arr3dw = np.pad(arr_w, ext, 'edge')
    ft_arr3dr = ft.fftn(padded_arr3dr) 
    ft_arr3dw = ft.fftn(padded_arr3dw)
    Corr = ft.fftshift(np.abs( ft.ifftn(ft_arr3dr * np.conj(ft_arr3dw)) )) / np.size(padded_arr3dw)
    Corr0 = Corr[*spans//2]
    
    # Locate the maximum of the correlation
    MaxCorrInds = np.unravel_index(np.argmax(Corr), Corr.shape)
    slc = [slice(max(MaxCorrInd-1, 0), min(MaxCorrInd+2, span), 1) for MaxCorrInd, span in zip(MaxCorrInds, spans)]
    cntrCorr = Corr[*slc]
    cntr_locs, CorrMax = mass_center(cntrCorr, np.zeros(len(szs), dtype=np.int32)+20)[:2]
    
    shifts = [max(MaxCorrInd-1,0)+cntr_loc-(span//2+0.5) for MaxCorrInd, cntr_loc, span in zip(MaxCorrInds, cntr_locs, spans)]
    shifts = np.array(shifts) / upscalars
    shiftDist = np.sqrt(np.sum(shifts * shifts))
    
    return Corr0, CorrMax, shifts, shiftDist