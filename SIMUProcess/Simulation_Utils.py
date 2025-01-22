import os
import inspect
import ctypes
import numpy as np
import numpy.ctypeslib as ctl

"""
PhotonFlux = phi * P * sigma / e
phi:    the quantum yield of the dye
P:      the laser power in W / cm**2 
sigma:  1000*ln(10)*epsilon/AVOGADRO, the absorption cross section in cm**2 per molecule, where epsilon is the Molar excitation coefficient in cm**2/mol
e:      PLANCK * LIGHTSPEED / wavelength, energy of 1 photon      
"""
PLANCK      = 6.626e-34     # unit = m**2 kg / s
AVOGADRO    = 6.022e23      # unit = 1
LIGHTSPEED  = 3.0e8         # unit = m / s
Fluorophore = {'Dentra2':   {'phi':0.55, 'wavelength':573e-9, 'epsilon':4000},
               'AF647':     {'phi':0.33, 'wavelength':660e-9, 'epsilon':239000} }



def get_PhotonFlux(illumination, exposure, dye='AF647'):
    """
    Calculate the photon flux for each emitter per frame
    INPUT:
        illumination:   float, power of the illumination laser in W/cm**2
        exposure:       float, exposure time per frame, unit = sec per frame
        dye:            str, 'AF647' or 'Dentra2', dye prameters are set according to the given dye
    RETURN:
        PhotonFlux:     the emitted photon flux for an emitter per frame
    """
    fluor = Fluorophore[dye]
    e = PLANCK * LIGHTSPEED / fluor['wavelength']                   # Energy of a photon in J
    sigma = 1000.0 * np.log(10) * fluor['epsilon'] / AVOGADRO       # Absorption cross section in cm**2 per molecule
    return fluor['phi'] * sigma * illumination * exposure / e
    


def emitter_sampler_FrameIndependent(density, roisz, zrange, photon_range):
    """
    sample the emitters that are uniformly distributed within the roi and their photons
    INPUTS:
        density:        float, the desired density of the emitters, unit = pixel**-2
        roisz:          (2,) float ndarray, [roiszx, roiszy] the size of the roi
        zrange:         (2,) float ndarray, [zmin, zmax] the zrange for 3D sampling, ignored if ndim==2
        photon_range:   (2,) float ndarray, [min, max] of the photon flux per camera frame
    RETURN:
        locs:           (nemitters, 3) float ndarray, [[locx, locy, locz],...] the localization of the emitters
        phot:           (nemitters,) float ndarray, the photon number of each emitter
    """ 
    rng = np.random.default_rng()
    nem_avg = np.prod(roisz[:2]) * density
    nemitters = rng.poisson(lam=nem_avg)
    
    locs = rng.random(size=(nemitters, 3))
    locs[:, :2] *= roisz
    locs[:, 2] = zrange[0] + locs[:, 2] * (zrange[1] - zrange[0])
    phot = rng.uniform(low=photon_range[0], high=photon_range[1], size=nemitters)

    return locs, phot



def emitter_sampler_Blinking(density, roisz, zrange, nfrm, photon_mu_sig, lifetime):
    """
    sample the emitters that are uniformly distributed within the roi and their photons
    only one blinking cycle
    INPUTS:
        density:        float, the desired density of the emitters, unit = pixel**-2
        roisz:          (2,) float ndarray, [roiszx, roiszy] the size of the roi
        zrange:         (2,) float ndarray, [zmin, zmax] the zrange for 3D sampling, None if ndim==2
        nfrm:           int, number of frames
        photon_mu_sig:  (2,) float ndarray, the mu and sigma of the photon flux per frame
        lifetime:       float, the average lifetime of an emitter, unit = frame
    RETURN:
        locs:           (nemitters, 3) float ndarray, [[locx, locy, locz],...] the localization of the emitters
        t0:             (nemitters,) float ndarray, time point an emitter is turned on, in between (-3.0*lifetime, nfrm+3.0*lifetime)
        te:             (nemitters,) float ndarray, time point an emitter is turned off, te = t0 + ton where ton is the exponentially(scale=lifetime) sampled on time of an emitter
        phot:           (nemitters,) float ndarray, the photon flux (per second) of each emitter
    """ 
    rng = np.random.default_rng()
    trange = (-3.0*lifetime, nfrm+3.0*lifetime)
    nem_avg = np.prod(roisz[:2]) * density * (trange[1] - trange[0]) / (lifetime + 1.0)
    nemitters = rng.poisson(lam=nem_avg)
    
    locs = rng.random(size=(nemitters, 3))
    locs[:, :2] *= roisz
    locs[:, 2] = zrange[0] + locs[:, 2] * (zrange[1] - zrange[0])
    
    t0 = rng.uniform(low=trange[0], high=trange[1], size=nemitters)
    ton = rng.exponential(scale=lifetime, size=nemitters)
    te = t0 + ton
    phot = rng.normal(loc=photon_mu_sig[0], scale=photon_mu_sig[1], size=nemitters)

    return locs, t0, te, phot



def emitter_sampler_pop(nnum, boxsz, zrange, nfrm, photon_mu_sig, fracN):
    """
    sample the emitters in consecutive nfrms within fitting boxsz, for proof of principle only
    INPUTS:
        nnum:           int, the desired number of the emitters
        boxsz:          int, the size of the fitting box
        zrange:         (2,) float ndarray, [zmin, zmax] the zrange for 3D sampling, None if ndim==2
        nfrm:           int, number of frames
        photon_mu_sig:  (nfluorophores, 2) float ndarray, the mu and sigma of the photon flux per frame
        fracN:          (nfluorophores) float ndarray, the fracN of the emitters
    RETURN:
        locs:           (nfrm * nnum, ndim) float ndarray, [[locx, locy, locz],...] the localization of the emitters 
        phot:           (nfrm * nnum) float ndarray, the photon flux of each localization at each frame of each emitter
    """
    assert boxsz >= 5, "boxsz should >= 5"
    
    rng = np.random.default_rng()
    nfluorophores = len(photon_mu_sig)
    assert len(fracN) == nfluorophores, "length mismatch: len(photon_mu_sig) should be the same as len(fracN)"

    # locs
    locs = np.zeros((nfrm, nnum, 3))
    locs[:, 0, :2] = 0.5 * boxsz
    locs[:, 0, 2] = rng.uniform(low=zrange[0]+0.25*(zrange[1]-zrange[0]), high=zrange[1]-0.25*(zrange[1]-zrange[0]), size=nfrm)
    
    if nnum > 1:
        radius = 3.5
        dtheta = 2.0 * np.pi / (nnum - 1)

        rho = np.zeros((nfrm, nnum-1)) + radius
        theta = np.repeat(0.5 * np.pi * rng.random(size=nfrm), nnum-1).reshape((nfrm, nnum-1)) + np.tile(np.arange(nnum-1) * dtheta, nfrm).reshape((nfrm, nnum-1))
        locs[:, 1:, 0] = 0.5 * boxsz + rho * np.cos(theta)
        locs[:, 1:, 1] = 0.5 * boxsz + rho * np.sin(theta)
        locs[:, 1:, 2] = rng.uniform(low=zrange[0], high=zrange[1], size=(nfrm, nnum-1))
    
    locs = np.reshape(locs, (nfrm * nnum, 3))
    
    # photons for each emitter at each frame
    idx = rng.choice(nfluorophores, size=(nfrm*nnum))
    phot = rng.normal(loc=photon_mu_sig[idx, 0], scale=photon_mu_sig[idx, 1])
    frac = fracN[idx]
    return locs, phot, frac



def get_frame_emitter(nfrm, locs_i, t0, te, photon_flux):
    """
    rearange the input locs from its emitterwise form to framewise form with its on- and off- time point
    INPUTS:
        nfrm:           int, number of frames to rearange
        locs_i:         (nemitters, ndim) float ndarray, the [[locx, locy, locz],...] localizations of the input locs
        t0:             (nemitters,) float ndarray, time point an emitter is turned on, in between (-3.0*lifetime, nfrm+3.0*lifetime)
        te:             (nemitters,) float ndarray, time point an emitter is turned off, te = t0 + ton where ton is the exponentially(scale=lifetime) sampled on time of an emitter
        photon_flux:    (nemitters,) float ndarray, the photon flux (per second) of each emitter
    RETURN:
        indf:           (nspots,) int ndarray, the frame index of each framewise localizations
        locs:           (nspots, ndim) float ndarray, the framewise localizations rearanged from its emitterwise form
        phot:           (nspots,) float ndarray, the photon of each framewise localization
    """
    nemitters, ndim = locs_i.shape
    indf = np.array([], dtype=np.int32)
    locs = np.zeros((0, ndim))
    phot = np.array([])
    for f in range(nfrm):
        _ind = (t0 < f + 1) & (te > f)
        indf = np.hstack((indf, np.zeros(_ind.sum(), dtype=np.int32) + f))
        locs = np.vstack((locs, locs_i[_ind]))

        durations = np.minimum(f+1, te[_ind]) - np.maximum(f, t0[_ind])
        phot = np.hstack((phot, photon_flux[_ind] * durations))

    return indf, locs, phot

        

def get_frame_psf(PSF, roisz, emitter_locs, emitter_phot, bkg, chID=0):
    """
    get frame photons from the emitters and the background 
    INPUT:
        PSF:            dict, see Documentation.md
        roisz:          (2,) int ndarray, [roiszx, roiszy] the size of the roi for simulation
        emitter_locs:   (nemitters, ndim) float ndarray, [[locx, locy, locz],...] the locations of the emitters(emitters)
        emitter_phot:   (nemitters,) float ndarray, photon of each emitter
        bkg:            float, uniform background photons
        chID:           int, the channel ID for n-channel PSF
    RETUEN:
        frame_photon:   (roiszx, roiszy) float ndarray, the simulated photon frame       
    """  
    ndim = 3 if 'splineszz' in PSF.keys() else 2
    imWidth, imHeight = roisz    
    splineszx = PSF['splineszx']
    nspots = len(emitter_phot)
    
    coeff_PSF = np.ascontiguousarray(PSF['coeff'][chID], dtype=np.float32)
    locs = np.ascontiguousarray(emitter_locs.flatten(), dtype=np.float32)
    phot = np.ascontiguousarray(emitter_phot, dtype=np.float32)
    frame_photon = np.ascontiguousarray(np.zeros(imHeight * imWidth), dtype=np.float32)

    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_PhotonRender')
    fname = 'PhotonRender.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))

    if ndim == 2:
        render_kernel = lib.PhotRender_2d
        render_kernel.argtypes = [ctypes.c_int32, 
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),                                              
                                    ctypes.c_int32,
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                                    ctypes.c_int32, ctypes.c_int32,
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous")]
        render_kernel(splineszx, coeff_PSF, nspots, locs, phot, imHeight, imWidth, frame_photon)
        frame_photon = np.reshape(frame_photon, (imHeight, imWidth))
    
    if ndim == 3:
        splineszz = PSF['splineszz']
        render_kernel = lib.PhotRender_3d
        render_kernel.argtypes = [ctypes.c_int32, ctypes.c_int32, 
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),                                              
                                    ctypes.c_int32,
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                                    ctypes.c_int32, ctypes.c_int32,
                                    ctl.ndpointer(np.float32, flags="aligned, c_contiguous")]
        render_kernel(splineszx, splineszz, coeff_PSF, nspots, locs, phot, imHeight, imWidth, frame_photon)
        frame_photon = np.reshape(frame_photon, (imHeight, imWidth))
    
    return frame_photon + bkg



def get_frame_cam(Camsetting, frame_photon):
    """
    get camera frames from frame_photons with the camera noise 
    INPUT:
        Camsetting:     dict, simulated camera properties {'offset', 'var', 'A2D', 'EMGain', 'QE', 'c'}
        frame_photon:   (roiszx, roiszy) float ndarray, the simulated photon frame 
    RETUEN:
        frame_cam:      (canvaszy, canvaszx) int ndarray, the simulated camera image       
    """
    rng = np.random.default_rng()
    frame_photon[frame_photon < 0] = 0
    frame_ie = np.float64(rng.poisson(Camsetting['QE']*frame_photon + Camsetting['c']))
    frame_oe = rng.gamma(frame_ie, scale=Camsetting['EMGain']) if Camsetting['EMGain'] > 1.0 else frame_ie.copy()
    if Camsetting['var'] > 0.0:
        frame_oe = rng.normal(loc=frame_oe, scale=np.sqrt(Camsetting['var']))
        
    frame_cam = np.uint16(np.minimum(np.maximum(Camsetting['A2D'] * frame_oe + Camsetting['offset'] + 0.5, 0), 65535))

    return frame_cam