import os
import inspect
import ctypes
import numpy.ctypeslib as ctl
import numpy as np
from matplotlib import pyplot as plt
from tifffile import TiffWriter



def im_draw(imin, boxsz, intcorner, intcenter):
    """
    draw RED square rois onto an gray-scale uint16 image (flattened 2d array)
    INPUT:
        imin:           (nchannels, imszh, imszw) int ndarray, uint16 n-channel gray-scale images to draw square rois
        boxsz:          int, size of the roi, must be odd
        indcorner:      (nchannels, 2, nspots) int ndarray, [[leftcorner], [uppercorners]] of each channel of the rois
        indcenter:      (nchannels, 2, nspots) int ndarray, [[xcenter], [ycenter]] of each channel of the roi
    RETURN:
        imout:          (nchannels, imszh, imszw, 3) int ndarray, rgb images with red roi square drawn on the gray-scale imin
    """
     
    imin = np.array(imin)
    nchannels, imszh, imszw = imin.shape
    nspots = len(intcenter[0][0])
    
    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_imdrawer')
    fname = 'imdrawer.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))

    imin = np.ascontiguousarray(imin.flatten(), dtype=np.uint16)
    ind_corner = np.ascontiguousarray(intcorner.flatten(), dtype=np.int32)
    ind_center = np.ascontiguousarray(intcenter.flatten(), dtype=np.int32)
    imout = np.ascontiguousarray(np.zeros(nchannels * imszh * imszw * 3), dtype=np.uint8)
    imdrawer_kernel = lib.im_drawer
    imdrawer_kernel.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                            ctl.ndpointer(np.uint16, flags="aligned, c_contiguous"),
                            ctypes.c_int32, ctypes.c_int32,
                            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"), 
                            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.uint8, flags="aligned, c_contiguous")]
    imdrawer_kernel(nchannels, imszh, imszw, imin, boxsz, nspots, ind_corner, ind_center, imout)
    return imout.reshape((nchannels, imszh, imszw, 3))



if __name__ == '__main__':

    rng = np.random.default_rng()

    # parameters
    boxsz = 9
    rendersz = 2 * boxsz + 1
    PSFsigma = 1.4
    
    nfrms = 10
    nchannels = 2
    imszh = 128
    imszw = 64
    nrois = 8

    # gauss template
    YY, XX = np.meshgrid(np.arange(-boxsz, boxsz+1), np.arange(-boxsz, boxsz+1), indexing='ij')
    gauss_temp = 65535.0 / 2.0 / np.pi / PSFsigma / PSFsigma * np.exp(-0.5 * XX * XX / PSFsigma / PSFsigma - 0.5 * YY * YY / PSFsigma / PSFsigma)
    gauss_temp = np.uint16(gauss_temp)
    
    if nchannels == 1:
        im_stacks = np.zeros((nfrms, imszh, imszw, 3), dtype=np.uint8)
    elif nchannels == 2:
        im_stacks = np.zeros((nfrms, imszh, imszw*2, 3), dtype=np.uint8)
    else:
        im_stacks = np.zeros((nfrms, imszh*2, imszw*2, 3), dtype=np.uint8)
    
    for f in range(nfrms):
        
        # centers and corners
        ind_centers = np.zeros((nchannels, 2, nrois), dtype=np.int32)
        for j in range(nchannels):
            ind_centers[j, 0] = rng.integers(boxsz, imszw-boxsz, size=nrois) 
            ind_centers[j, 1] = rng.integers(boxsz, imszh-boxsz, size=nrois)

        ind_corners = ind_centers.copy()
        ind_corners[:, 0] -= boxsz//2
        ind_corners[:, 1] -= boxsz//2

        # imin
        imin = np.zeros((nchannels, imszh, imszw), dtype=np.uint16)
        for j in range(nchannels):
            for i in range(nrois):
                indx = ind_centers[j, 0, i]
                indy = ind_centers[j, 1, i]
                imin[j, indy-boxsz:indy+boxsz+1, indx-boxsz:indx+boxsz+1] += gauss_temp

        # draw
        imout = im_draw(imin, boxsz, ind_corners, ind_centers)
        if nchannels == 1:
            im_stacks[f] = imout[0]
        elif nchannels == 2:
            im_stacks[f, :, :imszw] = imout[0]
            im_stacks[f, :, imszw:] = imout[1]
        else:
            im_stacks[f, :imszh, :imszw] = imout[0]
            im_stacks[f, :imszh, imszw:] = imout[1]
            im_stacks[f, imszh:, :imszw] = imout[2]
            im_stacks[f, imszh:, imszw:] = imout[3]

    with TiffWriter("test.tif") as tf:
        for f in range(nfrms):
            tf.save(im_stacks[f])
    

        
    