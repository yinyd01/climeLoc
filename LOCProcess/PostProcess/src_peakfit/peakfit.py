import os
import inspect
import ctypes
import numpy as np
from scipy import stats
import numpy.ctypeslib as ctl


def peakfit(img, MAX_ITERS=100):
    """
    fit the input img with a 2D Gauss kernel
    """
    vnum = 7 # constant
    
    h_data = np.asarray(img, dtype=np.float32)
    if h_data.ndim < 2:
        raise TypeError('PSF should be cropped into a 2D square, but the input is an 1D array\n')
    if h_data.ndim == 2:
        NFits = 1 
        imHeight, imWidth = h_data.shape
    elif h_data.ndim > 2:
        NFits, imHeight, imWidth = h_data.shape
    
    tinv095 = stats.t.ppf(0.975, imHeight * imWidth - vnum)
    
    h_data = np.ascontiguousarray(h_data.flatten(), dtype=np.float32)
    h_xvec = np.ascontiguousarray(np.zeros(NFits * vnum), dtype=np.float32)
    h_Loss = np.ascontiguousarray(np.zeros(NFits), dtype=np.float32)
    h_crlb = np.ascontiguousarray(np.zeros(NFits * vnum), dtype=np.float32)
    h_ci = np.ascontiguousarray(np.zeros(NFits * vnum), dtype=np.float32)

    ## Load cuda module file
    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fname = 'peakfit.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    peakfit_kernel = lib.kernel
    peakfit_kernel.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, 
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                            ctypes.c_float, ctypes.c_int32,
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                            ctl.ndpointer(np.float32, flags="aligned, c_contiguous")]
    peakfit_kernel(NFits, imHeight, imWidth, h_data, tinv095, MAX_ITERS, h_xvec, h_Loss, h_crlb, h_ci)
    return h_xvec.reshape(NFits, vnum), h_Loss, h_crlb.reshape(NFits, vnum), h_ci.reshape(NFits, vnum)



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



if __name__ == '__main__':
    
    from numpy.random import default_rng
    from matplotlib import pyplot as plt
    import time
    
    rng = default_rng()

    NFits, imWidth, imHeight = 8192, 13, 15
    xc, yc, rho = 0.5*imWidth, 0.5*imHeight, 0.7
    sigmax, sigmay = 0.1*imWidth, 0.2*imHeight
    Intensity = 1000.0
    bkg = 10.0

    XX = np.tile(np.arange(imWidth)+0.5, imHeight).reshape(imHeight, imWidth)
    YY = np.repeat(np.arange(imHeight)+0.5, imWidth).reshape(imHeight, imWidth)
    VNUM = 7
    GTvec = np.array([xc, yc, sigmax, sigmay, rho, Intensity, bkg])
    GTpara = ['err_x', 'err_y', 'err_sx', 'err_sy', 'err_rho', 'err_I', 'err_b']
    dumXX = (XX - xc) / sigmax
    dumYY = (YY - yc) / sigmay
    
    dum = Intensity / (2.0*np.pi*sigmax*sigmay*np.sqrt(1-rho*rho)) * np.exp(-0.5/(1-rho*rho)*(dumXX*dumXX - 2.0*rho*dumXX*dumYY + dumYY*dumYY)) + bkg
    dum = np.tile(dum[np.newaxis,...], (NFits, 1, 1))
    dum += rng.normal(loc=0.0, scale=5.0, size=(NFits, imHeight, imWidth))
    fig, ax = plt.subplots()
    ax.imshow(dum[10])
    plt.show()

    tas = time.time()
    xvec, loss, crlb, ci = peakfit(dum)
    print("{t:f} secs ellapsed by PeakFit_Gauss for {NFits:d} {szy:d}x{szx:d} PSFs".format(t=time.time()-tas, NFits=NFits, szy=imHeight, szx=imWidth))
    
    fig, axs = plt.subplots(3, VNUM+1)
    for i in range(VNUM):
        axs[0, i].plot(np.arange(NFits), xvec[:, i]-GTvec[i])
        txtPosx, txtPosy = _textPos(axs[0, i], (0.05, 0.05))
        axs[0, i].text(txtPosx, txtPosy, 'RMSE={acc:.2f}'.format(acc=np.sqrt(np.mean((xvec[:, i]-GTvec[i])*(xvec[:, i]-GTvec[i])))))
        axs[0, i].set_title(GTpara[i])
    axs[0, VNUM].plot(np.arange(NFits), loss)
    axs[0, VNUM].set_title('Loss')
    for i in range(VNUM):
        axs[1, i].plot(np.arange(NFits), np.sqrt(crlb[:, i]))
        txtPosx, txtPosy = _textPos(axs[1, i], (0.05, 0.05))
        axs[1, i].text(txtPosx, txtPosy, 'sCRLB={crlb:.2f}'.format(crlb=np.sqrt(np.mean(crlb[:, i]))))
    
    for i in range(VNUM):
        axs[2, i].plot(np.arange(NFits), ci[:, i])
        txtPosx, txtPosy = _textPos(axs[2, i], (0.05, 0.05))
        axs[2, i].text(txtPosx, txtPosy, 'ci={ci:.2f}'.format(ci=np.mean(ci[:, i])))
    plt.show()
