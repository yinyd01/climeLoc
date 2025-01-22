import numpy as np
from scipy.optimize import curve_fit



def astigmatic(zdata, sigmax0, sigmay0, shiftx, shifty, dof, Ax, Bx, Ay, By):
    """
    To build the Astigmatic model
    INPUT:
        zdata:      (ndata,) ndarray, the user input z position array (sorted)
        sigmax0:    float, the minimum gauss-sigma (x-axis) of the psf
        sigmay0:    float, the minimum gauss-sigma (y-axis) of the psf
        shiftx:     float, the z-position where the gauss-sigma (x-axis) of the psf drops to minimum
        shifty:     float, the z-position where the gauss-sigma (y-axis) of the psf drops to minimum
        dof:        float, the depth of the field
        Ax:         float, the coefficient for the 3rd-order correction of the astigmatic (x-axis) profile
        Bx:         float, the coefficient for the 4th-order correction of the astigmatic (x-axis) profile
        Ay:         float, the coefficient for the 3rd-order correction of the astigmatic (y-axis) profile
        By:         float, the coefficient for the 4th-order correction of the astigmatic (y-axis) profile
    RETURN:
        model:      (2*ndata,) ndarray, the Astigmatic model for x and y, respectively
    """
    zx = (zdata - shiftx) / dof
    zy = (zdata - shifty) / dof
    model = np.zeros(2 * len(zdata))
    model[: len(zdata)] = sigmax0 * np.sqrt(np.maximum(1.0 + zx*zx + Ax*zx*zx*zx + Bx*zx*zx*zx*zx, 1.0))
    model[len(zdata) : 2*len(zdata)] = sigmay0 * np.sqrt(np.maximum(1.0 + zy*zy + Ay*zy*zy*zy + By*zy*zy*zy*zy, 1.0)) 
    return model



def get_astig_parameters(zdata, sigmax, sigmay):
    """
    fit the sigmax-zdata, sigmay-zdata with _astigmatic
    INPUT:
        zdata:              (ndata,) ndarray, the user input z position array (sorted)
        sigmax:             (ndata,) ndarray, the gauss sigmas (x-axis) at each zdata
        sigmay:             (ndata,) ndarray, the gauss sigmas (y-axis) at each zdata
    RETURN:
        astig_parameters:   [sigmax0, sigmay0, shiftx, shifty, dof, Ax, Bx, Ay, By]
    """
    ind_shiftx = np.argmin(sigmax)
    ind_shifty = np.argmin(sigmay)
    pguess = np.array([sigmax[ind_shiftx], sigmay[ind_shifty], zdata[ind_shiftx], zdata[ind_shifty], 0.5*(zdata[-1]-zdata[0]), .0, .0, .0, .0])
    plower = np.array([0, 0, zdata[0], zdata[0], 0, -2.0, -2.0, -2.0, -2.0])
    pupper = np.array([1.5*sigmax.max(), 1.5*sigmay.max(), zdata[-1], zdata[-1], np.inf, 2.0, 2.0, 2.0, 2.0])
    astig_parameters = curve_fit(astigmatic, zdata, np.hstack((sigmax, sigmay)), p0=pguess, bounds=(plower, pupper))[0]
    return astig_parameters
    


def get_astig_z(zdata, astig_parameters):
    """
    determine the z where sigmax == sigmay
    INPUT:
        zdata:              (ndata,) ndarray, the user input z position array (sorted)
        astig_parameters:   [sigmax0, sigmay0, shiftx, shifty, dof, Ax, Bx, Ay, By] ndarray, see _astigmatic
    RETURN:
        zFocus:             int, the index of the zdata where sigmax == sigmay
    """
    sigmax_fit, sigmay_fit = astigmatic(zdata, *astig_parameters).reshape((2, -1))
    ind_shiftx = np.argmin(np.abs(astig_parameters[2]-zdata))
    ind_shifty = np.argmin(np.abs(astig_parameters[3]-zdata))
    sandwich_min = min(ind_shiftx, ind_shifty)
    sandwich_max = max(ind_shiftx, ind_shifty)
    foo = np.abs(sigmax_fit[sandwich_min:sandwich_max+1] - sigmay_fit[sandwich_min:sandwich_max+1])
    zFocus = np.int32(zdata[sandwich_min]) + np.argmin(foo)
    return zFocus