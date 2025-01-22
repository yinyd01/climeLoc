import numpy as np
from scipy.interpolate import LSQUnivariateSpline



def _spline1d(xdata, ydata, nbreaks, sorted=True):
    """
    spline fit of ydata to f(xdata) via LSQunivariateSpline
    INPUT:
        xdata:      the input 1D datavec, will be sorted asccendingly if sorted is False
        ydata:      the input 1D datavec for spline fit at xdata correspondingly
        nbreaks:    number of pieces for setting up the knots
        sorted:     indicating if the datavec is ascendingly sorted, default is True
    RETURN:
        fspline:    the fitted spline function object  
    """
    
    n = len(xdata)
    if sorted is False:
        
        _dumind = np.argsort(xdata)
        xdata, ydata = xdata[_dumind], ydata[_dumind]
        xdata_new, ydata_new = [], []
        
        ind1 = 0
        ind0 = ind1
        while ind1 < n:
            while (ind1 < n) and (xdata[ind1] == xdata[ind0]):
                ind1 += 1
            xdata_new.append(xdata[ind0])
            ydata_new.append(np.mean(ydata[ind0:ind1]))
            ind0 = ind1
        
        xdata, ydata = np.array(xdata_new), np.array(ydata_new)
        n = len(xdata)
    
    _dumind = np.int32(np.linspace(0, n, nbreaks+1, endpoint=True))
    interknots = xdata[_dumind[1:-1]]
    fspline = LSQUnivariateSpline(xdata, ydata, interknots)

    return fspline



def _get_splknl(D, CFlag):
    """
    get BSpline filter kernel
    INPUT:
        D:              integer, degree of BSpline. D can be any integer in the range [0,5].
        CFlag:          logical, true for centred BSplines, false for shifted. BSplines.  Default CFlag = true.
    RETURN:
        F :             1d ndarray, the convolution kernel used to implement the BSpline transform. 
    NOTE: Originally implemented in matlab by Nathan D. Cahill (ndcahill@gmail.com) 18 April 2008
    """
    
    if D == 0:
        F = [1.0]
    elif D == 1:
        F = [1.0] if CFlag else [1/2, 1/2]
    elif D == 2:
        F = [1/8, 3/4, 1/8] if CFlag else [1/2, 1/2]
    elif D == 3:
        F = [1/6, 2/3, 1/6] if CFlag else [1/48, 23/48, 23/48, 1/48]
    elif D == 4:
        F = [1/384, 19/96, 115/192, 19/96, 1/384] if CFlag else [1/24, 11/24, 11/24, 1/24]
    elif D == 5:
        F = [1/120, 13/60, 33/60, 13/60, 1/120] if CFlag else [1/3840, 79/1280, 841/1920, 841/1920, 79/1280, 1/3840]
    
    return np.asarray(F)



def _get_spl_coeff(D, lmbda=0.0):
    """
    _get_spl_coeff: get BSpline filter coefficients (for centered BSpline only)
    lmbda is forced to 0 for even degree
    INPUT:
        D:              integer, degree of BSpline. D can be any integer in the range [0,5].
        lmbda:          smoothing parameter.  Default lambda = 0. (forced to be 0 if D is even)
    RETURN:
        F:              1d ndarray, a vector of the roots of the z-transform of the direct BSpline filter.
        C0:             float, the numerator of the direct BSpline filter.
    NOTE: Originally implemented in matlab by Nathan D. Cahill (ndcahill@gmail.com) 18 April 2008
    """
    
    # non-zero lmbdas are ignored for even degrees
    if D == 0:
        F, C0 = [], 1.0
    
    elif D == 1:
        if lmbda == 0:
            F, C0 = [], 1.0
        else:
            F = [1+1/(2*lmbda)-np.sqrt(1+4*lmbda)/(2*lmbda)]
            C0 = -1/lmbda
    
    elif D == 2:
        F, C0 = [2*np.sqrt(2)-3], 8.0

    elif D == 3:
        if lmbda == 0:
            F, C0 = [np.sqrt(3)-2], 6.0
        else:
            p = [1, -4, 6, -4, 1]
            p[1] += 1/(6*lmbda)
            p[2] += 2/(3*lmbda)
            p[3] += 1/(6*lmbda)
            dumF = np.roots(p)
            F = dumF[np.abs(dumF) <= 1]
            C0 = 1/lmbda
    
    elif D == 4:
        F, C0 = [-0.36134122590022, -0.0137254292973391], 384

    elif D == 5:
        if lmbda == 0:
            F, C0 = [-0.430575347099973 -0.0430962882032647], 120
        else:
            p = [1, -6, 15, -20, 15, -6, 1]
            p[1] -= 1/(120*lmbda)
            p[2] -= 13/(60*lmbda)
            p[3] -= 11/(20*lmbda)
            p[4] -= 13/(60*lmbda)
            p[5] -= 1/(120*lmbda)
            dumF = np.roots(p)
            F = dumF[np.abs(dumF) <= 1]
            C0 = -1/lmbda
        
    return np.array(F), C0
    


def _spl_eval(xi, deg):
    """
    evaluate BSpline basis function
    INPUT:
        xi:     ndarray, positions at which to evaluate BSpline basis function
        deg:    integer, degree of desired basis function, only suport for 0=<deg<=5
    RETURN:
        bi:     BSpline basis function
    NOTE: Originally implemented in matlab by Nathan D. Cahill (ndcahill@gmail.com) 18 April 2008
    """
    xi = np.asarray(xi, dtype=np.float64)
    bi = np.zeros_like(xi)
    
    if deg == 0:
        ind = (xi>=-1/2) & (xi < 1/2)
        bi[ind] = 1.0
    
    elif deg == 1:
        ind = (xi>=0) & (xi<1)
        bi[ind] = 1.0 - xi[ind]
        ind = (xi>=-1) & (xi<0)
        bi[ind] = 1.0 + xi[ind]
    
    elif deg == 2:
        x2 = xi*xi
        ind = (xi>=1/2) & (xi<3/2)
        bi[ind] = 9/8 - 3/2*xi[ind] + 1/2*x2[ind]
        ind = (xi>=-1/2) & (xi<1/2)
        bi[ind] = 3/4 - x2[ind]
        ind = (xi>=-3/2) & (xi<-1/2)
        bi[ind] = 9/8 + 3/2*xi[ind] + 1/2*x2[ind]
    
    elif deg == 3:
        x2 = xi*xi
        x3 = x2*xi
        ind = (xi>=1) & (xi<2)
        bi[ind] = 4/3 - 2*xi[ind] + x2[ind] - 1/6*x3[ind]
        ind = (xi>=0) & (xi<1)
        bi[ind] = 2/3 - x2[ind] + 1/2*x3[ind]
        ind = (xi>=-1) & (xi<0)
        bi[ind] = 2/3 - x2[ind] - 1/2*x3[ind]
        ind = (xi>=-2) & (xi<-1)
        bi[ind] = 4/3 + 2*xi[ind] + x2[ind] + 1/6*x3[ind]

    elif deg == 4:
        x2 = xi*xi
        x3 = x2*xi
        x4 = x3*xi
        ind = (xi>=3/2) & (xi<5/2)
        bi[ind] = 625/384 - 125/48*xi[ind] + 25/16*x2[ind] - 5/12*x3[ind] + 1/24*x4[ind]
        ind = (xi>=1/2) & (xi<3/2)
        bi[ind] = 55/96 + 5/24*xi[ind] - 5/4*x2[ind] + 5/6*x3[ind] - 1/6*x4[ind]
        ind = (xi>=-1/2) & (xi<1/2)
        bi[ind] = 115/192 - 5/8*x2[ind] + 1/4*x4[ind]
        ind = (xi>=-3/2) & (xi<-1/2)
        bi[ind] = 55/96 - 5/24*xi[ind] - 5/4*x2[ind] - 5/6*x3[ind] - 1/6*x4[ind]
        ind = (xi>=-5/2) & (xi<-3/2)
        bi[ind] = 625/384 + 125/48*xi[ind] + 25/16*x2[ind] + 5/12*x3[ind] + 1/24*x4[ind]

    elif deg == 5:
        x2 = xi*xi 
        x3 = x2*xi
        x4 = x3*xi
        x5 = x4*xi
        ind = (xi>=2) & (xi<3)
        bi[ind] = 81/40 - 27/8*xi[ind] + 9/4*x2[ind] - 3/4*x3[ind] + 1/8*x4[ind] - 1/120*x5[ind]
        ind = (xi>=1) & (xi<2)
        bi[ind] = 17/40 + 5/8*xi[ind] - 7/4*x2[ind] + 5/4*x3[ind] - 3/8*x4[ind] + 1/24*x5[ind]
        ind = (xi>=0) & (xi<1)
        bi[ind] = 11/20 - 1/2*x2[ind] + 1/4*x4[ind] - 1/12*x5[ind]
        ind = (xi>=-1) & (xi<0)
        bi[ind] = 11/20 - 1/2*x2[ind] + 1/4*x4[ind] + 1/12*x5[ind]
        ind = (xi>=-2) & (xi<-1)
        bi[ind] = 17/40 - 5/8*xi[ind] - 7/4*x2[ind] - 5/4*x3[ind] - 3/8*x4[ind] - 1/24*x5[ind]
        ind = (xi>=-3) & (xi<-2)
        bi[ind] = 81/40 + 27/8*xi[ind] + 9/4*x2[ind] + 3/4*x3[ind] + 1/8*x4[ind] + 1/120*x5[ind]
    
    else:
        dum1 = (xi + (deg+1)/2) * _spl_eval(xi+1/2, deg-1)
        dum2 = ((deg+1)/2 - xi) * _spl_eval(xi-1/2, deg-1)
        bi = (dum1 + dum2) /deg
    
    return bi



def _symExpFilt(X, C0, Zi, K0, KVec):
    """
    symmetric exponential filter
    INPUT:
        X:      1D ndarray of input data
        C0:     scaling constant
        Zi:     pole of direct BSpline filter
        K0:     parameter for computing initial condition
        KVec:   vector of indices for computing initial condition (reflects from boundaries if necessary)
    RETURN:
        Y:      vector of output data
    
    NOTE: This function implements the recursive filter in Equations (2.5) and (2.6) of 
    M. Unser, A. Aldroubi, and M. Eden, "B-Spline Signal Processing: Part II - Efficient Design and Applications," 
    IEEE Trans. Signal Processing, 41(2):834-848, February 1993.
    
    Originally implemented in matlab by Nathan D. Cahill (ndcahill@gmail.com), 18 April 2008
    """
    
    XLen = len(X)
    Y = np.zeros(XLen, dtype=np.complex128)

    # first element in Y
    for k in range(K0):
        Y[0] += Zi**k * X[KVec[k]]
       
    # filter in forward direction
    for k in range(1, XLen):
        Y[k] = X[k] + Zi * Y[k-1]
    
    # update the last element of Y
    Y[XLen-1] = (2*Y[XLen-1] - X[XLen-1]) * C0
    
    # filter in reverse direction
    for k in range(XLen-2, -1, -1):
        Y[k] = (Y[k+1] - Y[k]) * Zi
    
    return Y



def _get_spl_filt1d(A, D, lmbda):
    """
    compute B-spline smoother of the 1D data array
    padding mode is 'reflect'
    INPUTS: 
        A:              1d ndarray for generating the smoother
        D:              integer, degrees of each dimension of tensor product BSpline.
        lmbda:          float, smoothing factor for each dimension
    RETURN:
        bspldict:       'coeff':    array of BSpline coefficients
                        'deg':      D
    
    NOTE: Originally implemented in matlab by Nathan D. Cahill (ndcahill@gmail.com) 18 April 2008
    """
    
    sz = len(A)
    indReflect = np.hstack((np.arange(sz), np.arange(sz-2,0,-1)))

    if (D<2) and (lmbda == 0):
        return np.pad(A, (0, 1), mode='reflect')

    # get coefficients for BSpline filters for each dimension
    F, F0 = _get_spl_coeff(D, lmbda)
    
    # initialize the output and the tolorence
    C = np.copy(A)
    K0Tol = np.finfo(float).eps

    
    ## direct filtering
    # loop through poles of direct BSpline filter
    C = np.asarray(C, dtype=np.complex128)
    for i in range(len(F)):
        
        # KVec for the current pole
        K0 = np.int_(-(-np.log(K0Tol)//np.log(np.abs(F[i]))))
        numReps = -(-K0//(2*sz-2))
        KVec = np.tile(indReflect, numReps)
        KVec = KVec[:K0]
                    
        # Scaling factor for current pole
        C0 = -F[i] / (1 - F[i]*F[i])
        
        # symmetric exponential filter for each row
        C = _symExpFilt(C, C0, F[i], K0, KVec)
    
    # multiply by numerator of direct BSpline filter
    C.real *= F0
    

    ## pad dimensions by reflection
    padNum = D//2
    ind = indReflect[np.arange(-padNum, sz+padNum+1) % (2*sz-2)]
    coeff = C.real[ind]

    BSplineFilt = {'coeff':coeff, 'deg':D, 'lmbda':lmbda}

    return BSplineFilt



def _get_spl_filt2d(A, D, lmbda):
    """
    compute B-spline smoother of the 2D data array
    padding mode is 'reflect'
    INPUTS: 
        A:              2d ndarray for generating the smoother
        D:              tuple, (degy, degx), degrees of each dimension of tensor product BSpline.
        lmbda:          tuple, (lmbday, lmbdax), smoothing factor for each dimension
    RETURN:
        C:              array of BSpline coefficients
    
    NOTE: Originally implemented in matlab by Nathan D. Cahill (ndcahill@gmail.com) 18 April 2008
    """
    # check singleton dimentions
    if np.any(A.shape == 1):
        print("the shape of the input data array is")
        print(A.shape)
        raise TypeError("The input data array contains singleton dimentions")
    
    szy, szx = A.shape
    indReflecty = np.hstack((np.arange(szy), np.arange(szy-2,0,-1)))
    indReflectx = np.hstack((np.arange(szx), np.arange(szx-2,0,-1)))

    if (np.all(np.array(D) < 2)) and np.all(np.array(lmbda) == 0):
        return np.pad(A, (0, 1), mode='reflect')

    # get coefficients for BSpline filters for each dimension
    degy, degx = D
    lmbday, lmbdax = lmbda
    
    Fy, F0y = _get_spl_coeff(degy, lmbday)
    Fx, F0x = _get_spl_coeff(degx, lmbdax)
    
    # initialize the output and the tolorence
    C = np.copy(A)
    C = np.asarray(C, dtype=np.complex128)
    K0Tol = np.finfo(float).eps

    
    ## direct filtering over each row
    # loop through poles of direct BSpline filter
    for j in range(len(Fx)):
        
        # KVec for the current pole
        K0 = np.int_(-(-np.log(K0Tol)//np.log(np.abs(Fx[j]))))
        numReps = -(-K0//(2*szx-2))
        KVec = np.tile(indReflectx, numReps)
        KVec = KVec[:K0]
                    
        # Scaling factor for current pole
        C0 = -Fx[j] / (1 - Fx[j]*Fx[j])
        
        # symmetric exponential filter for each row
        for i in range(szy):
            C[i, :] = _symExpFilt(C[i, :], C0, Fx[j], K0, KVec)
    
    # multiply by numerator of direct BSpline filter
    C.real *= F0x
    

    ## direct filtering over each coloumn
    # loop through poles of direct BSpline filter
    for i in range(len(Fy)):
        
        # KVec for current pole
        K0 = np.int_(-(-np.log(K0Tol)//np.log(np.abs(Fy[i]))))
        numReps = -(-K0//(2*szy-2))
        KVec = np.tile(indReflecty, numReps)
        KVec = KVec[:K0]
                    
        # Scaling factor for current pole
        C0 = -Fy[i] / (1 - Fy[i]*Fy[i])
        
        # symmetric exponential filter for each column
        for j in range(szx):
            C[:, j] = _symExpFilt(C[:, j], C0, Fy[i], K0, KVec)
    
    # multiply by numerator of direct BSpline filter
    C.real *= F0y
    
    ## pad dimensions by reflection
    padNumy, padNumx = degy//2, degx//2
    indy = indReflecty[np.arange(-padNumy, szy+padNumy+1) % (2*szy-2)]
    indx = indReflectx[np.arange(-padNumx, szx+padNumx+1) % (2*szx-2)]
    coeff = C.real[indy[...,np.newaxis], indx[np.newaxis,...]]

    BSplineFilt = {'coeff':coeff, 'deg':D, 'lmbda':lmbda}
    return BSplineFilt



def _get_spl_filt3d(A, D, lmbda):
    """
    compute B-spline smoother of the 3D data array
    padding mode is 'reflect's
    INPUTS: 
        A:              3d ndarray for generating the smoother
        D:              tuple, (degz, degy, degx), degrees of each dimension of tensor product BSpline.
        lmbda:          tuple, (lmbdaz, lmbday, lmbdax), smoothing factor for each dimension
    RETURN:
        C:              array of BSpline coefficients
    NOTE: Originally implemented in matlab by Nathan D. Cahill (ndcahill@gmail.com) 18 April 2008
    """
    # check singleton dimentions
    if np.any(A.shape == 1):
        print("the shape of the input data array is")
        print(A.shape)
        raise TypeError("The input data array contains singleton dimentions")
    
    szz, szy, szx = A.shape
    indReflectz = np.hstack((np.arange(szz), np.arange(szz-2,0,-1)))
    indReflecty = np.hstack((np.arange(szy), np.arange(szy-2,0,-1)))
    indReflectx = np.hstack((np.arange(szx), np.arange(szx-2,0,-1)))
    
    if (np.all(np.array(D) < 2)) and np.all(np.array(lmbda) == 0):
        return np.pad(A, (0, 1), mode='reflect')

    # get coefficients for BSpline filters for each dimension
    degz, degy, degx = D
    lmbdaz, lmbday, lmbdax = lmbda
    
    Fz, F0z = _get_spl_coeff(degz, lmbdaz)
    Fy, F0y = _get_spl_coeff(degy, lmbday)
    Fx, F0x = _get_spl_coeff(degx, lmbdax)
    
    # initialize the output and the tolorence
    C = np.copy(A)
    C = np.asarray(C, dtype=np.complex128)
    K0Tol = np.finfo(float).eps

    
    ## direct filtering over each row
    # loop through poles of direct BSpline filter
    for k in range(len(Fx)):
        
        # KVec for the current pole
        K0 = np.int_(-(-np.log(K0Tol)//np.log(np.abs(Fx[k]))))
        numReps = -(-K0//(2*szx-2))
        KVec = np.tile(indReflectx, numReps)
        KVec = KVec[:K0]
                    
        # Scaling factor for current pole
        C0 = -Fx[k] / (1 - Fx[k]*Fx[k])
        
        # symmetric exponential filter for each row
        for i in range(szz):
            for j in range(szy):
                C[i, j, :] = _symExpFilt(C[i, j, :], C0, Fx[k], K0, KVec)
    
    # multiply by numerator of direct BSpline filter
    C.real *= F0x

    ## direct filtering over each coloumn
    # loop through poles of direct BSpline filter
    for j in range(len(Fy)):
        
        # KVec for current pole
        K0 = np.int_(-(-np.log(K0Tol)//np.log(np.abs(Fy[j]))))
        numReps = -(-K0//(2*szy-2))
        KVec = np.tile(indReflecty, numReps)
        KVec = KVec[:K0]
                    
        # Scaling factor for current pole
        C0 = -Fy[j] / (1 - Fy[j]*Fy[j])
        
        # symmetric exponential filter for each column
        for i in range(szz):
            for k in range(szx):
                C[i, :, k] = _symExpFilt(C[i, :, k], C0, Fy[j], K0, KVec)
    
    # multiply by numerator of direct BSpline filter
    C.real *= F0y
    
    ## direct filtering over each slice
    # loop through poles of direct BSpline filter
    for i in range(len(Fz)):
        
        # KVec for current pole
        K0 = np.int_(-(-np.log(K0Tol)//np.log(np.abs(Fz[i]))))
        numReps = -(-K0//(2*szz-2))
        KVec = np.tile(indReflectz, numReps)
        KVec = KVec[:K0]
                    
        # scaling factor for current pole
        C0 = -Fz[i] / (1 - Fz[i]*Fz[i])
        
        # symmetric exponential filter for each column
        for j in range(szy):
            for k in range(szx):
                C[:, j, k] = _symExpFilt(C[:, j, k], C0, Fz[i], K0, KVec)
    
    # multiply by numerator of direct BSpline filter
    C.real *= F0z
    
    
    ## pad dimensions by reflection
    padNumz, padNumy, padNumx = degz//2, degy//2, degx//2
    indz = indReflectz[np.arange(-padNumz, szz+padNumz+1) % (2*szz-2)]
    indy = indReflecty[np.arange(-padNumy, szy+padNumy+1) % (2*szy-2)]
    indx = indReflectx[np.arange(-padNumx, szx+padNumx+1) % (2*szx-2)]
    coeff = C.real[indz[...,np.newaxis,np.newaxis], indy[np.newaxis,...,np.newaxis], indx[np.newaxis,np.newaxis,...]]
    
    BSplineFilt = {'coeff':coeff, 'deg':D, 'lmbda':lmbda}
    return BSplineFilt



def BSplineSmooth1D(A, D, lmbda, xi=None):
    """
    smooth the input 1d ndarray BSpline interpolation
    padding mode is 'reflect'
    INPUT:
        A:              1d gridded ndarray for smooth
        D:              integer, degree of tensor product BSpline.
        lmbda:          float, smoothing factor
        xi:             1d ndarray, input 1d data for bspline interpolation of the smoothed A
                        return smoothed A if xi is None (default)
    RETURN:
        xo:             1d ndarray, the interploation of the smoothed data corresponding to xi
    """
    smoother = _get_spl_filt1d(A, D, lmbda)
    Coeff = smoother['coeff']
    deg = smoother['deg']

    sz_Coeff, padNum = len(Coeff), deg//2
    sz_Data = sz_Coeff - (2*padNum+1)
    ind_offset = padNum

    # round off the edges
    if not (xi is None):
        xi = np.asarray(xi, dtype=np.float64)
        xi[xi<0] = 0
        xi[xi>sz_Data] = sz_Data
        
        spID = ind_offset + np.int_(xi)
        delta = xi - np.int_(xi)
        spID[xi == sz_Data] -= 1
        delta[xi == sz_Data] = 1.0

        xo = np.zeros_like(xi)
    
    else:
        spID = ind_offset + np.arange(sz_Data)
        delta = 0.0
        xo = np.zeros(sz_Data, dtype=np.float64)
    
    # BSpline smooth
    for i in range(-(-(deg+1)//2)):
        delta0, spID0 = delta+i, spID-i
        delta1, spID1 = delta-(i+1), spID+(i+1)
        
        Bx0 = _spl_eval(delta0, deg)
        Bx1 = _spl_eval(delta1, deg)

        xo += Coeff[spID0]*Bx0 + Coeff[spID1]*Bx1
    
    return xo



def BSplineSmooth2D(A, D, lmbda, xi=None):
    """
    smooth the input 1d ndarray BSpline interpolation
    padding mode is 'reflect'
    INPUT:
        A:              2d gridded ndarray for smooth
        D:              tuple (degy, degx), degree of tensor product BSpline for each dimension
                        put deg as 0 if no smoothing is expected along a certain axis
        lmbda:          tuple (lmbday, lmbdax), smoothing factor for each dimension
        xi:             tuple (YY, XX), YY and XX are 2d grided ndarray for bspline interpolation of the smoothed data
                        return smoothed A if xi is None (default)
    RETURN:
        xo:             2d ndarray, the interploation of the smoothed data corresponding to xi
    """
    smoother = _get_spl_filt2d(A, D, lmbda)
    coeff = smoother['coeff']
    degy, degx = smoother['deg']

    szy_coeff, szx_coeff = coeff.shape 
    padNumy, padNumx = degy//2, degx//2
    szy_data, szx_data = szy_coeff-(2*padNumy+1), szx_coeff-(2*padNumx+1)
    indy_offset, indx_offset = padNumy, padNumx

    # round off the edges
    if not (xi is None):
        YY, XX = np.asarray(xi, dtype=np.float64)
        YY[YY<0], XX[XX<0] = 0.0, 0.0
        YY[YY>szy_data], XX[XX>szx_data] = szy_data, szx_data
        
        spIDy, spIDx = indy_offset+np.int_(YY), indx_offset + np.int_(XX)
        deltay, deltax = YY-np.int_(YY), XX-np.int_(XX)
        spIDy[YY == szy_data] -= 1
        spIDx[XX == szx_data] -= 1
        deltay[YY == szy_data] = 1.0
        deltax[XX == szx_data] = 1.0

        xo = np.zeros_like(YY)
    
    else:
        YY = np.repeat(np.arange(szy_data), szx_data).reshape(szy_data, szx_data)
        XX = np.tile(np.arange(szx_data), szy_data).reshape(szy_data, szx_data)
        spIDy, spIDx = indy_offset+YY, indx_offset+XX
        deltay, deltax = 0.0, 0.0
        xo = np.zeros((szy_data, szx_data), dtype=np.float64)
    
    # BSpline smooth
    for i in range(-(-(degy+1)//2)):
        deltay0, spIDy0 = deltay+i, spIDy-i
        deltay1, spIDy1 = deltay-(i+1), spIDy+(i+1)
        By0 = _spl_eval(deltay0, degy)
        By1 = _spl_eval(deltay1, degy)
        
        for j in range(-(-(degx+1)//2)):
            deltax0, spIDx0 = deltax+j, spIDx-j
            deltax1, spIDx1 = deltax-(j+1), spIDx+(j+1)
            Bx0 = _spl_eval(deltax0, degx)
            Bx1 = _spl_eval(deltax1, degx)

            xo += coeff[spIDy0, spIDx0]*By0*Bx0 + coeff[spIDy0, spIDx1]*By0*Bx1 + \
                    coeff[spIDy1, spIDx0]*By1*Bx0 + coeff[spIDy1, spIDx1]*By1*Bx1
    
    return xo



def BSplineSmooth3D(A, D, lmbda, xi=None):
    """
    smooth the input 1d ndarray BSpline interpolation
    padding mode is 'reflect'
    INPUT:
        A:              3d gridded ndarray for smooth
        D:              tuple (degz, degy, degx), degree of tensor product BSpline for each dimension
                        put deg as 0 if no smoothing is expected along a certain axis
        lmbda:          tuple (lmbdaz, lmbday, lmbdax), smoothing factor for each dimension
        xi:             tuple (ZZZ, YYY, XXX), ZZZ, YYY, and XXX are 3d grided ndarray for bspline interpolation of the smoothed data
                        return smoothed A if xi is None (default)
    RETURN:
        xo:             3d ndarray, the interploation of the smoothed data corresponding to xi
    """
    smoother = _get_spl_filt3d(A, D, lmbda)
    coeff = smoother['coeff']
    degz, degy, degx = smoother['deg']

    szz_coeff, szy_coeff, szx_coeff = coeff.shape 
    padNumz, padNumy, padNumx = degz//2, degy//2, degx//2
    szz_data, szy_data, szx_data = szz_coeff-(2*padNumz+1), szy_coeff-(2*padNumy+1), szx_coeff-(2*padNumx+1)
    indz_offset, indy_offset, indx_offset = padNumz, padNumy, padNumx

    # round off the edges
    if not (xi is None):
        ZZZ, YYY, XXX = np.asarray(xi, dtype=np.float64)
        ZZZ[ZZZ<0], YYY[YYY<0], XXX[XXX<0] = 0.0, 0.0, 0.0
        ZZZ[ZZZ>szz_data], YYY[YYY>szy_data], XXX[XXX>szx_data] = szz_data, szy_data, szx_data
        
        spIDz, spIDy, spIDx = indz_offset+np.int_(ZZZ), indy_offset+np.int_(YYY), indx_offset+np.int_(XXX)
        deltaz, deltay, deltax = ZZZ-np.int_(ZZZ), YYY-np.int_(YYY), XXX-np.int_(XXX)
        spIDz[ZZZ == szz_data] -= 1
        spIDy[YYY == szy_data] -= 1
        spIDx[XXX == szx_data] -= 1
        deltaz[ZZZ == szz_data] = 1.0
        deltay[YYY == szy_data] = 1.0
        deltax[XXX == szx_data] = 1.0

        xo = np.zeros_like(ZZZ)
    
    else:
        ZZZ = np.repeat(np.arange(szz_data), szy_data*szx_data).reshape(szz_data, szy_data, szx_data)
        YYY = np.tile(np.repeat(np.arange(szy_data), szx_data), szz_data).reshape(szz_data, szy_data, szx_data)
        XXX = np.tile(np.arange(szx_data), szy_data*szz_data).reshape(szz_data, szy_data, szx_data)
        spIDz, spIDy, spIDx = indz_offset+ZZZ, indy_offset+YYY, indx_offset+XXX
        deltaz, deltay, deltax = 0.0, 0.0, 0.0
        xo = np.zeros((szz_data, szy_data, szx_data), dtype=np.float64)
    
    # BSpline smooth
    for i in range(-(-(degz+1)//2)):
        deltaz0, spIDz0 = deltaz+i, spIDz-i
        deltaz1, spIDz1 = deltaz-(i+1), spIDz+(i+1)
        Bz0 = _spl_eval(deltaz0, degz)
        Bz1 = _spl_eval(deltaz1, degz)

        for j in range(-(-(degy+1)//2)):
            deltay0, spIDy0 = deltay+j, spIDy-j
            deltay1, spIDy1 = deltay-(j+1), spIDy+(j+1)
            By0 = _spl_eval(deltay0, degy)
            By1 = _spl_eval(deltay1, degy)
            
            for k in range(-(-(degx+1)//2)):
                deltax0, spIDx0 = deltax+k, spIDx-k
                deltax1, spIDx1 = deltax-(k+1), spIDx+(k+1)
                Bx0 = _spl_eval(deltax0, degx)
                Bx1 = _spl_eval(deltax1, degx)

                xo += coeff[spIDz0, spIDy0, spIDx0]*Bz0*By0*Bx0 + coeff[spIDz0, spIDy0, spIDx1]*Bz0*By0*Bx1 + \
                        coeff[spIDz0, spIDy1, spIDx0]*Bz0*By1*Bx0 + coeff[spIDz0, spIDy1, spIDx1]*Bz0*By1*Bx1 + \
                        coeff[spIDz1, spIDy0, spIDx0]*Bz1*By0*Bx0 + coeff[spIDz1, spIDy0, spIDx1]*Bz1*By0*Bx1 + \
                        coeff[spIDz1, spIDy1, spIDx0]*Bz1*By1*Bx0 + coeff[spIDz1, spIDy1, spIDx1]*Bz1*By1*Bx1
    
    return xo