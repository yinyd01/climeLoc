import os
import inspect
import numpy as np
import ctypes
import numpy.ctypeslib as ctl 



def _mulpa_xvec_inbox(ndim, nchannels, boxsz, nnum, xvec, crlb):
    """
    in-palce filterout the out-box emitters
    INPUT:
        ndim:           int, number of dimensions
        nchannels:      int, number of imaging channels
        boxsz:          int, size of the box
        xvec:           (NFits, NMAX * (ndim + nchannels) + nchannels) float, the xvec from the MULPA fit
        crlb:           (NFits, NMAX * (ndim + nchannels) + nchannels) float, crlb corresponding to the xvec
        nnum:           (NFits) int ndarray, the number of fitted emiters for each fit
    OUTPUT:
        xvec_o:         (nnum.sum(), ndim + nchannels + nchannels) float ndarray, the rearranged xvec
        crlb_o:         (nnum.sum(), ndim + nchannels + nchannels) float ndarray, crlb corresponding to the xvec_o     
    NOTE:   almost 10 time faster than numpy
    """
    NFits = len(xvec)
    vmax = len(xvec[0])
    
    xvec = np.ascontiguousarray(xvec.flatten(), dtype=np.float32)
    crlb = np.ascontiguousarray(crlb.flatten(), dtype=np.float32)
    nnum = np.ascontiguousarray(nnum, dtype=np.int32)
    
    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_xvec_unpack')
    fname = 'xvec_unpack.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    kernel = lib.xvec_inroi
    kernel.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                        ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),
                        ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                        ctl.ndpointer(np.float32, flags="aligned, c_contiguous")]
    kernel(NFits, boxsz, vmax, ndim, nchannels, nnum, xvec, crlb)
    return nnum, xvec.reshape((NFits, vmax)), crlb.reshape((NFits, vmax)) 



def _mulpa_xvec_unpack2col(ndim, nchannels, nnum, xvec, crlb):
    """
    unpacking the xvec from the MULPA fit
    INPUT:
        ndim:           int, number of dimensions
        nchannels:      int, number of imaging channels
        xvec:           (NFits, NMAX * (ndim + nchannels) + nchannels) float, the xvec from the MULPA fit
        crlb:           (NFits, NMAX * (ndim + nchannels) + nchannels) float, crlb corresponding to the xvec
        nnum:           (NFits) int ndarray, the number of fitted emiters for each fit
    OUTPUT:
        xvec_o:         (ndim + nchannels + nchannels, nnum.sum()) float ndarray, the rearranged xvec
        crlb_o:         (ndim + nchannels + nchannels, nnum.sum()) float ndarray, crlb corresponding to the xvec_o     
    NOTE:   almost 10 time faster than numpy
    """
    NFits = len(xvec)
    vmax = len(xvec[0])
    nsum = nnum.sum()
    vnum = ndim + nchannels + nchannels

    xvec = np.ascontiguousarray(xvec.flatten(), dtype=np.float32)
    nnum = np.ascontiguousarray(nnum, dtype=np.int32)
    xvec_o = np.ascontiguousarray(np.zeros(vnum * nsum), dtype=np.float32)
    crlb_o = np.ascontiguousarray(np.zeros(vnum * nsum), dtype=np.float32)
    
    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_xvec_unpack')
    fname = 'xvec_unpack.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    kernel = lib.xvec_unpack2col
    kernel.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                        ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),
                        ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                        ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                        ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                        ctl.ndpointer(np.float32, flags="aligned, c_contiguous")]
    kernel(NFits, vmax, ndim, nchannels, nsum, nnum, xvec, crlb, xvec_o, crlb_o)
    return xvec_o.reshape((vnum, nsum)), crlb_o.reshape((vnum, nsum)) 




def _mulpa_xvec_unpack2row(ndim, nchannels, nnum, xvec, crlb):
    """
    unpacking the xvec from the MULPA fit
    INPUT:
        ndim:           int, number of dimensions
        nchannels:      int, number of imaging channels
        xvec:           (NFits, NMAX * (ndim + nchannels) + nchannels) float, the xvec from the MULPA fit
        crlb:           (NFits, NMAX * (ndim + nchannels) + nchannels) float, crlb corresponding to the xvec
        nnum:           (NFits) int ndarray, the number of fitted emiters for each fit
    OUTPUT:
        xvec_o:         (nnum.sum(), ndim + nchannels + nchannels) float ndarray, the rearranged xvec
        crlb_o:         (nnum.sum(), ndim + nchannels + nchannels) float ndarray, crlb corresponding to the xvec_o     
    NOTE:   almost 10 time faster than numpy
    """
    NFits = len(xvec)
    vmax = len(xvec[0])
    nsum = nnum.sum()
    vnum = ndim + nchannels + nchannels

    xvec = np.ascontiguousarray(xvec.flatten(), dtype=np.float32)
    nnum = np.ascontiguousarray(nnum, dtype=np.int32)
    xvec_o = np.ascontiguousarray(np.zeros(nsum * vnum), dtype=np.float32)
    crlb_o = np.ascontiguousarray(np.zeros(nsum * vnum), dtype=np.float32)
    
    fpth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpth = os.path.join(fpth, 'src_xvec_unpack')
    fname = 'xvec_unpack.dll'
    lib = ctypes.CDLL(os.path.join(fpth, fname))
    kernel = lib.xvec_unpack2row
    kernel.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                        ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),
                        ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                        ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                        ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                        ctl.ndpointer(np.float32, flags="aligned, c_contiguous")]
    kernel(NFits, vmax, ndim, nchannels, nnum, xvec, crlb, xvec_o, crlb_o)
    return xvec_o.reshape((nsum, vnum)), crlb_o.reshape((nsum, vnum)) 



if __name__ == '__main__':
    NFits = 3
    boxsz = 11
    ndim = 3
    nchannels = 2
    nmax = 7
    
    vnum0 = ndim + nchannels
    vmax = nmax * vnum0 + nchannels
    nnum = np.array([5, 3, 7])

    xvec = np.zeros((NFits * vmax))
    for i in range(NFits):
        for n in range(nnum[i]):
            xvec[i * vmax + n * vnum0 + 2] = 100.0 * (i+1) + (n+1)
            xvec[i * vmax + n * vnum0 + ndim] = 10000.0 * (i+1) - 100.0 * (n+1)
            xvec[i * vmax + n * vnum0 + ndim + 1] = 0.2 * (i+1) + 0.01 * (n+1)
        
        for n in range(nchannels):
            xvec[i * vmax + nnum[i] * vnum0 + n] = 10.0 * (i+1) + (n+1)
    
    
    xvec[0 * vmax + 0 * vnum0 + 0] =  5.5 
    xvec[0 * vmax + 0 * vnum0 + 1] =  5.5
    xvec[0 * vmax + 1 * vnum0 + 0] = -0.1 
    xvec[0 * vmax + 1 * vnum0 + 1] =  7.2
    xvec[0 * vmax + 2 * vnum0 + 0] =  3.7 
    xvec[0 * vmax + 2 * vnum0 + 1] = 11.2
    xvec[0 * vmax + 3 * vnum0 + 0] =  5.4 
    xvec[0 * vmax + 3 * vnum0 + 1] =  9.2
    xvec[0 * vmax + 4 * vnum0 + 0] =  3.9 
    xvec[0 * vmax + 4 * vnum0 + 1] = -0.5
    
    xvec[1 * vmax + 0 * vnum0 + 0] =  5.4
    xvec[1 * vmax + 0 * vnum0 + 1] =  4.9
    xvec[1 * vmax + 1 * vnum0 + 0] = 11.3 
    xvec[1 * vmax + 1 * vnum0 + 1] =  7.5
    xvec[1 * vmax + 2 * vnum0 + 0] = -0.7 
    xvec[1 * vmax + 2 * vnum0 + 1] = 11.5
    
    xvec[2 * vmax + 0 * vnum0 + 0] = -1.0
    xvec[2 * vmax + 0 * vnum0 + 1] =  7.5
    xvec[2 * vmax + 1 * vnum0 + 0] =  5.4
    xvec[2 * vmax + 1 * vnum0 + 1] =  5.4
    xvec[2 * vmax + 2 * vnum0 + 0] =  3.5
    xvec[2 * vmax + 2 * vnum0 + 1] =  9.4
    xvec[2 * vmax + 3 * vnum0 + 0] = 11.3
    xvec[2 * vmax + 3 * vnum0 + 1] =  7.5
    xvec[2 * vmax + 4 * vnum0 + 0] = 12.0
    xvec[2 * vmax + 4 * vnum0 + 1] =  5.5
    xvec[2 * vmax + 5 * vnum0 + 0] =  7.1
    xvec[2 * vmax + 5 * vnum0 + 1] =  3.1
    xvec[2 * vmax + 6 * vnum0 + 0] =  6.7 
    xvec[2 * vmax + 6 * vnum0 + 1] = -0.5
    xvec = np.reshape(xvec, (NFits, vmax))
    crlb = xvec.copy()

    nnum, xvec, crlb = _mulpa_xvec_inbox(ndim, nchannels, boxsz, nnum, xvec, crlb)

    print(nnum)
    for i in range(NFits):
        print('\n')
        for n in range(nnum[i]):
            print(xvec[i, n * vnum0 : (n + 1) * vnum0])
        