import os
import struct
import shutil
from PIL import Image
from zipfile import ZipFile
import numpy as np



def _readROI(roi_fname):
    
    with open(roi_fname, 'rb') as fid:
        roi_info = fid.read()
    
    # the first 4 byte should be a string "Iout"
    strMagic = "".join([chr(b) for b in roi_info[:4]])
    if strMagic != 'Iout':
        raise TypeError("Invalid ImageJ roi")
    
    type_code = roi_info[6]                      # roi_type: the 6th byte
    if not type_code in (1, 3):
        raise TypeError("Only rectangle and line type of roi is supported")
    roi_type = 'rect' if type_code == 1 else 'line'

    if type_code == 1:
        top     = (roi_info[8]<<8) + roi_info[9]      # top: the 8-9th bytes
        left    = (roi_info[10]<<8) + roi_info[11]    # left: the 10-11th bytes
        bottom  = (roi_info[12]<<8) + roi_info[13]    # bottom: the 12-13th bytes
        right   = (roi_info[14]<<8) + roi_info[15]    # right: the 14-15th bytes
        roi = [top, left, bottom, right]
    else:
        x1 = struct.unpack('>f', bytes([roi_info[i] for i in range(18, 22)]))[0]
        y1 = struct.unpack('>f', bytes([roi_info[i] for i in range(22, 26)]))[0]
        x2 = struct.unpack('>f', bytes([roi_info[i] for i in range(26, 30)]))[0]
        y2 = struct.unpack('>f', bytes([roi_info[i] for i in range(30, 34)]))[0]
        roi = [[x1, y1], [x2, y2]] if x1 < x2 else [[x2, y2], [x1, y1]]
        
    return roi_type, roi



def _readroifile(roi_fname):
    # dedicated to read rectangle ImageJ roi from the filename
    with open(roi_fname, 'rb') as fid:
        roi = fid.read()
        fid.close()
    
    # the first 4 byte should be a string "Iout"
    strMagic = "".join([chr(b) for b in roi[:4]])
    if strMagic != 'Iout':
        raise TypeError("Invalid ImageJ roi")
    
    roi_type = roi[6]                   # roi_type: the 6th byte
    if roi_type != 1:
        raise TypeError("Only rectangle roi is supported")

    top = (roi[8]<<8) + roi[9]          # top: the 8-9th bytes
    left = (roi[10]<<8) + roi[11]       # left: the 10-11th bytes
    bottom = (roi[12]<<8) + roi[13]     # bottom: the 12-13th bytes
    right = (roi[14]<<8) + roi[15]      # right: the 14-15th bytes    

    return top, left, bottom, right



def _readroizip(roi_zipname, mode='None'):
    # collect roi from roi_zipname
    fpth = os.path.dirname(os.path.abspath(roi_zipname))
    
    tmpfpth = os.path.join(fpth, '_tmp')
    if os.path.exists(tmpfpth):
        shutil.rmtree(tmpfpth)
    os.makedirs(tmpfpth)

    with ZipFile(os.path.abspath(roi_zipname), 'r') as zipObj:
        zipObj.extractall(path=tmpfpth)
    
    rois = []
    for roi_fname in zipObj.namelist():
        roi = _readROI(os.path.join(tmpfpth, roi_fname))
        if mode == 'None':
            rois.append(roi)
        elif roi[0] == mode:
            rois.append(roi[1])

    shutil.rmtree(tmpfpth)
    return rois



def getROIs(roi_fname, mode='None'):
    """
    read the rois from the given roi_fname
    INPUT:  roi_fname:      (string) the absolute file name of the .roi or .zip 
            mode:           only rois with type==mode are returned. All roi returned if mode=='None'
    RETURN: rois:           list of the rois
    """
    
    if roi_fname.endswith('.zip'):
        rois = _readroizip(roi_fname, mode)
        return rois
    
    if roi_fname.endswith('.roi'):
        roi = _readROI(roi_fname)
        if mode == 'None':
            return [roi]
        elif roi[0] == mode:
            return [roi[1]]    
    
    return None



def getchipROI(Camsetting, camim_fname):
    """
    read the rois from the given roi_fname
    INPUT:  Camsetting:     dictionary, see Documentation.md for detals
            camim_fname:    str, the absolute path of the camera image
    RETURN: rois:           list of rectangle rois
    """
    
    view_type = Camsetting['view_type']

    ImObj = Image.open(camim_fname)
    chipszy = ImObj.height
    chipszx = ImObj.width

    if view_type == 'fullview':
        rois = np.array([[0, 0, chipszy, chipszx]])

    else:
        assert chipszy == Camsetting['chipszh'], "chipszh mismatch: chipszh(camsetting)={}, chipszh(camim)={}".format(Camsetting['chipszh'], chipszy)
        assert chipszx == Camsetting['chipszw'], "chipszw mismatch: chipszw(camsetting)={}, chipszw(camim)={}".format(Camsetting['chipszw'], chipszx)
        if view_type == 'dualview':
            rois = np.array([[0, 0, chipszy, chipszx//2]])
        elif view_type == 'quadralview':
            rois = np.array([[0, 0, chipszy//2, chipszx//2]])
    
    return rois




def im_loader(Camsetting, camim_fname, zPos=None, rois=None):
    """
    Load and rearange the images according to the camera setting and view_type
    INPUT:  
        Camsetting:     dictionary, see Documentation.md for detals
        camim_fname:    str, the absolute path of the camera image
        zPos:           int or None, the zPos-th frame of the image will be loaded, None for all the frames, ignored if the scource image is single-frame
        rois:           (nrois, 4) ndarray, [top, left, bottom, right] bounds for each roi, None for full range
                        roi should be drawn accroding to the view_type
    RETURN: 
        ims:            (nrois,) dictionary, keys are:
                        'roi':          (4,) int ndarray, [top, left, bottom, right] bounds for each roi
                        'nfrm':         int, number of frames of the input image
                        'cam_offset':   (nchannels, roiszy, roiszx) or (nchannels,) float ndarray, cam_offset from each channel within the roi
                        'cam_var':      (nchannels, roiszy, roiszx) or (nchannels,) float ndarray, cam_var from each channel within the roi, None if Camsetting['scmos_var'] == False
                        'cam_a2d':      (nchannels, roiszy, roiszx) or (nchannels,) float ndarray, cam_A2D from each channel within the roi
                        'cam_gain':     float, EMgain of the camera if available
                        'imstack':      (nchannels, roiszy, roiszx) or (nchannels, nfrm, roiszy, roiszx) uint16 ndarray, image stack from each channel within the roi
    """

    # read the Camsettings
    view_type = Camsetting['view_type']
    scmos_var = Camsetting['scmos_var']
    fccam_offset = Camsetting['offset']
    fccam_var = Camsetting['var']
    fccam_a2d = Camsetting['A2D']
    fccam_gain = Camsetting['EMGain']

    # read the information of the images
    ImObj = Image.open(camim_fname)
    nfrm = ImObj.n_frames
    chipszy = ImObj.height
    chipszx = ImObj.width
    if view_type == 'dualview':
        assert chipszy == Camsetting['chipszh'], "chipszh mismatch: chipszh(camsetting)={}, chipszh(camim)={}".format(Camsetting['chipszh'], chipszy)
        assert chipszx == Camsetting['chipszw'], "chipszw mismatch: chipszw(camsetting)={}, chipszw(camim)={}".format(Camsetting['chipszw'], chipszx)

    # read the rois
    if rois is None:
        if view_type == 'fullview':
            rois = np.array([[0, 0, chipszy, chipszx]])
        elif view_type == 'dualview':
            rois = np.array([[0, 0, chipszy, chipszx//2]])
        elif view_type == 'quadralview':
            rois = np.array([[0, 0, chipszy//2, chipszx//2]])
    else:
        rois = np.asarray(rois)
        if rois.ndim == 1:
            rois = rois[np.newaxis, :]
    nrois = len(rois)

    # read the images according to the view_type and rois
    ims = [dict() for _ in range(nrois)]
    for i, roi in enumerate(rois):
        
        # roi in nchannels
        if view_type == 'fullview':
            nchannels = 1
            slc_0 = [(slice(roi[0], roi[2]), slice(roi[1], roi[3]))]
        elif view_type == 'dualview':
            nchannels = 2
            slc_0 = [(slice(roi[0], roi[2]), slice(j*chipszx//2+roi[1], j*chipszx//2+roi[3])) for j in range(2)]
        else:
            nchannels = 4
            slc_0 = [(slice(j//2*chipszy//2+roi[0], j//2*chipszy//2+roi[2]), slice(j%2*chipszx//2+roi[1], j%2*chipszx//2+roi[3])) for j in range(4)]
        
        # offset
        if isinstance(fccam_offset, (float, np.float32, np.float64)):
            cam_offset = [fccam_offset] * nchannels
        else:
            cam_offset = np.array([fccam_offset[slc_0[j]] for j in range(nchannels)], dtype=np.float32)

        # a2d
        if isinstance(fccam_a2d, (float, np.float32, np.float64)):
            cam_a2d = [fccam_a2d] * nchannels
        else:
            cam_a2d = np.array([fccam_a2d[slc_0[j]] for j in range(nchannels)], dtype=np.float32)
        
        # variance
        if scmos_var:
            if isinstance(fccam_var, (float, np.float32, np.float64)):
                cam_var = None
                Camsetting['scmos_var'] = False
            else:
                cam_var = np.array([fccam_var[slc_0[j]] for j in range(nchannels)], dtype=np.float32)
        else:
            cam_var = None
        
        # imstack 
        if nfrm > 1 and zPos is None: 
            cam_imstack = np.zeros((nchannels, nfrm, roi[2]-roi[0], roi[3]-roi[1]), dtype=np.uint16)
            for f in range(nfrm):
                ImObj.seek(f)
                dum_camim = np.asarray(ImObj, dtype=np.uint16)
                for j in range(nchannels):
                    cam_imstack[j][f] = dum_camim[slc_0[j]]        
        elif nfrm == 1:
            dum_camim = np.asarray(ImObj, dtype=np.uint16)
            cam_imstack = np.array([dum_camim[slc_0[j]] for j in range(nchannels)], dtype=np.uint16)
        else:
            cam_imstack = np.zeros((nchannels, roi[2]-roi[0], roi[3]-roi[1]), dtype=np.uint16)
            for j in range(nchannels):
                ImObj.seek(zPos[j])
                dum_camim = np.asarray(ImObj, dtype=np.uint16)
                cam_imstack[j] = dum_camim[slc_0[j]]
            
        # collect the information
        ims[i]['roi']           = roi
        ims[i]['nfrm']          = nfrm
        ims[i]['cam_offset']    = cam_offset
        ims[i]['cam_a2d']       = cam_a2d
        ims[i]['cam_var']       = cam_var
        ims[i]['cam_gain']      = fccam_gain
        ims[i]['imstack']       = cam_imstack  

    return ims    