from win32api import GetSystemMetrics
import numpy as np
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
from matplotlib import gridspec


scrWidth, scrHeight = GetSystemMetrics(0), GetSystemMetrics(1)



def _set_canvas(imszy, imszx, view_type):
    """
    Set display canvas either in figures raw sizes or adjusted to the monitor
    INPUT:
        imszy:          int, the height of the 2d-ndarray image to display
        imszx:          int, the width of the 2d-ndarray image to display
        view_type:      'fullview':         full chip channel (1-channel)
                        'dualview':         left-right dual channels (2-channel)
                        'quadralview':      田 quadral channels (4-channel) 
    RETURN:
        fig, ax:    pyplot figure handling
    """
    plt.rcParams['toolbar'] = 'None'
    px = 1/72 # use 72 dpi for display on monitor. Note that the default rcParams['figure.dpi']=100

    if view_type == 'fullview':        
        _scalar = min(scrWidth/imszx, scrHeight/imszy)
        fig, ax = plt.subplots(figsize = (0.85*imszx*_scalar*px, 0.85*imszy*_scalar*px))
        axs = [ax]
    elif view_type == 'dualview':
        _scalar = min(scrWidth/imszx/2, scrHeight/imszy)
        fig = plt.figure(figsize = (0.85*imszx*2*_scalar*px, 0.85*imszy*_scalar*px)) 
        grid = gridspec.GridSpec(1, 2, figure=fig)
        axs = [fig.add_subplot(grid[i]) for i in range(2)]
    elif view_type == 'quadralview':
        _scalar = min(scrWidth/imszx/2, scrHeight/imszy/2)
        fig = plt.figure(figsize = (0.85*imszx*2*_scalar*px, 0.85*imszy*2*_scalar*px)) 
        grid = gridspec.GridSpec(2, 2, figure=fig)
        axs = [fig.add_subplot(grid[i//2, i%2]) for i in range(4)]
    else:
        raise ValueError("viewtype mismatch: view_type={}, supported ('single_chnl', 'dual_chnl', 'quad_chnl')".format(view_type)) 
       
    for ax in axs:
        ax.set_xlim(0, imszx)
        ax.set_ylim(imszy, 0)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])    
    plt.tight_layout()
    
    return fig, axs



def _get_massc(im):
    """get the masscenter of the input 2d image"""
    im = np.array(im, dtype=np.float64)
    im_sum = im.sum()
    
    imszy, imszx = im.shape
    YY, XX = np.meshgrid(np.arange(imszy)+0.5, np.arange(imszx)+0.5, indexing='ij')
    xc = np.sum(XX * im) / im_sum
    yc = np.sum(YY * im) / im_sum
    return xc, yc




def im_peakpicker2d(ims, boxsz, npicks):
    """
    Return the coordinates user-clicked localmax on the image
    INPUT:
        ims:            (imszy, imszx) or (nchannels, imszy, imszx) ndarray, image to click on local maximas
        boxsz:          int, boxsz for the of the lclMax kernel, must be odd
        npicks:         int, number of peaks to click
    RETURN:
        indx:           (nchannels * npicks) int ndarray, indices (x-axis) of the local maxima
        indy:           (nchannels * npicks) int ndarray, indices (y-axis) of the local maxima      
    """
    
    ########## my click function ##########
    def _click_frm(event, ims, boxsz, pts, npicks, fig, axs):
        
        # NOTE: NO INPUT CHECKS

        boxhsz = boxsz//2
        nchannels, imszy, imszx = ims.shape
        disp_color = ['tab:red', 'tab:green', 'tab:blue', 'tab:purple']
        
        gch = len(pts) % nchannels
        gax = axs[gch]
        if gax != event.inaxes:
            return
        
        xc = np.int32(event.xdata)
        yc = np.int32(event.ydata)
        foo = ims[gch, yc-boxhsz:yc+boxhsz+1, xc-boxhsz:xc+boxhsz+1]
        
        x_massc, y_massc = _get_massc(foo)
        ind_xc = np.int32(x_massc + 0.5)
        ind_yc = np.int32(y_massc + 0.5)
        xc += ind_xc - boxhsz
        yc += ind_yc - boxhsz 
        if yc < boxhsz or yc >= imszy-boxhsz or xc < boxhsz or xc >= imszx-boxhsz:
            print("invalid pick, too close to the edge")
            return
        
        ybnd = [yc-boxhsz, yc-boxhsz, yc+boxhsz+1, yc+boxhsz+1, yc-boxhsz]
        xbnd = [xc-boxhsz, xc+boxhsz+1, xc+boxhsz+1, xc-boxhsz, xc-boxhsz]
        gax.plot(xbnd, ybnd, color=disp_color[gch])
        gax.figure.canvas.draw()
        if nchannels > 1:
            print("pick bead #{n:d} in channel #{chn:d} at [y={y:d}, x={x:d}]".format(n=len(pts)//nchannels, chn=gch, y=yc, x=xc))
        else:
            print("pick bead #{n:d} at [y={y:d}, x={x:d}]".format(n=len(pts)//nchannels, y=yc, x=xc))
        pts.append([xc, yc])
        
        if len(pts) >= nchannels * npicks:
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)
        
        return
    #######################################
    
    # parse the inputs
    if boxsz % 2 == 0: 
        raise ValueError("not odd: boxsz={}, should be odd".format(boxsz))
    if ims.ndim == 2:
        ims = ims[np.newaxis,...]
    if ims.ndim != 3:
        raise ValueError("ndim mismatch: ims.shape={}, should be (nchannels, imszy, imszx)".format(ims.shape))
    nchannels, imszy, imszx = ims.shape
    if nchannels not in (1, 2, 4):
        raise ValueError("nchannels not supported: nchannels={}, should be (1, 2, or 4)".format(nchannels))
    if nchannels == 1:
        view_type = 'fullview'
    elif nchannels == 2:
        view_type = 'dualview'
    else:
        view_type = 'quadralview'

    # display and pick on the image
    fig, axs = _set_canvas(imszy, imszx, view_type)
    for im, ax in zip(ims, axs):
        ax.imshow(im, cmap='gray')
    
    pts = []
    cid = fig.canvas.mpl_connect('button_press_event', lambda event: _click_frm(event, ims, boxsz, pts, npicks, fig, axs))
    plt.show()
    plt.close(fig)
    
    return np.asarray(pts, dtype=np.int32).T



def im_peakpicker3d(ims, boxsz, npicks, zstepsz, zrange, zsmooth=1.0):
    """
    Return the coordinates user-clicked localmax on the image
    INPUT:
        ims:            (imszz, imszy, imszx) or (nchannels, imszz, imszy, imszx) ndarray, image to click on local maximas
        boxsz:          int, boxsz for the of the lclMax kernel, must be odd
        npicks:         int, number of peaks to click
        zstepsz:        float, nm of the z step size
        zrange:         int, zrange size for the bead of the lclMax kernel, must be odd
        zsmooth:        float, smoothing factor to locate the axial maximum via spline interplotation
    RETURN:
        indx:           (npicks,) int ndarray, indices (x-axis) of the local maxima
        indy:           (npicks,) int ndarray, indices (y-axis) of the local maxima
        indz:           (npicks,) int ndarray, indices (z-axis) of the local maxima 
    """

    ########## my click function ##########
    def _click_stck(event, ims, boxsz, npicks, zstepsz, zrange, zsmooth, pts, fig, axs):

        # NOTE: NO INPUT CHECKS

        boxhsz = boxsz//2
        nchannels, imszz, imszy, imszx = ims.shape
        disp_color = ['tab:red', 'tab:green', 'tab:blue', 'tab:purple']
        
        gch = len(pts) % nchannels
        gax = axs[gch]
        if gax != event.inaxes:
            return

        # localize the y- and x-center of the local maximum
        xc = np.int_(event.xdata)
        yc = np.int_(event.ydata)
        foo = np.mean(ims[gch, :, yc-boxhsz:yc+boxhsz+1, xc-boxhsz:xc+boxhsz+1].astype(np.float32), axis=0)
        x_massc, y_massc = _get_massc(foo)
        ind_xc = np.int32(x_massc + 0.5)
        ind_yc = np.int32(y_massc + 0.5)
        xc += ind_xc - boxhsz
        yc += ind_yc - boxhsz  
        if yc <= boxhsz or yc >= imszy-boxhsz or xc <= boxhsz or xc >= imszx-boxhsz:
            print("invalid pick, too close to the edge")
            return
        
        # localize the z-center of the local maximum
        dum_stack = ims[gch, :, yc-boxhsz:yc+boxhsz+1, xc-boxhsz:xc+boxhsz+1]
        dum_intensity = np.float32(np.max(np.max(dum_stack, axis=-1), axis=-1))
        dum_spl = splrep(np.arange(imszz), dum_intensity, k=3, s=zsmooth/zstepsz*100.0)
        zc = np.argmax(splev(np.arange(imszz), dum_spl))
        if zc < zrange//2 or zc >= imszz-zrange//2:
            print("invalid pick, zcenter not within the zrange")
            return

        ybnd = [yc-boxhsz, yc-boxhsz, yc+boxhsz+1, yc+boxhsz+1, yc-boxhsz]
        xbnd = [xc-boxhsz, xc+boxhsz+1, xc+boxhsz+1, xc-boxhsz, xc-boxhsz]
        gax.plot(xbnd, ybnd, color=disp_color[gch])
        gax.figure.canvas.draw()
        if nchannels > 1:
            print("pick bead #{n:d} in channel #{chn:d} at [z={z:d}, y={y:d}, x={x:d}]".format(n=len(pts)//nchannels, chn=gch, z=zc, y=yc, x=xc))
        else:
            print("pick bead #{n:d} at [z={z:d}, y={y:d}, x={x:d}]".format(n=len(pts)//nchannels, z=zc, y=yc, x=xc))
        pts.append([xc, yc, zc])
        
        # terminate the click
        if len(pts) >= npicks:
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)
        
        return
    #######################################

    # parse the inputs
    if boxsz % 2 == 0: 
        raise ValueError("not odd: boxsz={}, should be odd".format(boxsz))
    if ims.ndim == 3:
        ims = ims[np.newaxis,...]
    if ims.ndim != 4:
        raise ValueError("ndim mismatch: ims.shape={}, should be (nchannels, imszz, imszy, imszx)".format(ims.shape))
    nchannels, imszz, imszy, imszx = ims.shape
    if nchannels not in (1, 2, 4):
        raise ValueError("nchannels not supported: nchannels={}, should be (1, 2, or 4)".format(nchannels))
    if nchannels == 1:
        view_type = 'fullview'
    elif nchannels == 2:
        view_type = 'dualview'
    else:
        view_type = 'quadralview'

    # display and pick on the image
    fig, axs = _set_canvas(imszy, imszx, view_type)
    for im, ax in zip(ims, axs):
        ax.imshow(np.mean(im, axis=0), cmap='gray')

    pts = []
    cid = fig.canvas.mpl_connect('button_press_event', lambda event: _click_stck(event, ims, boxsz, npicks, zstepsz, zrange, zsmooth, pts, fig, axs))
    plt.show()
    plt.close(fig)

    return np.asarray(pts, dtype=np.int32).T