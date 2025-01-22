import os
import time
import pickle
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import numpy as np
from PIL import Image
from CHNProcess.Polywarp_Utils import im_polywarpB2A



def main_chnlcomb(img_fpath, z_proj, view_type, coeff_Ts2R, upscalars=[4, 4]):
    """
    Main wrapper function for 
    1.  combination of the channels of the raw camera image according to the view_type
    2.  z-projection according to the z_proj for easier draw of rois afterwards 
    INPUT:
        img_fpath:      str, the absolute path saving the tiff images
        z_proj:         str, method for the z-projection
                        '0':                the first slice
                        '-1':               the last slice
                        'x':                the x-th slice 
                        'maximum':          the maximum projection along the z-axis
                        'mean':             the mean projection along the z-axis
        view_type:      'str', the channel_type of the experimental settings, provided by user in UI, determined by the microscope configuration.
                        'fullview':         full chip channel (1-channel)
                        'dualview':         left-right dual channels (2-channel) spliting the chip window
                        'quadralview':      田 quadral channels (4-channel) spliting the chip window
        coeff_Ts2R:     (nchannels-1, warpdeg*warpdeg) float ndarray, the warpdeg-degree polynomial coefficients warping from other channelsto the 0-th (reference) channel
        upscalars:      (2,) int ndarray, [upscalary, upscalarx] the upscaling factor for finer image warp, [1, 1] for no upscaling      
    """
    
    # result image file name
    fname = os.path.basename(img_fpath) + '_zProj.png'
    
    # read the first .tif image from the input img_fpath
    print("Processing {}".format(img_fpath), end='\r')
    tas = time.time()
    img_fname_list = [img_fname for img_fname in os.listdir(img_fpath) if img_fname.endswith('.tif')]
    if len(img_fname_list) == 0:
        raise ValueError("No tiff images detected")
    img_fullfname = os.path.join(img_fpath, img_fname_list[0])

    # Image information
    ImObj = Image.open(img_fullfname)
    nfrm = ImObj.n_frames
    chipszy = ImObj.height
    chipszx = ImObj.width

    # z projection
    if nfrm == 1:
        camim = np.asarray(ImObj, dtype=np.float32)
    elif nfrm > 1 and z_proj not in ['mean', 'maximum', '-1']:
        ImObj.seek(int(z_proj))
        camim = np.asarray(ImObj, dtype=np.float32)
    elif nfrm > 1 and z_proj == '-1':
        ImObj.seek(nfrm-1)
        camim = np.asarray(ImObj, dtype=np.float32)
    elif nfrm > 1 and z_proj == 'mean':
        camim = np.zeros((chipszy, chipszx), dtype=np.float32)
        for f in range(nfrm):
            ImObj.seek(f)
            dumim = np.asarray(ImObj, dtype=np.float32)
            camim += dumim / nfrm
    elif nfrm > 1 and z_proj == 'maximum':
        camim = np.zeros((chipszy, chipszx), dtype=np.uint16)
        for f in range(nfrm):
            ImObj.seek(f)
            dumim = np.asarray(ImObj, dtype=np.uint16)
            camim = np.maximum(camim, dumim)
        camim = np.float32(camim)
    else:
        raise ValueError("unsupported z_projection method: only a number, 0, -1, 'mean', 'maximum' are supported so far")

    # channel warp
    if view_type == 'fullview':
        combined_im = (camim - camim.min()) / camim.ptp() * 255.0
        ImObj = Image.fromarray(np.uint8(combined_im))
        ImObj.save(os.path.join(img_fpath, fname))
        print("{} elapsed for {}".format(time.time()-tas, img_fpath))
        return
    
    if view_type == 'dualview':
        nchannels = 2
        combined_im = np.zeros((3, chipszy, chipszx//2), dtype=np.float32)
        combined_im[0] = camim[:, :chipszx//2]
        combined_im[1] = camim[:, chipszx-chipszx//2:chipszx]
    elif view_type == 'quadralview': # the bottom-right corner is ignored in this case
        nchannels = 3
        combined_im = np.zeros((3, chipszy//2, chipszx//2), dtype=np.float32)
        combined_im[0] = camim[:chipszy//2, :chipszx//2]
        combined_im[1] = camim[:chipszy//2, chipszx-chipszx//2:chipszx]
        combined_im[2] = camim[chipszy-chipszy//2:chipszy, :chipszx//2]
    else:
        raise ValueError("unsupported view_type: only 'fullview', 'dualview', 'quadralview' are supported")
    
    for j in range(nchannels):
        combined_im[j] = (combined_im[j] - combined_im[j].min()) / combined_im[j].ptp() * 255.0
    for j in range(1, nchannels):
        combined_im[j] = im_polywarpB2A(combined_im[j], coeff_Ts2R[j-1], upscalars)
    ImObj = Image.fromarray(np.uint8(combined_im.transpose(1, 2, 0)))
    ImObj.save(os.path.join(img_fpath, fname))
    print("{} elapsed for {}".format(time.time()-tas, img_fpath))

    return



def _tkfopenfile(StrVar):
    fname = filedialog.askopenfilename()
    StrVar.set(fname)

def _tkfopendir(StrVar):
    fpath = filedialog.askdirectory()
    StrVar.set(fpath)

def _viewtypechk(e):    
    if view_type_vals.get() == 'fullview':
        chnlwarp_fname_vals.set('None')
        chnlwarp_fname_entry.config(state='disable')
        fopen_chnlwarp_button.config(state='disable')
    else:
        chnlwarp_fname_entry.config(state='readonly')
        fopen_chnlwarp_button.config(state='normal')

def _batchlvchk():
    if isbatch_vals.get():
        batchlv_option.config(state='readonly')
    else:
        batchlv_option.config(state='disabled')

def _tkexit(win):
    win.destroy()
   


if __name__ == '__main__':
    
    config_GUI = tk.Tk()
    config_GUI.title("Channel Combination")
    config_GUI.geometry("350x325")
    ft = 'Times New Roman'
    row_track = -1

    # Parameters
    row_track += 1
    label = ttk.Label(config_GUI, text="Configurations", font=(ft, 14))
    label.grid(column=0, columnspan=4, row=row_track, padx=10, pady=(20, 0), sticky='w')
    label.config(foreground='gray')

    # view_types
    row_track += 1
    ttk.Label(config_GUI, text="view_type:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    view_type_vals = tk.StringVar(config_GUI)
    view_type_vals.set('fullview')
    view_type_option = ttk.Combobox(config_GUI, width=21, textvariable=view_type_vals, state='readonly')
    view_type_option['values'] = ['fullview', 'dualview', 'quadralview']
    view_type_option.grid(column=1, columnspan=3, row=row_track, sticky='w')
    view_type_option.bind("<<ComboboxSelected>>", _viewtypechk)
    
    # z-project
    row_track += 1
    ttk.Label(config_GUI, text="z_project:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    z_project_vals = tk.StringVar(config_GUI)
    z_project_vals.set('maximum')
    z_project_option = ttk.Combobox(config_GUI, width=21, textvariable=z_project_vals, state='normal')
    z_project_option['values'] = ['0', 'maximum', 'mean', '-1']
    z_project_option.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # Channel_warp
    row_track += 1
    ttk.Label(config_GUI, text="Chnlwarp:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    chnlwarp_fname_vals = tk.StringVar(config_GUI)
    chnlwarp_fname_vals.set('None')
    chnlwarp_fname_entry = ttk.Entry(config_GUI, width=21, textvariable=chnlwarp_fname_vals, state='disable')
    chnlwarp_fname_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')
    fopen_chnlwarp_button = tk.Button(config_GUI, text='Browse', font=(ft, 8), height=1, state='disable', command=lambda : _tkfopenfile(chnlwarp_fname_vals))
    fopen_chnlwarp_button.grid(column=4, row=row_track, sticky='w')
    
    
    # Image Files
    row_track += 1
    label = ttk.Label(config_GUI, text="Image Files", font=(ft, 14))
    label.grid(column=0, columnspan=4, row=row_track, padx=10, pady=(20, 0), sticky='w')
    label.config(foreground='gray')

    # isbatch
    row_track += 1
    ttk.Label(config_GUI, text="batch:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    isbatch_vals = tk.BooleanVar(config_GUI)
    isbatch_vals.set(False)
    isbatch_option = ttk.Checkbutton(config_GUI, takefocus=0, variable=isbatch_vals, offvalue=False, onvalue=True, command=_batchlvchk)
    isbatch_option.grid(column=1, row=row_track, padx=0, pady=0, sticky='w')

    # batch level
    row_track += 1
    batchlv_vals = tk.StringVar(config_GUI)
    batchlv_vals.set('spools')
    ttk.Label(config_GUI, text="batch lv:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    batchlv_option = ttk.Combobox(config_GUI, width=21, textvariable=batchlv_vals, state='disabled')
    batchlv_option['values'] = ['spools', 'samples']
    batchlv_option.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # image file
    row_track += 1
    ttk.Label(config_GUI, text="img file:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    im_fpath_vals = tk.StringVar(config_GUI)
    im_fpath_entry = ttk.Entry(config_GUI, width=21, textvariable=im_fpath_vals)
    im_fpath_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')
    fopen_fname_button = tk.Button(config_GUI, text='Browse', font=(ft, 8), height=1, command=lambda : _tkfopendir(im_fpath_vals))
    fopen_fname_button.grid(column=4, row=row_track, sticky='w')


    # TERMINATE
    row_track += 1
    exit_button = tk.Button(config_GUI, text='Save and Run', font=(ft, 12), height=2, command=lambda : _tkexit(config_GUI))
    exit_button.grid(column=4, row=row_track, pady=(20, 0), sticky='w')
    
    config_GUI.mainloop()


    # collect the parameters
    fpath = im_fpath_vals.get()
    if not os.path.isdir(fpath):
        raise ValueError("The image path is not found")
    batchlv = batchlv_vals.get() if isbatch_vals.get() else None

    z_proj = z_project_vals.get()
    view_type = view_type_vals.get()
    if chnlwarp_fname_vals.get() == 'None':
       coeff_Ts2R = None 
    else:
        with open(chnlwarp_fname_vals.get(), 'rb') as fid:
            chnl_warp = pickle.load(fid)
        coeff_Ts2R = chnl_warp['Ts2R']

    if batchlv is None:
        if os.path.isdir(fpath):
            main_chnlcomb(fpath, z_proj, view_type, coeff_Ts2R)
    
    elif batchlv == 'spools':
        img_fpaths = [img_fpath for img_fpath in os.listdir(fpath) if os.path.isdir(os.path.join(fpath, img_fpath)) and img_fpath.startswith('spool')]
        for img_fpath in img_fpaths:
            main_chnlcomb(os.path.join(fpath, img_fpath), z_proj, view_type, coeff_Ts2R)
        
    elif batchlv == 'samples':
        sample_fpaths = [sample_fpath for sample_fpath in os.listdir(fpath) if os.path.isdir(os.path.join(fpath, sample_fpath))]
        for sample_fpath in sample_fpaths:
            sample_fpath = os.path.join(fpath, sample_fpath)
            img_fpaths = [img_fpath for img_fpath in os.listdir(sample_fpath) if os.path.isdir(os.path.join(sample_fpath, img_fpath)) and img_fpath.startswith('spool')]
            for img_fpath in img_fpaths:
                main_chnlcomb(os.path.join(sample_fpath, img_fpath), z_proj, view_type, coeff_Ts2R)