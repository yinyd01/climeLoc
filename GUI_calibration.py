import os
import pickle
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import numpy as np

from IMGProcess.ImageLoader import im_loader
from main_psfcalibration import PSFcali_2dfrm_unlinked, PSFcali_3dstck_unlinked
from main_chnlwarp import chnlwarp


def _tkfopen(StrVar):
    fname = filedialog.askopenfilename()
    StrVar.set(fname)

def _modalitychk(e, modality_option):
    if modality_option.get() != '2Dfrm':
        zfocus_ch0_entry.config(state='normal')
        zfocus_ch1_entry.config(state='normal')
        zstepsz_entry.config(state='normal')
        cal_zrange_entry.config(state='normal')
    else:
        zstepsz_entry.config(state='disable')
        zfocus_ch0_entry.config(state='disable')
        zfocus_ch1_entry.config(state='disable')
        cal_zrange_entry.config(state='disable')
        
def _camtypechk():
    if isfile_vals.get():
        cam_fname_entry.config(state='readonly')
        chipszx_entry.config(state='disable')
        chipszy_entry.config(state='disable')
        cam_offset_entry.config(state='disable')
        cam_var_entry.config(state='disable')
        cam_A4D_entry.config(state='disable')
        cam_emgain_entry.config(state='disable')
        scmos_var_option.config(state='normal')
    else:
        cam_fname_entry.config(state='disable')
        chipszx_entry.config(state='normal')
        chipszy_entry.config(state='normal')
        cam_offset_entry.config(state='normal')
        cam_var_entry.config(state='normal')
        cam_A4D_entry.config(state='normal')
        cam_emgain_entry.config(state='normal')
        scmos_var_option.config(state='disable')

def _chnlchk(e, view_type_option):
    if view_type_option.get() == 'dualview':
        zfocus_ch1_entry.config(state='normal')
    else:
        zfocus_ch1_entry.config(state='disable')

def _tkclose(win, tab_active):
    tab_active[0] = True
    win.destroy()
   

if __name__ == '__main__':
    

    ## ################################ GUI WINDOW ################################ ##
    config_GUI = tk.Tk()
    config_GUI.title("CALIBRATION")
    config_GUI.geometry("375x700")
    ft = 'Times New Roman'

    tab_control = ttk.Notebook(config_GUI)

    psf_config = ttk.Frame(tab_control)
    tab_control.add(psf_config, text='PSF Calibration')
    psftab_active = [False]
    
    chnl_config = ttk.Frame(tab_control)
    tab_control.add(chnl_config, text='CHNL Calibration')
    tab_control.pack(expand=1, fill='both')
    chnltab_active = [False]



    ## ################################ PSF TAB ################################ ##
    row_track = -1
    ## Beads info
    row_track += 1
    label = ttk.Label(psf_config, text="Beads Info", font=(ft, 14))
    label.grid(column=0, columnspan=4, row=row_track, padx=10, pady=(20, 0), sticky='w')
    label.config(foreground='gray')

    row_track += 1
    ttk.Label(psf_config, text="modality:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    modality_vals = tk.StringVar(psf_config)
    modality_vals.set('AS')
    modality_option = ttk.Combobox(psf_config, width=21, textvariable=modality_vals, state='readonly')
    modality_option['values'] = ['2Dfrm', '2D', 'AS', 'BP', 'DH']
    modality_option.grid(column=1, columnspan=3, row=row_track, sticky='w')
    modality_option.bind("<<ComboboxSelected>>", lambda e : _modalitychk(e, modality_option))

    row_track += 1
    ttk.Label(psf_config, text="bead fname:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    psfbead_fname_vals = tk.StringVar(psf_config)
    psfbead_fname_entry = ttk.Entry(psf_config, width=21, textvariable=psfbead_fname_vals, state='readonly')
    psfbead_fname_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')
    fopen_button = tk.Button(psf_config, text='Browse', font=(ft, 8), height=1, command=lambda : _tkfopen(psfbead_fname_vals))
    fopen_button.grid(column=4, row=row_track, sticky='w')

    ## Camera
    row_track += 1
    label = ttk.Label(psf_config, text="Camera Info", font=(ft, 14))
    label.grid(column=0, columnspan=4, row=row_track, padx=10, pady=(20, 0), sticky='w')
    label.config(foreground='gray')

    row_track += 1
    ttk.Label(psf_config, text="from file:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=(10, 0), sticky='e')
    isfile_vals = tk.BooleanVar(psf_config)
    isfile_vals.set(True)
    isfile_option = ttk.Checkbutton(psf_config, takefocus=0, variable=isfile_vals, offvalue=False, onvalue=True, command=_camtypechk)
    isfile_option.grid(column=1, row=row_track, padx=0, pady=(10, 0), sticky='w')

    row_track += 1
    ttk.Label(psf_config, text="cam_spec:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    cam_fname_vals = tk.StringVar(psf_config)
    cam_fname_entry = ttk.Entry(psf_config, width=21, textvariable=cam_fname_vals, state='readonly')
    cam_fname_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')
    fopen_button = tk.Button(psf_config, text='Browse', font=(ft, 8), height=1, command=lambda : _tkfopen(cam_fname_vals))
    fopen_button.grid(column=4, row=row_track, sticky='w')

    row_track += 1
    ttk.Label(psf_config, text="chip size:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    chipszx_vals = tk.IntVar(psf_config)
    chipszy_vals = tk.IntVar(psf_config)
    chipszx_vals.set(1200)
    chipszy_vals.set(1200)
    chipszx_entry = ttk.Entry(psf_config, width=7, textvariable=chipszx_vals, state='disable')
    chipszx_entry.grid(column=1, row=row_track, sticky='w')
    chipszy_entry = ttk.Entry(psf_config, width=7, textvariable=chipszy_vals, state='disable')
    chipszy_entry.grid(column=2, row=row_track, sticky='w')

    row_track += 1
    ttk.Label(psf_config, text="cam_offset:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    cam_offset_vals = tk.StringVar(psf_config)
    cam_offset_vals.set('100.0')
    cam_offset_entry = ttk.Entry(psf_config, width=21, textvariable=cam_offset_vals, state='disable')
    cam_offset_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    row_track += 1
    ttk.Label(psf_config, text="cam_var:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    cam_var_vals = tk.StringVar(psf_config)
    cam_var_vals.set('50.0')
    cam_var_entry = ttk.Entry(psf_config, width=21, textvariable=cam_var_vals, state='disable')
    cam_var_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    row_track += 1
    ttk.Label(psf_config, text="cam_A4D:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    cam_A4D_vals = tk.StringVar(psf_config)
    cam_A4D_vals.set('45.0')
    cam_A4D_entry = ttk.Entry(psf_config, width=21, textvariable=cam_A4D_vals, state='disable')
    cam_A4D_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    row_track += 1
    ttk.Label(psf_config, text="cam_emgain:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    cam_emgain_vals = tk.StringVar(psf_config)
    cam_emgain_vals.set('300.0')
    cam_emgain_entry = ttk.Entry(psf_config, width=21, textvariable=cam_emgain_vals, state='disable')
    cam_emgain_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    ## Camera Plus: additional camera information
    row_track += 1
    label = ttk.Label(psf_config, text="Camera Plus", font=(ft, 14))
    label.grid(column=0, columnspan=4, row=row_track, padx=10, pady=(20, 0), sticky='w')
    label.config(foreground='gray')

    row_track += 1
    ttk.Label(psf_config, text="view_type:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    view_type_vals = tk.StringVar(psf_config)
    view_type_vals.set('fullview')
    view_type_option = ttk.Combobox(psf_config, width=21, textvariable=view_type_vals, state='readonly')
    view_type_option['values'] = ['fullview', 'dualview']
    view_type_option.grid(column=1, columnspan=3, row=row_track, sticky='w')
    view_type_option.bind("<<ComboboxSelected>>", lambda e : _chnlchk(e, view_type_option))

    row_track += 1
    ttk.Label(psf_config, text="scmos_var:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=(10, 0), sticky='e')
    scmos_var_vals = tk.BooleanVar(psf_config)
    scmos_var_vals.set(False)
    scmos_var_option = ttk.Checkbutton(psf_config, takefocus=0, variable=scmos_var_vals, offvalue=False, onvalue=True)
    scmos_var_option.grid(column=1, row=row_track, padx=0, pady=(10, 0), sticky='w')

    row_track += 1
    ttk.Label(psf_config, text="px size (nm):", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    cam_pxszx_vals = tk.StringVar(psf_config)
    cam_pxszy_vals = tk.StringVar(psf_config)
    cam_pxszx_vals.set('98.41')
    cam_pxszy_vals.set('98.58')
    cam_pxszx_entry = ttk.Entry(psf_config, width=7, textvariable=cam_pxszx_vals)
    cam_pxszx_entry.grid(column=1, row=row_track, sticky='w')
    cam_pxszy_entry = ttk.Entry(psf_config, width=7, textvariable=cam_pxszy_vals)
    cam_pxszy_entry.grid(column=2, row=row_track, sticky='w')

    ## Axial info
    row_track += 1
    label = ttk.Label(psf_config, text="Axial Info", font=(ft, 14))
    label.grid(column=0, columnspan=4, row=row_track, padx=10, pady=(20, 0), sticky='w')
    label.config(foreground='gray')

    row_track += 1
    ttk.Label(psf_config, text="zstepsz (nm):", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    zstepsz_vals = tk.StringVar(psf_config)
    zstepsz_vals.set('10.0')
    zstepsz_entry = ttk.Entry(psf_config, width=21, textvariable=zstepsz_vals, state='normal')
    zstepsz_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    row_track += 1
    ttk.Label(psf_config, text="zfocus (slice):", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    zfocus_ch0_vals = tk.IntVar(psf_config)
    zfocus_ch1_vals = tk.IntVar(psf_config)
    zfocus_ch0_vals.set(-1)
    zfocus_ch1_vals.set(-1)
    zfocus_ch0_entry = ttk.Entry(psf_config, width=7, textvariable=zfocus_ch0_vals, state='normal')
    zfocus_ch0_entry.grid(column=1, row=row_track, sticky='w')
    zfocus_ch1_entry = ttk.Entry(psf_config, width=7, textvariable=zfocus_ch1_vals, state='normal')
    zfocus_ch1_entry.grid(column=2, row=row_track, sticky='w')
    
    row_track += 1
    ttk.Label(psf_config, text="zrange (nm):", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    cal_zrange_vals = tk.StringVar(psf_config)
    cal_zrange_vals.set('1200.0')
    cal_zrange_entry = ttk.Entry(psf_config, width=21, textvariable=cal_zrange_vals, state='normal')
    cal_zrange_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    ## Working parameters
    row_track += 1
    label = ttk.Label(psf_config, text="Working parameters", font=(ft, 14))
    label.grid(column=0, columnspan=4, row=row_track, padx=10, pady=(20, 0), sticky='w')
    label.config(foreground='gray')

    row_track += 1
    ttk.Label(psf_config, text="box size:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    boxsz_vals = tk.IntVar(psf_config)
    boxsz_vals.set(9)
    boxsz_entry = ttk.Entry(psf_config, width=21, textvariable=boxsz_vals)
    boxsz_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    row_track += 1
    ttk.Label(psf_config, text="beads number:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    psfnbeads_vals = tk.IntVar(psf_config)
    psfnbeads_vals.set(10)
    psfnbeads_entry = ttk.Entry(psf_config, width=21, textvariable=psfnbeads_vals)
    psfnbeads_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    ## terminate
    row_track += 1
    psftab_active = [False]
    exit_button = tk.Button(psf_config, text='Save and Run', font=(ft, 12), height=2, command=lambda : _tkclose(config_GUI, psftab_active))
    exit_button.grid(column=4, row=row_track, pady=(20, 0), sticky='w')
    


    ## ################################ CHNL TAB ################################ ##
    row_track = -1
    ## Beads info
    row_track += 1
    label = ttk.Label(chnl_config, text="Beads Info", font=(ft, 14))
    label.grid(column=0, columnspan=4, row=row_track, padx=10, pady=(20, 0), sticky='w')
    label.config(foreground='gray')

    row_track += 1
    ttk.Label(chnl_config, text="bead fname:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    chnlbead_fname_vals = tk.StringVar(chnl_config)
    chnlbead_fname_entry = ttk.Entry(chnl_config, width=21, textvariable=chnlbead_fname_vals)
    chnlbead_fname_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')
    fopen_button = tk.Button(chnl_config, text='Browse', font=(ft, 8), height=1, command=lambda : _tkfopen(chnlbead_fname_vals))
    fopen_button.grid(column=4, row=row_track, sticky='w')

    ## Working parameters
    row_track += 1
    label = ttk.Label(chnl_config, text="Working parameters", font=(ft, 14))
    label.grid(column=0, columnspan=4, row=row_track, padx=10, pady=(20, 0), sticky='w')
    label.config(foreground='gray')

    row_track += 1
    ttk.Label(chnl_config, text="number of pairs:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    chnlnbeads_vals = tk.IntVar(chnl_config)
    chnlnbeads_vals.set(9)
    nbeads_entry = ttk.Entry(chnl_config, width=21, textvariable=chnlnbeads_vals)
    nbeads_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    row_track += 1
    ttk.Label(chnl_config, text="warpdeg:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    warpdeg_vals = tk.IntVar(chnl_config)
    warpdeg_vals.set(3)
    warpdeg_entry = ttk.Entry(chnl_config, width=21, textvariable=warpdeg_vals)
    warpdeg_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    row_track += 1
    ttk.Label(chnl_config, text="tol (nm):", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    tolr_nm_vals = tk.StringVar(chnl_config)
    tolr_nm_vals.set('35.0')
    tolr_nm_entry = ttk.Entry(chnl_config, width=21, textvariable=tolr_nm_vals)
    tolr_nm_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    ## terminate
    row_track += 1
    exit_button = tk.Button(chnl_config, text='Save and Run', font=(ft, 12), height=2, command=lambda : _tkclose(config_GUI, chnltab_active))
    exit_button.grid(column=4, row=row_track, pady=(20, 0), sticky='w')



    ## ################################ EXECUTE ################################ ##
    config_GUI.mainloop()
    
    if psftab_active[0]:

        # beads file information
        modality = modality_vals.get()
        bead_fname = psfbead_fname_vals.get()
        if not os.path.isfile(bead_fname):
            raise ValueError("The bead file is not found")
        bead_fpath = os.path.dirname(bead_fname) 
        
        # camera info
        view_type = view_type_vals.get() if modality != 'BP' else 'dualview'
        cam_pxszx = float(cam_pxszx_vals.get())
        cam_pxszy = float(cam_pxszy_vals.get())
        scmos_var = scmos_var_vals.get()
        
        if isfile_vals.get():
            cam_fname = cam_fname_vals.get()
            if not os.path.isfile(cam_fname_vals.get()):
                raise ValueError("no pkl file for cam_specs is not found")
            with open(cam_fname, 'rb') as fid:
                cam_specs = pickle.load(fid)
            chipszh = cam_specs['chipszh']
            chipszw = cam_specs['chipszw']
            cam_offset = cam_specs['offset']
            cam_var = cam_specs['var']
            cam_A2D = cam_specs['A2D']
            cam_gain = cam_specs['EMGain']
            
            if np.any([isinstance(item, (float, np.float32, np.float64)) for item in [cam_offset, cam_var, cam_A2D]]):
                scmos_var = False 
            if not scmos_var:
                cam_offset = np.mean(cam_offset) if isinstance(cam_offset, (list, tuple, np.ndarray)) else cam_offset
                cam_var = np.mean(cam_var) if isinstance(cam_var, (list, tuple, np.ndarray)) else cam_var
                cam_A2D = np.mean(cam_A2D) if isinstance(cam_A2D, (list, tuple, np.ndarray)) else cam_A2D
            
        else:
            scmos_var = False
            cam_fname = None
            chipszh = chipszx_vals.get()
            chipszw = chipszy_vals.get()
            cam_offset = float(cam_offset_vals.get())
            cam_var = float(cam_var_vals.get())
            cam_A2D = 1.0 / float(cam_A4D_vals.get())
            cam_gain = float(cam_emgain_vals.get())
        
        Camsetting = {  'cam_pth':      cam_fname,
                        'view_type':    view_type,
                        'chipszh':      chipszh, 
                        'chipszw':      chipszw,
                        'cam_pxszx':    cam_pxszx,
                        'cam_pxszy':    cam_pxszy,
                        'scmos_var':    scmos_var, 
                        'offset':       cam_offset, 
                        'var':          cam_var, 
                        'A2D':          cam_A2D, 
                        'EMGain':       cam_gain    }
        
        with open(os.path.join(bead_fpath, 'Camera_settings.pkl'), 'wb') as fid:
            pickle.dump(Camsetting, fid, pickle.HIGHEST_PROTOCOL)

        # axial info
        zstepsz_nm = float(zstepsz_vals.get())
        zfocus = None if modality == '2Dfrm' else np.array([zfocus_ch0_vals.get(), zfocus_ch1_vals.get()])
        cal_zrange_nm = float(cal_zrange_vals.get())
        
        # working parameters
        boxsz = boxsz_vals.get()
        nbeads = psfnbeads_vals.get()
        PSFsigmax_nm = 143.0
        PSFsigmay_nm = 143.0

        PSFsigmax = PSFsigmax_nm / cam_pxszx
        PSFsigmay = PSFsigmay_nm / cam_pxszy
        cal_zrange = 2 * np.int32(cal_zrange_nm/2.0/zstepsz_nm+0.5) + 1
        nbreaks = np.int32(cal_zrange_nm/50+0.5)
        
        ### run calibration
        ims, = im_loader(Camsetting, bead_fname)
        if modality == '2Dfrm':
            PSFcali_2dfrm_unlinked(bead_fpath, ims, nbeads, boxsz, PSFsigmax, PSFsigmay)
        else:
            PSFcali_3dstck_unlinked(modality, bead_fpath, ims, nbeads, boxsz, PSFsigmax, PSFsigmay, zstepsz_nm, zfocus, cal_zrange, reg_zrange=25, smoothZ=1.0, nbreaks=nbreaks)
    

    elif chnltab_active[0]:
        
        # beads file information
        bead_fname = chnlbead_fname_vals.get()
        if not os.path.isfile(bead_fname):
            raise ValueError("The bead file is not found")
        bead_fpath = os.path.dirname(bead_fname)

        with open(os.path.join(bead_fpath, 'Camera_settings.pkl'), 'rb') as fid:
            Camsetting = pickle.load(fid)
        with open(os.path.join(bead_fpath, 'PSFCalibration_unlinked.pkl'), 'rb') as fid:
            PSF = pickle.load(fid)
        
        # working parameters
        nbeads = chnlnbeads_vals.get()
        tolr_nm = float(tolr_nm_vals.get())
        warpdeg = warpdeg_vals.get()
        
        # run calibration
        ims, = im_loader(Camsetting, bead_fname, zPos=PSF['zFocus'])
        Segmentation = {'boxsz':PSF['boxsz'], 'filtmethod':'DoG', 'segmethod':'nms', 'threshmode':'dynamic', 'threshval':3.0}
        chnlwarp(bead_fpath, Camsetting, ims, nbeads, PSF, tolr_nm, warpdeg, Segmentation, iterations=100, batsz=65535)