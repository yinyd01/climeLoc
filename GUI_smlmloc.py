import os
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from main_smlmloc import parachk, main_localization


def _tkfopenfile(StrVar):
    fname = filedialog.askopenfilename()
    StrVar.set(fname)

def _tkfopendir(StrVar):
    fpath = filedialog.askdirectory()
    StrVar.set(fpath)

def _threshmodechk(threshmode_option, threshval_vals):
    if threshmode_option.get() == 'dynamic':
        threshval_vals.set('1.7')
    elif threshmode_option.get() == 'pvalue':
        threshval_vals.set('0.05')

def _isdensechk(isdense_vals, nmax_entry):
    if isdense_vals.get():
        nmax_entry.config(state='normal')
        pval_entry.config(state='normal')
    else:
        nmax_entry.config(state='disabled')
        pval_entry.config(state='disabled')

def _tkexit(win):
    win.destroy()
   


if __name__ == '__main__':
    
    config_GUI = tk.Tk()
    config_GUI.title("SMLM configurations")
    config_GUI.geometry("375x700")
    ft = 'Times New Roman'
    row_track = -1


    # Camsetting and PSFs
    row_track += 1
    label = ttk.Label(config_GUI, text="Camsetting and PSF", font=(ft, 14))
    label.grid(column=0, columnspan=4, row=row_track, padx=10, pady=(20, 0), sticky='w')
    label.config(foreground='gray')

    # Camsetting
    row_track += 1
    ttk.Label(config_GUI, text="Camsetting:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    cam_fname_vals = tk.StringVar(config_GUI)
    cam_fname_entry = ttk.Entry(config_GUI, width=21, textvariable=cam_fname_vals, state='readonly')
    cam_fname_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')
    fopen_button = tk.Button(config_GUI, text='Browse', font=(ft, 8), height=1, command=lambda : _tkfopenfile(cam_fname_vals))
    fopen_button.grid(column=4, row=row_track, sticky='w')

    row_track += 1
    ttk.Label(config_GUI, text="emgain:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    cam_emgain_vals = tk.StringVar(config_GUI)
    cam_emgain_vals.set('None')
    cam_emgain_entry = ttk.Entry(config_GUI, width=21, textvariable=cam_emgain_vals, state='normal')
    cam_emgain_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # PSF
    row_track += 1
    ttk.Label(config_GUI, text="PSF:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    psf_fname_vals = tk.StringVar(config_GUI)
    psf_fname_entry = ttk.Entry(config_GUI, width=21, textvariable=psf_fname_vals, state='readonly')
    psf_fname_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')
    fopen_button = tk.Button(config_GUI, text='Browse', font=(ft, 8), height=1, command=lambda : _tkfopenfile(psf_fname_vals))
    fopen_button.grid(column=4, row=row_track, sticky='w')

    # Channel_warp
    row_track += 1
    ttk.Label(config_GUI, text="Chnlwarp:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    chnlwarp_fname_vals = tk.StringVar(config_GUI)
    chnlwarp_fname_vals.set('None')
    chnlwarp_fname_entry = ttk.Entry(config_GUI, width=21, textvariable=chnlwarp_fname_vals, state='readonly')
    chnlwarp_fname_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')
    fopen_button = tk.Button(config_GUI, text='Browse', font=(ft, 8), height=1, command=lambda : _tkfopenfile(chnlwarp_fname_vals))
    fopen_button.grid(column=4, row=row_track, sticky='w')
    
    
    # Image Files
    row_track += 1
    label = ttk.Label(config_GUI, text="Image Files", font=(ft, 14))
    label.grid(column=0, columnspan=4, row=row_track, padx=10, pady=(20, 0), sticky='w')
    label.config(foreground='gray')

    # image file
    row_track += 1
    ttk.Label(config_GUI, text="img file:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    im_fpath_vals = tk.StringVar(config_GUI)
    im_fpath_entry = ttk.Entry(config_GUI, width=21, textvariable=im_fpath_vals, state='readonly')
    im_fpath_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')
    fopen_button = tk.Button(config_GUI, text='Browse', font=(ft, 8), height=1, command=lambda : _tkfopendir(im_fpath_vals))
    fopen_button.grid(column=4, row=row_track, sticky='w')

    # roi file
    row_track += 1
    ttk.Label(config_GUI, text="roi file:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    roi_fname_vals = tk.StringVar(config_GUI)
    roi_fname_vals.set('None')
    roi_fname_entry = ttk.Entry(config_GUI, width=21, textvariable=roi_fname_vals, state='readonly')
    roi_fname_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')
    fopen_button = tk.Button(config_GUI, text='Browse', font=(ft, 8), height=1, command=lambda : _tkfopenfile(roi_fname_vals))
    fopen_button.grid(column=4, row=row_track, sticky='w')


    # Image Segmentation
    row_track += 1
    label = ttk.Label(config_GUI, text="Image Segmentation", font=(ft, 14))
    label.grid(column=0, columnspan=4, row=row_track, padx=10, pady=(20, 0), sticky='w')
    label.config(foreground='gray')

    # boxsz
    row_track += 1
    ttk.Label(config_GUI, text="boxsz:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    boxsz_vals = tk.IntVar(config_GUI)
    boxsz_vals.set(13)
    boxsz_entry = ttk.Entry(config_GUI, width=21, textvariable=boxsz_vals)
    boxsz_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # filtmethod
    row_track += 1
    ttk.Label(config_GUI, text="filter method:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    filtmethod_vals = tk.StringVar(config_GUI)
    filtmethod_vals.set('DoG')
    filtmethod_option = ttk.Combobox(config_GUI, width=21, textvariable=filtmethod_vals, state='readonly')
    filtmethod_option['values'] = ['Gauss', 'DoG', 'DoA', 'PSF', 'MIP']
    filtmethod_option.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # segmethod
    row_track += 1
    ttk.Label(config_GUI, text="segmethod:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    segmethod_vals = tk.StringVar(config_GUI)
    segmethod_vals.set('nms')
    segmethod_option = ttk.Combobox(config_GUI, width=21, textvariable=segmethod_vals, state='readonly')
    segmethod_option['values'] = ['nms', 'grid', 'None']
    segmethod_option.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # threshold
    row_track += 1
    ttk.Label(config_GUI, text="threshold:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    threshmode_vals = tk.StringVar(config_GUI)
    threshmode_vals.set('dynamic')
    threshmode_option = ttk.Combobox(config_GUI, width=12, textvariable=threshmode_vals, state='readonly')
    threshmode_option['values'] = ['dynamic', 'pvalue']
    threshmode_option.grid(column=1, columnspan=2, row=row_track, sticky='w')
    
    threshval_vals = tk.StringVar(config_GUI)
    threshval_vals.set('1.7')
    threshmode_option.bind("<<ComboboxSelected>>", lambda _ : _threshmodechk(threshmode_option, threshval_vals))
    threshval_entry = ttk.Entry(config_GUI, width=7, textvariable=threshval_vals)
    threshval_entry.grid(column=3, row=row_track, sticky='w')
    

    # PSFFIT
    row_track += 4
    label = ttk.Label(config_GUI, text="PSF Fitter", font=(ft, 14))
    label.grid(column=0, columnspan=4, row=row_track, padx=10, pady=(20, 0), sticky='w')
    label.config(foreground='gray')

    # isdense
    row_track += 1
    ttk.Label(config_GUI, text="isdense:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    _tmp_isdensechk = lambda : _isdensechk(isdense_vals, nmax_entry)
    isdense_vals = tk.BooleanVar(config_GUI)
    isdense_vals.set(False)
    isdense_option = ttk.Checkbutton(config_GUI, takefocus=0, variable=isdense_vals, offvalue=False, onvalue=True, command=_tmp_isdensechk)
    isdense_option.grid(column=1, row=row_track, padx=0, pady=0, sticky='w')

    # nmax
    row_track += 1
    ttk.Label(config_GUI, text="nmax:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    nmax_vals = tk.IntVar(config_GUI)
    nmax_vals.set(5)
    nmax_entry = ttk.Entry(config_GUI, width=21, textvariable=nmax_vals, state='disabled')
    nmax_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # pval
    row_track += 1
    ttk.Label(config_GUI, text="pval:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    pval_vals = tk.StringVar(config_GUI)
    pval_vals.set('0.05')
    pval_entry = ttk.Entry(config_GUI, width=21, textvariable=pval_vals, state='disabled')
    pval_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # fit method
    row_track += 1
    ttk.Label(config_GUI, text="fit method:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    modality_vals = tk.StringVar(config_GUI)
    modality_vals.set('AS')
    modality_option = ttk.Combobox(config_GUI, width=4, textvariable=modality_vals, state='readonly')
    modality_option['values'] = ['2Dfrm', '2D', 'AS', 'BP', 'DH']
    modality_option.grid(column=1, row=row_track, sticky='w')

    fitmethod_vals = tk.StringVar(config_GUI)
    fitmethod_vals.set('cspline')
    fitmethod_option = ttk.Combobox(config_GUI, width=13, textvariable=fitmethod_vals, state='readonly')
    fitmethod_option['values'] = ['cspline', 'gauss']
    fitmethod_option.grid(column=2, columnspan=2, row=row_track, sticky='w')
    
    # iterations
    row_track += 1
    ttk.Label(config_GUI, text="iterations:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    iterations_vals = tk.IntVar(config_GUI)
    iterations_vals.set(100)
    iterations_entry = ttk.Entry(config_GUI, width=21, textvariable=iterations_vals)
    iterations_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # batchsz
    row_track += 1
    ttk.Label(config_GUI, text="GPU batchsz:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    batsz_vals = tk.IntVar(config_GUI)
    batsz_vals.set(10000)
    batsz_entry = ttk.Entry(config_GUI, width=21, textvariable=batsz_vals)
    batsz_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    
    # TERMINATE
    row_track += 1
    exit_button = tk.Button(config_GUI, text='Save and Run', font=(ft, 12), height=2, command=lambda : _tkexit(config_GUI))
    exit_button.grid(column=4, row=row_track, pady=(20, 0), sticky='w')
    
    config_GUI.mainloop()


    # collect the parameters
    if not os.path.isfile(cam_fname_vals.get()):
        raise ValueError("The camera file is not found")
    if not os.path.isfile(psf_fname_vals.get()):
        raise ValueError("The PSF calibration file is not found")
    if not os.path.isdir(im_fpath_vals.get()):
        raise ValueError("The image path is not found")
    if chnlwarp_fname_vals.get() != 'None' and not os.path.isfile(roi_fname_vals.get()):
        raise ValueError("The roi fname is not None but the given file is not found")
    if chnlwarp_fname_vals.get() != 'None' and not os.path.isfile(chnlwarp_fname_vals.get()):
        raise ValueError("The channel warp fname is not None but the given file is not found")
   
    Segmentation = {'boxsz':boxsz_vals.get(),
                    'filtmethod':filtmethod_vals.get(), 
                    'segmethod':segmethod_vals.get(), 
                    'threshmode':threshmode_vals.get(),
                    'threshval':float(threshval_vals.get()) }
    
    PSFfit = {  'modality':modality_vals.get(),   
                'isdense':isdense_vals.get(),
                'nmax':nmax_vals.get(),
                'pval':float(pval_vals.get()),
                'fitmethod':'cspline' if modality_vals.get() in {'2D', 'BP', 'DH'} else fitmethod_vals.get(),
                'optmethod':'MLE',
                'iterations':iterations_vals.get(),
                'batsz':batsz_vals.get()    }

    config_loc = {'Camsetting':cam_fname_vals.get(), 'cam_emgain':cam_emgain_vals.get(), 'PSF':psf_fname_vals.get(), 'chnl_warp':chnlwarp_fname_vals.get(),
                    'img_fpath':im_fpath_vals.get(), 'roi_fname':roi_fname_vals.get(), 'Segmentation':Segmentation, 'PSFfit':PSFfit}

    Camsetting, PSF, coeff_R2T, coeff_T2R, PSFsigmax, PSFsigmay = parachk(config_loc)
    main_localization(config_loc, Camsetting, PSF, coeff_R2T, coeff_T2R, PSFsigmax, PSFsigmay)