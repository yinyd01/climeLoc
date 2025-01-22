import os
import json
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import numpy as np

from main_smlmpp import main_ratio_metric, main_pp, main_render


def _tkfopenfile(StrVar):
    fname = filedialog.askopenfilename()
    StrVar.set(fname)
    return

def _tkfopendir(StrVar):
    fpath = filedialog.askdirectory()
    StrVar.set(fpath)
    return

def _enablechk(isopt_vals, entries):
    if isopt_vals.get():
        for entry in entries:
            entry.config(state='normal')
    else:
        for entry in entries:
            entry.config(state='disable')
    return

def _disablechk(isopt_vals, entries):
    if isopt_vals.get():
        for entry in entries:
            entry.config(state='disable')
    else:
        for entry in entries:
            entry.config(state='normal')
    return

def _nfluorchk(nfluorophores_vals, flname_options, cmin_entries, cmax_entries, colormap_options):
    for i in range(nfluorophores_vals.get()):
        flname_options[i].config(state='readonly')
        cmin_entries[i].config(state='normal')
        cmax_entries[i].config(state='normal')
        colormap_options[i].config(state='readonly')
    for i in range(nfluorophores_vals.get(), 4):
        flname_options[i].config(state='disabled')
        cmin_entries[i].config(state='disabled')
        cmax_entries[i].config(state='disabled')
        colormap_options[i].config(state='disabled')
    return

def _rgbchk(isRGB_vals, feature_option, colorbits_entry, nfluorophores_vals, nfluorophores_option, flname_options, cmin_entries, cmax_entries, colormap_options):
    if isRGB_vals.get():
        feature_option.config(state='readonly')
        colorbits_entry.config(state='normal')
        nfluorophores_vals.set(1)
        nfluorophores_option.config(state='readonly')
        _nfluorchk(nfluorophores_vals, flname_options, cmin_entries, cmax_entries, colormap_options)     
    else:
        feature_option.config(state='disabled')
        colorbits_entry.config(state='disabled')
        nfluorophores_vals.set(1)
        nfluorophores_option.config(state='disable')
        _nfluorchk(nfluorophores_vals, flname_options, cmin_entries, cmax_entries, colormap_options)
    return
        
def _batchlvchk(isbatch_vals, batchlv_option):
    if isbatch_vals.get():
        batchlv_option.config(state='readonly')
    else:
        batchlv_option.config(state='disabled')

def _tkclose(win, tab_active):
    tab_active[0] = True
    win.destroy()

FLUOR_OPTION = ['DY634', 'AF647', 'CF660C', 'CF680']   
CMAP_OPTION = [ 'turbo', 'viridis', 'plasma', 'inferno', 'magma', 'cividis',
                'Red', 'Green', 'Blue', 'Cyan', 'Magenta', 'Yellow', 
                'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
                'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper']



if __name__ == '__main__':
    

    ## ################################ GUI WINDOW ################################ ##
    config_GUI = tk.Tk()
    config_GUI.title("SMLM Post Process")
    config_GUI.geometry("450x700")
    ft = 'Times New Roman'

    tab_control = ttk.Notebook(config_GUI)
    
    pp_config = ttk.Frame(tab_control)
    tab_control.add(pp_config, text='Post Process')
    pptab_active = [False]
    
    creg_config = ttk.Frame(tab_control)
    tab_control.add(creg_config, text='Fluor Registration')
    cregtab_active = [False]

    vis_config = ttk.Frame(tab_control)
    tab_control.add(vis_config, text='Visualization')
    vistab_active = [False]

    tab_control.pack(expand=1, fill='both')

    ## ################################ PP TAB ################################ ##
    row_track = -1
    
    ## ##### RAW LOCALIZATION FILES ##### ##
    row_track += 1
    label = ttk.Label(pp_config, text="raw localization files", font=(ft, 14))
    label.grid(column=0, columnspan=4, row=row_track, padx=10, pady=(20, 0), sticky='w')
    label.config(foreground='gray')

    # raw localization file
    row_track += 1
    ttk.Label(pp_config, text="_rawlocsnm.pkl:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    rawloc_fname_vals = tk.StringVar(pp_config)
    rawloc_fname_entry = ttk.Entry(pp_config, width=21, textvariable=rawloc_fname_vals, state='readonly')
    rawloc_fname_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')
    fopen_button = tk.Button(pp_config, text='Browse', font=(ft, 8), height=1, command=lambda : _tkfopenfile(rawloc_fname_vals))
    fopen_button.grid(column=4, row=row_track, sticky='w')

    ## ##### QUALITY CONTROL FILTER ##### ##
    row_track += 4
    label = ttk.Label(pp_config, text="QC filters", font=(ft, 14))
    label.grid(column=0, columnspan=4, row=row_track, padx=10, pady=(20, 0), sticky='w')
    label.config(foreground='gray')

    # p-value
    row_track += 1
    ttk.Label(pp_config, text="p-value:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    pval_vals = tk.StringVar(pp_config)
    pval_vals.set('1e-6')
    pval_entry = ttk.Entry(pp_config, width=21, textvariable=pval_vals)
    pval_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # filt_indf
    row_track += 1
    ttk.Label(pp_config, text="frm_rng:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    filt_fmin_vals = tk.IntVar(pp_config)
    filt_fmax_vals = tk.IntVar(pp_config)
    filt_fmin_vals.set(0)
    filt_fmax_vals.set(-1)
    filt_fmin_entry = ttk.Entry(pp_config, width=7, textvariable=filt_fmin_vals).grid(column=1, row=row_track, sticky='w')
    filt_fmax_entry = ttk.Entry(pp_config, width=7, textvariable=filt_fmax_vals).grid(column=2, row=row_track, sticky='w')

    # filt_photon
    row_track += 1
    ttk.Label(pp_config, text="min_photon:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    filt_photon_vals = tk.StringVar(pp_config)
    filt_photon_vals.set('200.0')
    filt_photon_entry = ttk.Entry(pp_config, width=21, textvariable=filt_photon_vals)
    filt_photon_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # filt_zrange
    row_track += 1
    ttk.Label(pp_config, text="zrange (nm):", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    filt_zmin_vals = tk.StringVar(pp_config)
    filt_zmax_vals = tk.StringVar(pp_config)
    filt_zmin_vals.set('100.0')
    filt_zmax_vals.set('900.0')
    filt_zmin_entry = ttk.Entry(pp_config, width=7, textvariable=filt_zmin_vals).grid(column=1, row=row_track, sticky='w')
    filt_zmax_entry = ttk.Entry(pp_config, width=7, textvariable=filt_zmax_vals).grid(column=2, row=row_track, sticky='w')

    # filt_precision
    row_track += 1
    ttk.Label(pp_config, text="prec (nm):", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    filt_precx_vals = tk.StringVar(pp_config)
    filt_precy_vals = tk.StringVar(pp_config)
    filt_precz_vals = tk.StringVar(pp_config)
    filt_precx_vals.set('10.0')
    filt_precy_vals.set('10.0')
    filt_precz_vals.set('30.0')
    filt_precx_entry = ttk.Entry(pp_config, width=7, textvariable=filt_precx_vals).grid(column=1, row=row_track, sticky='w')
    filt_precy_entry = ttk.Entry(pp_config, width=7, textvariable=filt_precy_vals).grid(column=2, row=row_track, sticky='w')
    filt_precz_entry = ttk.Entry(pp_config, width=7, textvariable=filt_precz_vals).grid(column=3, row=row_track, sticky='w')

    ## ##### RCC DRIFT ##### ##
    row_track += 4
    label = ttk.Label(pp_config, text="RCC drift correction", font=(ft, 14))
    label.grid(column=0, columnspan=4, row=row_track, padx=10, pady=(20, 0), sticky='w')
    label.config(foreground='gray')

    # rcc
    row_track += 1
    ttk.Label(pp_config, text="rcc:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    isrcc_vals = tk.BooleanVar(pp_config)
    isrcc_vals.set(True)
    isrcc_option = ttk.Checkbutton(pp_config, takefocus=0, variable=isrcc_vals, offvalue=False, onvalue=True, 
                    command=lambda : _enablechk(isrcc_vals, [rcc_tar_pxsz_entry, frmbinsz_entry, ccorrszx_entry, ccorrszy_entry, ccorrszz_entry, fitszx_entry, fitszy_entry, fitszz_entry]))
    isrcc_option.grid(column=1, row=row_track, padx=0, pady=0, sticky='w')

    # tar_pxsz
    row_track += 1
    ttk.Label(pp_config, text="tar_pxsz:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    rcc_tar_pxsz_vals = tk.IntVar(pp_config)
    rcc_tar_pxsz_vals.set(5)
    rcc_tar_pxsz_entry = ttk.Entry(pp_config, width=21, textvariable=rcc_tar_pxsz_vals, state='normal')
    rcc_tar_pxsz_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # frmbinsz
    row_track += 1
    ttk.Label(pp_config, text="frmbinsz:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    frmbinsz_vals = tk.IntVar(pp_config)
    frmbinsz_vals.set(200)
    frmbinsz_entry = ttk.Entry(pp_config, width=21, textvariable=frmbinsz_vals, state='normal')
    frmbinsz_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # ccorrsz
    row_track += 1
    ttk.Label(pp_config, text="ccorrsz (nm):", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    ccorrszx_vals = tk.StringVar(pp_config)
    ccorrszy_vals = tk.StringVar(pp_config)
    ccorrszz_vals = tk.StringVar(pp_config)
    ccorrszx_vals.set('250.0')
    ccorrszy_vals.set('250.0')
    ccorrszz_vals.set('800.0')
    ccorrszx_entry = ttk.Entry(pp_config, width=7, textvariable=ccorrszx_vals, state='normal')
    ccorrszx_entry.grid(column=1, row=row_track, sticky='w')
    ccorrszy_entry = ttk.Entry(pp_config, width=7, textvariable=ccorrszy_vals, state='normal')
    ccorrszy_entry.grid(column=2, row=row_track, sticky='w')
    ccorrszz_entry = ttk.Entry(pp_config, width=7, textvariable=ccorrszz_vals, state='normal')
    ccorrszz_entry.grid(column=3, row=row_track, sticky='w')
    
    # fitsz
    row_track += 1
    ttk.Label(pp_config, text="fitsz (nm):", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    fitszx_vals = tk.StringVar(pp_config)
    fitszy_vals = tk.StringVar(pp_config)
    fitszz_vals = tk.StringVar(pp_config)
    fitszx_vals.set('100.0')
    fitszy_vals.set('100.0')
    fitszz_vals.set('100.0')
    fitszx_entry = ttk.Entry(pp_config, width=7, textvariable=fitszx_vals, state='normal')
    fitszx_entry.grid(column=1, row=row_track, sticky='w')
    fitszy_entry = ttk.Entry(pp_config, width=7, textvariable=fitszy_vals, state='normal')
    fitszy_entry.grid(column=2, row=row_track, sticky='w')
    fitszz_entry = ttk.Entry(pp_config, width=7, textvariable=fitszz_vals, state='normal')
    fitszz_entry.grid(column=3, row=row_track, sticky='w')


    ## ##### CONSECUTIVE GROUP ##### ##
    row_track += 4
    label = ttk.Label(pp_config, text="In-Frame Merge & Consecutive Group", font=(ft, 14))
    label.grid(column=0, columnspan=4, row=row_track, padx=10, pady=(20, 0), sticky='w')
    label.config(foreground='gray')

    # ismerge
    row_track += 1
    ttk.Label(pp_config, text="merge:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    ismerge_vals = tk.BooleanVar(pp_config)
    ismerge_vals.set(False)
    ismerge_option = ttk.Checkbutton(pp_config, takefocus=0, variable=ismerge_vals, offvalue=False, onvalue=True, 
                        command=lambda : _enablechk(ismerge_vals, [maxRm_entry, maxZm_entry]))
    ismerge_option.grid(column=1, row=row_track, padx=0, pady=0, sticky='w')

    # maxRm
    row_track += 1
    ttk.Label(pp_config, text="maxR (nm):", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    maxRm_vals = tk.StringVar(pp_config)
    maxRm_vals.set('150.0')
    maxRm_entry = ttk.Entry(pp_config, width=21, textvariable=maxRm_vals, state='disabled')
    maxRm_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # maxRm
    row_track += 1
    ttk.Label(pp_config, text="maxZ (nm):", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    maxZm_vals = tk.StringVar(pp_config)
    maxZm_vals.set('250.0')
    maxZm_entry = ttk.Entry(pp_config, width=21, textvariable=maxZm_vals, state='disabled')
    maxZm_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # isgroup
    row_track += 1
    ttk.Label(pp_config, text="group:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    isgroup_vals = tk.BooleanVar(pp_config)
    isgroup_vals.set(False)
    isgroup_option = ttk.Checkbutton(pp_config, takefocus=0, variable=isgroup_vals, offvalue=False, onvalue=True, 
                        command=lambda : _enablechk(isgroup_vals, [maxRg_entry, maxZg_entry, gap_tol_entry]))
    isgroup_option.grid(column=1, row=row_track, padx=0, pady=0, sticky='w')

    # maxR
    row_track += 1
    ttk.Label(pp_config, text="maxR (nm):", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    maxRg_vals = tk.StringVar(pp_config)
    maxRg_vals.set('75.0')
    maxRg_entry = ttk.Entry(pp_config, width=21, textvariable=maxRg_vals, state='disabled')
    maxRg_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # maxZ
    row_track += 1
    ttk.Label(pp_config, text="maxR (nm):", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    maxZg_vals = tk.StringVar(pp_config)
    maxZg_vals.set('150.0')
    maxZg_entry = ttk.Entry(pp_config, width=21, textvariable=maxZg_vals, state='disabled')
    maxZg_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # gap_tol
    row_track += 1
    ttk.Label(pp_config, text="gap frame_tol:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    gap_tol_vals = tk.IntVar(pp_config)
    gap_tol_vals.set(1)
    gap_tol_entry = ttk.Entry(pp_config, width=21, textvariable=gap_tol_vals, state='disabled')
    gap_tol_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')


    ## ##### TERMINATE ##### ##
    row_track += 1
    pptab_active = [False]
    exit_button = tk.Button(pp_config, text='Save and Run', font=(ft, 12), height=2, command=lambda : _tkclose(config_GUI, pptab_active))
    exit_button.grid(column=4, row=row_track, pady=(20, 0), sticky='w')
    

    ## ################################ CREG TAB ################################ ##
    row_track = -1

    # raw localization files
    row_track += 1
    label = ttk.Label(creg_config, text="localization files", font=(ft, 14))
    label.grid(column=0, columnspan=4, row=row_track, padx=10, pady=(20, 0), sticky='w')
    label.config(foreground='gray')

    # localization file
    row_track += 1
    ttk.Label(creg_config, text="_locsnm.pkl:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    creg_fname_vals = tk.StringVar(creg_config)
    creg_fname_entry = ttk.Entry(creg_config, width=21, textvariable=creg_fname_vals, state='readonly')
    creg_fname_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')
    fopen_button = tk.Button(creg_config, text='Browse', font=(ft, 8), height=1, command=lambda : _tkfopenfile(creg_fname_vals))
    fopen_button.grid(column=4, row=row_track, sticky='w')

    # configuration
    row_track += 1
    label = ttk.Label(creg_config, text="Configurations", font=(ft, 14))
    label.grid(column=0, columnspan=4, row=row_track, padx=10, pady=(20, 0), sticky='w')
    label.config(foreground='gray')

    # fromfile
    row_track += 1
    ttk.Label(creg_config, text="fromfile:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    creg_isff_vals = tk.BooleanVar(creg_config)
    creg_isff_vals.set(False)
    creg_isff_option = ttk.Checkbutton(creg_config, takefocus=0, variable=creg_isff_vals, offvalue=False, onvalue=True, 
                                            command = lambda : _disablechk(creg_isff_vals, [creg_nfluor_option]))
    creg_isff_option.grid(column=1, row=row_track, padx=0, pady=0, sticky='w')

    # configuration json file
    row_track += 1
    ttk.Label(creg_config, text="config json:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    creg_jsfile_vals = tk.StringVar(creg_config)
    creg_jsfile_entry = ttk.Entry(creg_config, width=21, textvariable=creg_jsfile_vals, state='readonly')
    creg_jsfile_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')
    fopen_jsfile_button = tk.Button(creg_config, text='Browse', font=(ft, 8), height=1, command=lambda : _tkfopenfile(creg_jsfile_vals))
    fopen_jsfile_button.grid(column=4, row=row_track, sticky='w')

    # nfluorophores
    row_track += 1
    ttk.Label(creg_config, text="nfluorophores:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    creg_nfluor_vals = tk.IntVar(creg_config)
    creg_nfluor_vals.set(1)
    creg_nfluor_option = ttk.Combobox(creg_config, width=20, textvariable=creg_nfluor_vals, state='normal')
    creg_nfluor_option['values'] = [1, 2, 3, 4]
    creg_nfluor_option.grid(column=1, columnspan=2, row=row_track, sticky='w')
    
    ## ##### TERMINATE ##### ##
    row_track += 1
    cregtab_active = [False]
    exit_button = tk.Button(creg_config, text='Save and Run', font=(ft, 12), height=2, command=lambda : _tkclose(config_GUI, cregtab_active))
    exit_button.grid(column=4, row=row_track, pady=(20, 0), sticky='w')


    ## ################################ VIS TAB ################################ ##
    row_track = -1
    
    ## ##### LOCALIZATION FILES ##### ##
    row_track += 1
    label = ttk.Label(vis_config, text="localization Files", font=(ft, 14))
    label.grid(column=0, columnspan=4, row=row_track, padx=10, pady=(20, 0), sticky='w')
    label.config(foreground='gray')

    # localization file
    row_track += 1
    ttk.Label(vis_config, text="_locsnm.pkl:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    qcloc_fname_vals = tk.StringVar(vis_config)
    qcloc_fname_entry = ttk.Entry(vis_config, width=36, textvariable=qcloc_fname_vals, state='readonly')
    qcloc_fname_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')
    fopen_button = tk.Button(vis_config, text='Browse', font=(ft, 8), height=1, command=lambda : _tkfopenfile(qcloc_fname_vals))
    fopen_button.grid(column=4, row=row_track, sticky='w')

    
    ## ##### BASIC RENDER ##### ##
    row_track += 4
    label = ttk.Label(vis_config, text="Visualization", font=(ft, 14))
    label.grid(column=0, columnspan=4, row=row_track, padx=10, pady=(20, 0), sticky='w')
    label.config(foreground='gray')

    # tar_pxsz
    row_track += 1
    ttk.Label(vis_config, text="tar_pxsz:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    vis_tar_pxsz_vals = tk.StringVar(vis_config)
    vis_tar_pxsz_vals.set('10.0')
    vis_tar_pxsz_entry = ttk.Entry(vis_config, width=20, textvariable=vis_tar_pxsz_vals)
    vis_tar_pxsz_entry.grid(column=1, columnspan=2, row=row_track, sticky='w')

    # frm_range
    row_track += 1
    ttk.Label(vis_config, text="frm_range:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    frm_op_vals = tk.IntVar(vis_config)
    frm_ed_vals = tk.IntVar(vis_config)
    frm_op_vals.set(0)
    frm_ed_vals.set(-1)
    frm_op_entry = ttk.Entry(vis_config, width=10, textvariable=frm_op_vals, state='normal')
    frm_op_entry.grid(column=1, row=row_track, sticky='w')
    frm_ed_entry = ttk.Entry(vis_config, width=10, textvariable=frm_ed_vals, state='normal')
    frm_ed_entry.grid(column=2, row=row_track, sticky='w')
    
    # non-singlet
    row_track += 1
    ttk.Label(vis_config, text="non-singlet:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    nosinglet_vals = tk.BooleanVar(vis_config)
    nosinglet_vals.set(True)
    nosinglet_option = ttk.Checkbutton(vis_config, takefocus=0, variable=nosinglet_vals, offvalue=False, onvalue=True)
    nosinglet_option.grid(column=1, row=row_track, padx=0, pady=0, sticky='w')
    rlim_nm_vals = tk.StringVar(vis_config)
    rlim_nm_vals.set('50.0')
    rlim_nm_entry = ttk.Entry(vis_config, width=10, textvariable=rlim_nm_vals)
    rlim_nm_entry.grid(column=2, columnspan=1, row=row_track, sticky='w')
    ttk.Label(vis_config, text="(nm)", font=(ft, 10)).grid(column=3, row=row_track, padx=0, pady=0, sticky='w')

    # isblur
    row_track += 1
    ttk.Label(vis_config, text="crlb blur:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    isblur_vals = tk.BooleanVar(vis_config)
    isblur_vals.set(True)
    isblur_option = ttk.Checkbutton(vis_config, takefocus=0, variable=isblur_vals, offvalue=False, onvalue=True)
    isblur_option.grid(column=1, row=row_track, padx=0, pady=0, sticky='w')
    sig_vals = tk.StringVar(vis_config)
    sig_vals.set('-1.0')
    sig_entry = ttk.Entry(vis_config, width=10, textvariable=sig_vals)
    sig_entry.grid(column=2, columnspan=1, row=row_track, sticky='w')
    ttk.Label(vis_config, text="(nm)", font=(ft, 10)).grid(column=3, row=row_track, padx=0, pady=0, sticky='w')

    # intensity mode
    row_track += 1
    ttk.Label(vis_config, text="Intensity:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    Intensity_vals = tk.StringVar(vis_config)
    Intensity_vals.set('blink')
    Intensity_option = ttk.Combobox(vis_config, width=20, textvariable=Intensity_vals, state='readonly')
    Intensity_option['values'] = ['blink', 'photon']
    Intensity_option.grid(column=1, columnspan=2, row=row_track, sticky='w')

    # alpha
    row_track += 1
    ttk.Label(vis_config, text="alpha (0~1):", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    alpha_vals = tk.StringVar(vis_config)
    alpha_vals.set('0.8')
    alpha_entry = ttk.Entry(vis_config, width=20, textvariable=alpha_vals)
    alpha_entry.grid(column=1, columnspan=2, row=row_track, sticky='w')

    
    ## ##### RGB PROPERTIES ##### ##
    row_track += 4
    label = ttk.Label(vis_config, text="RGB Options", font=(ft, 14))
    label.grid(column=0, columnspan=4, row=row_track, padx=10, pady=(20, 0), sticky='w')
    label.config(foreground='gray')

    # isRGB
    row_track += 1
    ttk.Label(vis_config, text="RGB disp:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    isRGB_vals = tk.BooleanVar(vis_config)
    isRGB_vals.set(True)
    isRGB_option = ttk.Checkbutton(vis_config, takefocus=0, variable=isRGB_vals, offvalue=False, onvalue=True, 
                    command=lambda : _rgbchk(isRGB_vals, feature_option, nfluorophores_vals, nfluorophores_option, 
                                            [ch0_name_option, ch1_name_option, ch2_name_option, ch3_name_option], 
                                            [ch0_cmin_entry, ch1_cmin_entry, ch2_cmin_entry, ch3_cmin_entry],
                                            [ch0_cmax_entry, ch1_cmax_entry, ch2_cmax_entry, ch3_cmax_entry],
                                            [ch0_colormap_option, ch1_colormap_option, ch2_colormap_option, ch3_colormap_option]))
    isRGB_option.grid(column=1, row=row_track, padx=0, pady=0, sticky='w')
    
    # feature
    row_track += 1
    ttk.Label(vis_config, text="feature:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    feature_vals = tk.StringVar(vis_config)
    feature_vals.set('photon')
    feature_option = ttk.Combobox(vis_config, width=20, textvariable=feature_vals, state='normal')
    feature_option['values'] = ['photon', 'loss', 'locz', 'precision_x', 'precision_y', 'precision_z', 'indf']
    feature_option.grid(column=1, columnspan=2, row=row_track, sticky='w')

    # zmin_nm
    row_track += 1
    ttk.Label(vis_config, text="zmin:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    zmin_vals = tk.StringVar(vis_config)
    zmin_vals.set('100.0')
    zmin_entry = ttk.Entry(vis_config, width=10, textvariable=zmin_vals, state='normal')
    zmin_entry.grid(column=1, row=row_track, sticky='w')
    ttk.Label(vis_config, text="(nm):", font=(ft, 10)).grid(column=2, row=row_track, padx=10, pady=0, sticky='w')

    # zmax_nm
    row_track += 1
    ttk.Label(vis_config, text="zmax:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    zmax_vals = tk.StringVar(vis_config)
    zmax_vals.set('900.0')
    zmax_entry = ttk.Entry(vis_config, width=10, textvariable=zmax_vals, state='normal')
    zmax_entry.grid(column=1, row=row_track, sticky='w')
    ttk.Label(vis_config, text="(nm):", font=(ft, 10)).grid(column=2, row=row_track, padx=10, pady=0, sticky='w')

    # clim
    row_track += 1
    ttk.Label(vis_config, text="clim (0.0~1.0):", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='w')

    # nfluorophores
    row_track += 1
    ttk.Label(vis_config, text="nfluorophores:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    nfluorophores_vals = tk.IntVar(vis_config)
    nfluorophores_vals.set(1)
    nfluorophores_option = ttk.Combobox(vis_config, width=20, textvariable=nfluorophores_vals, state='normal')
    nfluorophores_option['values'] = [1, 2, 3, 4]
    nfluorophores_option.grid(column=1, columnspan=2, row=row_track, sticky='w')
    nfluorophores_option.bind("<<ComboboxSelected>>", lambda _ : _nfluorchk(nfluorophores_vals, 
                                                                            [ch0_name_option, ch1_name_option, ch2_name_option, ch3_name_option], 
                                                                            [ch0_cmin_entry, ch1_cmin_entry, ch2_cmin_entry, ch3_cmin_entry],
                                                                            [ch0_cmax_entry, ch1_cmax_entry, ch2_cmax_entry, ch3_cmax_entry],
                                                                            [ch0_colormap_option, ch1_colormap_option, ch2_colormap_option, ch3_colormap_option]))
    
    row_track += 1
    ch0_name_vals = tk.StringVar(vis_config)
    ch0_name_vals.set('AF647')
    ch0_name_option = ttk.Combobox(vis_config, width=10, textvariable=ch0_name_vals, state='normal')
    ch0_name_option['values'] = FLUOR_OPTION
    ch0_name_option.grid(column=0, row=row_track, sticky='e')
    ch0_cmin_vals = tk.StringVar(vis_config)
    ch0_cmax_vals = tk.StringVar(vis_config)
    ch0_colormap_vals = tk.StringVar(vis_config)
    ch0_cmin_vals.set('0.0')
    ch0_cmax_vals.set('1.0')
    ch0_colormap_vals.set('turbo')
    ch0_cmin_entry = ttk.Entry(vis_config, width=10, textvariable=ch0_cmin_vals, state='normal')
    ch0_cmax_entry = ttk.Entry(vis_config, width=10, textvariable=ch0_cmax_vals, state='normal')
    ch0_colormap_option = ttk.Combobox(vis_config, width=10, textvariable=ch0_colormap_vals, state='normal')
    ch0_colormap_option['values'] = CMAP_OPTION
    ch0_cmin_entry.grid(column=1, row=row_track, sticky='w')
    ch0_cmax_entry.grid(column=2, row=row_track, sticky='w')
    ch0_colormap_option.grid(column=3, row=row_track, sticky='w')

    row_track += 1
    ch1_name_vals = tk.StringVar(vis_config)
    ch1_name_vals.set('None')
    ch1_name_option = ttk.Combobox(vis_config, width=10, textvariable=ch1_name_vals, state='disabled')
    ch1_name_option['values'] = FLUOR_OPTION
    ch1_name_option.grid(column=0, row=row_track, sticky='e')
    ch1_cmin_vals = tk.StringVar(vis_config)
    ch1_cmax_vals = tk.StringVar(vis_config)
    ch1_colormap_vals = tk.StringVar(vis_config)
    ch1_cmin_vals.set('0.0')
    ch1_cmax_vals.set('1.0')
    ch1_colormap_vals.set('turbo')
    ch1_cmin_entry = ttk.Entry(vis_config, width=10, textvariable=ch1_cmin_vals, state='disabled')
    ch1_cmax_entry = ttk.Entry(vis_config, width=10, textvariable=ch1_cmax_vals, state='disabled')
    ch1_colormap_option = ttk.Combobox(vis_config, width=10, textvariable=ch1_colormap_vals, state='disabled')
    ch1_colormap_option['values'] = CMAP_OPTION
    ch1_cmin_entry.grid(column=1, row=row_track, sticky='w')
    ch1_cmax_entry.grid(column=2, row=row_track, sticky='w')
    ch1_colormap_option.grid(column=3, row=row_track, sticky='w')

    row_track += 1
    ch2_name_vals = tk.StringVar(vis_config)
    ch2_name_vals.set('None')
    ch2_name_option = ttk.Combobox(vis_config, width=10, textvariable=ch2_name_vals, state='disabled')
    ch2_name_option['values'] = FLUOR_OPTION
    ch2_name_option.grid(column=0, row=row_track, sticky='e')
    ch2_cmin_vals = tk.StringVar(vis_config)
    ch2_cmax_vals = tk.StringVar(vis_config)
    ch2_colormap_vals = tk.StringVar(vis_config)
    ch2_cmin_vals.set('0.0')
    ch2_cmax_vals.set('1.0')
    ch2_colormap_vals.set('turbo')
    ch2_cmin_entry = ttk.Entry(vis_config, width=10, textvariable=ch2_cmin_vals, state='disabled')
    ch2_cmax_entry = ttk.Entry(vis_config, width=10, textvariable=ch2_cmax_vals, state='disabled')
    ch2_colormap_option = ttk.Combobox(vis_config, width=10, textvariable=ch2_colormap_vals, state='disabled')
    ch2_colormap_option['values'] = CMAP_OPTION
    ch2_cmin_entry.grid(column=1, row=row_track, sticky='w')
    ch2_cmax_entry.grid(column=2, row=row_track, sticky='w')
    ch2_colormap_option.grid(column=3, row=row_track, sticky='w')

    row_track += 1
    ch3_name_vals = tk.StringVar(vis_config)
    ch3_name_vals.set('None')
    ch3_name_option = ttk.Combobox(vis_config, width=10, textvariable=ch1_name_vals, state='disabled')
    ch3_name_option['values'] = FLUOR_OPTION
    ch3_name_option.grid(column=0, row=row_track, sticky='e')
    ch3_cmin_vals = tk.StringVar(vis_config)
    ch3_cmax_vals = tk.StringVar(vis_config)
    ch3_colormap_vals = tk.StringVar(vis_config)
    ch3_cmin_vals.set('0.0')
    ch3_cmax_vals.set('1.0')
    ch3_colormap_vals.set('turbo')
    ch3_cmin_entry = ttk.Entry(vis_config, width=10, textvariable=ch3_cmin_vals, state='disabled')
    ch3_cmax_entry = ttk.Entry(vis_config, width=10, textvariable=ch3_cmax_vals, state='disabled')
    ch3_colormap_option = ttk.Combobox(vis_config, width=10, textvariable=ch3_colormap_vals, state='disabled')
    ch3_colormap_option['values'] = CMAP_OPTION
    ch3_cmin_entry.grid(column=1, row=row_track, sticky='w')
    ch3_cmax_entry.grid(column=2, row=row_track, sticky='w')
    ch3_colormap_option.grid(column=3, row=row_track, sticky='w')


    ## ##### TERMINATE ##### ##
    row_track += 1
    exit_button = tk.Button(vis_config, text='Save and Run', font=(ft, 12), height=2, command=lambda : _tkclose(config_GUI, vistab_active))
    exit_button.grid(column=4, row=row_track, pady=(20, 0), sticky='w')



    ## ################################ EXECUTE ################################ ##
    config_GUI.mainloop()
    
    if cregtab_active[0]:
        
        if creg_isff_vals.get():
            with open(creg_jsfile_vals.get(), 'r') as jsfid:
                config_RatioMetric = json.load(jsfid)
            print(config_RatioMetric)
            assert 'crng' in config_RatioMetric.keys(), "no crng found in the {:s} file".format(creg_jsfile_vals.get())
        else:
            config_RatioMetric = {'nfluorophores':  creg_nfluor_vals.get()}
        
        fpath_save = os.path.dirname(creg_fname_vals.get())
        main_ratio_metric(config_RatioMetric, creg_fname_vals.get())
        

    elif pptab_active[0]:
        
        qcfilter = {'pval':                 float(pval_vals.get()),
                    'frm_rng':              [filt_fmin_vals.get(), filt_fmax_vals.get()],
                    'filt_zrange':          np.asarray([filt_zmin_vals.get(), filt_zmax_vals.get()], dtype=np.float32).tolist(),
                    'filt_photon':          float(filt_photon_vals.get()),
                    'filt_precision_nm':    np.asarray([filt_precx_vals.get(), filt_precy_vals.get(), filt_precz_vals.get()], dtype=np.float32).tolist()}
        
        rcc = { 'isrcc':        isrcc_vals.get(),
                'tar_pxsz':     float(rcc_tar_pxsz_vals.get()),
                'frmbinsz':     frmbinsz_vals.get(),
                'ccorrsz_nm':   np.asarray([ccorrszx_vals.get(), ccorrszy_vals.get(), ccorrszz_vals.get()], dtype=np.float32).tolist(),
                'fitsz_nm':     np.asarray([fitszx_vals.get(), fitszy_vals.get(), fitszz_vals.get()], dtype=np.float32).tolist()}
        
        inframe_merge = {'ismerge':  ismerge_vals.get(),
                        'maxR_nm':  float(maxRm_vals.get()),
                        'maxZ_nm':  float(maxZm_vals.get())}
        
        consec_merge = {'isgroup':  isgroup_vals.get(),
                        'maxR_nm':  float(maxRg_vals.get()),
                        'maxZ_nm':  float(maxZg_vals.get()), 
                        'gap_tol':  gap_tol_vals.get()}
        
        config_qc = {'qcfilter':qcfilter, 'rcc':rcc, 'infrm_merge':inframe_merge, 'consec_merge':consec_merge}
        
        jsonObj = json.dumps(config_qc, indent=4)
        fpath_save = os.path.dirname(rawloc_fname_vals.get())
        main_pp(config_qc, rawloc_fname_vals.get())
        with open(os.path.join(fpath_save, 'qc_settings.json'), "w") as jsfid:
            jsfid.write(jsonObj)
    


    elif vistab_active[0]:
        
        nfluorophores = nfluorophores_vals.get()
        
        if nfluorophores == 1:
            fluornames = [ch0_name_vals.get()]
            colormaps = [ch0_colormap_vals.get()]
            clims = [[ch0_cmin_vals.get(), ch0_cmax_vals.get()]]
        elif nfluorophores == 2:
            fluornames = [ch0_name_vals.get(), ch1_name_vals.get()]
            colormaps = [ch0_colormap_vals.get(), ch1_colormap_vals.get()]
            clims = [[ch0_cmin_vals.get(), ch0_cmax_vals.get()],
                     [ch1_cmin_vals.get(), ch1_cmax_vals.get()]]  
        elif nfluorophores == 3:
            fluornames = [ch0_name_vals.get(), ch1_name_vals.get(), ch2_name_vals.get()]
            colormaps = [ch0_colormap_vals.get(), ch1_colormap_vals.get(), ch2_colormap_vals.get()]
            clims = [[ch0_cmin_vals.get(), ch0_cmax_vals.get()],
                     [ch1_cmin_vals.get(), ch1_cmax_vals.get()],
                     [ch2_cmin_vals.get(), ch2_cmax_vals.get()]]
        elif nfluorophores == 4:
            fluornames = [ch0_name_vals.get(), ch1_name_vals.get(), ch2_name_vals.get(), ch3_name_vals.get()]
            colormaps = [ch0_colormap_vals.get(), ch1_colormap_vals.get(), ch2_colormap_vals.get(), ch3_colormap_vals.get()]
            clims = [[ch0_cmin_vals.get(), ch0_cmax_vals.get()],
                     [ch1_cmin_vals.get(), ch1_cmax_vals.get()],
                     [ch2_cmin_vals.get(), ch2_cmax_vals.get()],
                     [ch3_cmin_vals.get(), ch3_cmax_vals.get()]]
        
        config_render = {   'tar_pxsz':         float(vis_tar_pxsz_vals.get()),
                            'frm_range':        [frm_op_vals.get(), frm_ed_vals.get()],
                            'nosinglet':        nosinglet_vals.get(),
                            'rlim_nm':          float(rlim_nm_vals.get()),
                            'isblur':           isblur_vals.get(),
                            'sig':              float(sig_vals.get()) / float(vis_tar_pxsz_vals.get()),
                            'intensity':        Intensity_vals.get(),
                            'alpha':            float(alpha_vals.get()),
                            'isRGB':            isRGB_vals.get(),
                            'feature':          feature_vals.get(),
                            'zmin_nm':          float(zmin_vals.get()),
                            'zmax_nm':          float(zmax_vals.get()),
                            'nfluorophores':    nfluorophores,
                            'fluorname':        fluornames,
                            'clim':             clims,
                            'colormap':         colormaps   }

        jsonObj = json.dumps(config_render, indent=4)
        fpath_save = os.path.dirname(qcloc_fname_vals.get())
        main_render(config_render, qcloc_fname_vals.get())
        with open(os.path.join(fpath_save, 'render_settings.json'), "w") as jsfid:
            jsfid.write(jsonObj)
    