import os
import pickle
import json
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import numpy as np
import pandas as pd

from LOCAnalysis.cluster_optics import optics_batch
from LOCAnalysis.cluster_dbscan import dbscan_batch
from LOCAnalysis.PairCorrelation import acorr_batch, ccorr_batch


def _tkfopen(StrVar):
    fname = filedialog.askopenfilename()
    StrVar.set(fname)

def _tkfopendir(StrVar):
    fpath = filedialog.askdirectory()
    StrVar.set(fpath)

def _typechk(corrtype_vals, funcname_option):
    funcname_option['values'] = funcnames[corrtype_vals.get()]
    return

def _optchk(isopt_vals, entries):
    if isopt_vals.get():
        for entry in entries:
            entry.config(state='normal')
    else:
        for entry in entries:
            entry.config(state='disable')
    return

def _tkclose(win, tab_active):
    tab_active[0] = True
    win.destroy()


funcnames = {'ccorr':['gauss', 'agauss'], 'acorr':['gauss', 'exp', 'strexp']}
redges = np.array(list(range(0,32,2)) + list(range(32,192,5)) + list(range(192,513,20)))
rcenters = 0.5 * (redges[:-1] + redges[1:])


if __name__ == '__main__':
    
    ## ################################ GUI WINDOW ################################ ##
    config_GUI = tk.Tk()
    config_GUI.title("SMLM_DATA ANALYSIS")
    config_GUI.geometry("450x350")
    ft = 'Times New Roman'

    tab_control = ttk.Notebook(config_GUI)
    
    cluster_config = ttk.Frame(tab_control)
    tab_control.add(cluster_config, text='Cluster')
    clustertab_active = [False]
    
    pcorr_config = ttk.Frame(tab_control)
    tab_control.add(pcorr_config, text='PCorr')
    pcorrtab_active = [False]

    tab_control.pack(expand=1, fill='both')

    ## ################################ CLUSTER TAB ################################ ##
    row_track = -1

    # version
    row_track += 1
    ttk.Label(cluster_config, text="version:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    cl_version_vals = tk.StringVar(cluster_config)
    cl_version_vals.set('1.11')
    cl_version_option = ttk.Combobox(cluster_config, width=21, textvariable=cl_version_vals, state='readonly')
    cl_version_option['values'] = ['1.11', '1.10']
    cl_version_option.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # method
    row_track += 1
    ttk.Label(cluster_config, text="method:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    method_vals = tk.StringVar(cluster_config)
    method_vals.set('optics')
    method_option = ttk.Combobox(cluster_config, width=21, textvariable=method_vals, state='readonly')
    method_option['values'] = ['dbscan', 'optics']
    method_option.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # ndim
    row_track += 1
    ttk.Label(cluster_config, text="ndim:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    clndim_vals = tk.IntVar(cluster_config)
    clndim_vals.set(3)
    clndim_option = ttk.Combobox(cluster_config, width=21, textvariable=clndim_vals, state='readonly')
    clndim_option['values'] = [2, 3]
    clndim_option.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # max epsilon
    row_track += 1
    ttk.Label(cluster_config, text="max_eps(nm):", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    max_eps_vals = tk.StringVar(cluster_config)
    max_eps_vals.set('150')
    max_eps_entry = ttk.Entry(cluster_config, width=21, textvariable=max_eps_vals)
    max_eps_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # minpts
    row_track += 1
    ttk.Label(cluster_config, text="MinPts:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    MinPts_vals = tk.IntVar(cluster_config)
    MinPts_vals.set(25)
    MinPts_entry = ttk.Entry(cluster_config, width=21, textvariable=MinPts_vals)
    MinPts_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # xi_factor
    row_track += 1
    ttk.Label(cluster_config, text="xi:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    xi_vals = tk.StringVar(cluster_config)
    xi_vals.set('0.05')
    xi_entry = ttk.Entry(cluster_config, width=21, textvariable=xi_vals)
    xi_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # batch level
    row_track += 1
    cl_batchlv_vals = tk.StringVar(cluster_config)
    cl_batchlv_vals.set('spools')
    ttk.Label(cluster_config, text="batch lv:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    cl_batchlv_option = ttk.Combobox(cluster_config, width=21, textvariable=cl_batchlv_vals, state='normal')
    cl_batchlv_option['values'] = ['spools', 'samples']
    cl_batchlv_option.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # image file
    row_track += 1
    ttk.Label(cluster_config, text="smlm_locs file:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    cl_im_fpath_vals = tk.StringVar(cluster_config)
    cl_im_fpath_entry = ttk.Entry(cluster_config, width=21, textvariable=cl_im_fpath_vals)
    cl_im_fpath_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')
    cl_fopen_button = tk.Button(cluster_config, text='Browse', font=(ft, 8), height=1, command=lambda : _tkfopendir(cl_im_fpath_vals))
    cl_fopen_button.grid(column=4, row=row_track, sticky='w')

    ## terminate
    row_track += 1
    clustertab_active = [False]
    exit_button = tk.Button(cluster_config, text='Save and Run', font=(ft, 12), height=2, command=lambda : _tkclose(config_GUI, clustertab_active))
    exit_button.grid(column=4, row=row_track, pady=(20, 0), sticky='w')
    

    ## ################################ PCORR TAB ################################ ##
    row_track = -1

    # version
    row_track += 1
    ttk.Label(pcorr_config, text="version:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    pc_version_vals = tk.StringVar(pcorr_config)
    pc_version_vals.set('1.11')
    pc_version_option = ttk.Combobox(pcorr_config, width=21, textvariable=pc_version_vals, state='readonly')
    pc_version_option['values'] = ['1.11', '1.10']
    pc_version_option.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # type
    row_track += 1
    ttk.Label(pcorr_config, text="type:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    type_vals = tk.StringVar(pcorr_config)
    type_vals.set('ccorr')
    type_option = ttk.Combobox(pcorr_config, width=21, textvariable=type_vals, state='readonly')
    type_option.bind("<<ComboboxSelected>>", lambda _ : _typechk(type_vals, funcname_option))
    type_option['values'] = ['ccorr', 'acorr']
    type_option.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # ndim
    row_track += 1
    ttk.Label(pcorr_config, text="ndim:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    pcndim_vals = tk.IntVar(pcorr_config)
    pcndim_vals.set(3)
    pcndim_option = ttk.Combobox(pcorr_config, width=21, textvariable=pcndim_vals, state='readonly')
    pcndim_option['values'] = [2, 3]
    pcndim_option.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # fluor ID
    row_track += 1
    ttk.Label(pcorr_config, text="flID_A:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    flID_A_vals = tk.IntVar(pcorr_config)
    flID_A_vals.set(0)
    flID_A_option = ttk.Combobox(pcorr_config, width=21, textvariable=flID_A_vals, state='readonly')
    flID_A_option['values'] = [0, 1, 2, 3]
    flID_A_option.grid(column=1, columnspan=3, row=row_track, sticky='w')

    row_track += 1
    ttk.Label(pcorr_config, text="flID_B:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    flID_B_vals = tk.IntVar(pcorr_config)
    flID_B_vals.set(1)
    flID_B_option = ttk.Combobox(pcorr_config, width=21, textvariable=flID_B_vals, state='readonly')
    flID_B_option['values'] = [0, 1, 2, 3]
    flID_B_option.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # isfit
    row_track += 1
    ttk.Label(pcorr_config, text="fit:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    isfit_vals = tk.BooleanVar(pcorr_config)
    isfit_vals.set(True)
    isfit_option = ttk.Checkbutton(pcorr_config, takefocus=0, variable=isfit_vals, offvalue=False, onvalue=True, 
                        command=lambda : _optchk(isfit_vals, [funcname_option]))
    isfit_option.grid(column=1, row=row_track, padx=0, pady=0, sticky='w')

    # funcname
    row_track += 1
    ttk.Label(pcorr_config, text="func:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    funcname_vals = tk.StringVar(pcorr_config)
    funcname_vals.set('gauss')
    funcname_option = ttk.Combobox(pcorr_config, width=21, textvariable=funcname_vals, state='readonly')
    funcname_option['values'] = funcnames['ccorr'] if type_vals.get() == 'ccorr' else funcnames['acorr']
    funcname_option.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # batch level
    row_track += 1
    pc_batchlv_vals = tk.StringVar(pcorr_config)
    pc_batchlv_vals.set('spools')
    ttk.Label(pcorr_config, text="batch lv:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    pc_batchlv_option = ttk.Combobox(pcorr_config, width=21, textvariable=pc_batchlv_vals, state='normal')
    pc_batchlv_option['values'] = ['spools', 'samples']
    pc_batchlv_option.grid(column=1, columnspan=3, row=row_track, sticky='w')

    # image file
    row_track += 1
    ttk.Label(pcorr_config, text="smlm_locs file:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    pc_im_fpath_vals = tk.StringVar(pcorr_config)
    pc_im_fpath_entry = ttk.Entry(pcorr_config, width=21, textvariable=pc_im_fpath_vals)
    pc_im_fpath_entry.grid(column=1, columnspan=3, row=row_track, sticky='w')
    pc_fopen_button = tk.Button(pcorr_config, text='Browse', font=(ft, 8), height=1, command=lambda : _tkfopendir(pc_im_fpath_vals))
    pc_fopen_button.grid(column=4, row=row_track, sticky='w')

    ## terminate
    row_track += 1
    pcorrtab_active = [False]
    exit_button = tk.Button(pcorr_config, text='Save and Run', font=(ft, 12), height=2, command=lambda : _tkclose(config_GUI, pcorrtab_active))
    exit_button.grid(column=4, row=row_track, pady=(20, 0), sticky='w')



    ## ################################ EXECUTE ################################ ##
    config_GUI.mainloop()

    if clustertab_active[0]:
        # collect the parameters
        if not os.path.isdir(cl_im_fpath_vals.get()):
            raise ValueError("The file path is not found")
        fpath = cl_im_fpath_vals.get()
        
        version = cl_version_vals.get()
        method = method_vals.get()
        ndim = clndim_vals.get()
        max_eps = float(max_eps_vals.get())
        minpts = MinPts_vals.get()
        xi_factor = float(xi_vals.get())
        batchlv = cl_batchlv_vals.get()
        
        config_cluster = {'version':version, 'method':method, 'ndim':ndim, 'max_eps':max_eps, 'MinPts':minpts, 'xi_factor':xi_factor, 'batchlv':batchlv, 'fpath':fpath}
        jsonObj = json.dumps(config_cluster, indent=4)
        with open(os.path.join(fpath, 'config_cluster.json'), 'w') as jsfid:
            jsfid.write(jsonObj)

        ### run calibration
        if method == 'optics':
            if batchlv == 'spools':
                cl_statistics, cl_properties = optics_batch(fpath, ndim, max_eps, minpts, xi_factor, version=version)
                for i in range(4):
                    if len(cl_statistics[i]['roiname']) > 0:
                        df = pd.DataFrame(cl_statistics[i])
                        df.to_csv(os.path.join(fpath, os.path.basename(fpath)+'_{}_statistics_fl{}.csv'.format(method, i)))
                        df = pd.DataFrame(cl_properties[i])
                        df.to_csv(os.path.join(fpath, os.path.basename(fpath)+'_{}_properties_fl{}.csv'.format(method, i)))
            
            elif batchlv == 'samples':
                sample_fpaths = [sample_fpath for sample_fpath in os.listdir(fpath) if os.path.isdir(os.path.join(fpath, sample_fpath))]
                for sample_fpath in sample_fpaths:
                    sample_fpath = os.path.join(fpath, sample_fpath)
                    cl_statistics, cl_properties = optics_batch(sample_fpath, ndim, max_eps, minpts, xi_factor, version=version)
                    for i in range(4):
                        if len(cl_statistics[i]['roiname']) > 0:
                            df = pd.DataFrame(cl_statistics[i])
                            df.to_csv(os.path.join(sample_fpath, os.path.basename(sample_fpath)+'_{}_statistics_fl{}.csv'.format(method, i)))
                            df = pd.DataFrame(cl_properties[i])
                            df.to_csv(os.path.join(sample_fpath, os.path.basename(sample_fpath)+'_{}_properties_fl{}.csv'.format(method, i)))
        
        else:
            if batchlv == 'spools':
                cl_statistics, cl_properties = dbscan_batch(fpath, ndim, max_eps, minpts, version=version)
                for i in range(4):
                    if len(cl_statistics[i]['roiname']) > 0:
                        df = pd.DataFrame(cl_statistics[i])
                        df.to_csv(os.path.join(fpath, os.path.basename(fpath)+'_{}_statistics_fl{}.csv'.format(method, i)))
                        df = pd.DataFrame(cl_properties[i])
                        df.to_csv(os.path.join(fpath, os.path.basename(fpath)+'_{}_properties_fl{}.csv'.format(method, i)))
            
            elif batchlv == 'samples':
                sample_fpaths = [sample_fpath for sample_fpath in os.listdir(fpath) if os.path.isdir(os.path.join(fpath, sample_fpath))]
                for sample_fpath in sample_fpaths:
                    sample_fpath = os.path.join(fpath, sample_fpath)
                    cl_statistics, cl_properties = dbscan_batch(sample_fpath, ndim, max_eps, minpts, version=version)
                    for i in range(4):
                        if len(cl_statistics[i]['roiname']) > 0:
                            df = pd.DataFrame(cl_statistics[i])
                            df.to_csv(os.path.join(sample_fpath, os.path.basename(sample_fpath)+'_{}_statistics_fl{}.csv'.format(method, i)))
                            df = pd.DataFrame(cl_properties[i])
                            df.to_csv(os.path.join(sample_fpath, os.path.basename(sample_fpath)+'_{}_properties_fl{}.csv'.format(method, i)))
    
    elif pcorrtab_active[0]:
        # collect the parameters
        if not os.path.isdir(pc_im_fpath_vals.get()):
            raise ValueError("The file path is not found")
        fpath = pc_im_fpath_vals.get()
        
        version = pc_version_vals.get()
        corrtype = type_vals.get()
        ndim = pcndim_vals.get()
        flID_A = flID_A_vals.get()
        flID_B = flID_B_vals.get()
        fit = isfit_vals.get()
        funcname = funcname_vals.get() if fit else None
        batchlv = pc_batchlv_vals.get()
        
        config_pcorr = {'version':version, 'type':corrtype, 'ndim':ndim, 'flID_A':flID_A, 'flID_B':flID_B, 'fit':fit, 'funcname':funcname, 'batchlv':batchlv, 'fpath':fpath}
        jsonObj = json.dumps(config_pcorr, indent=4)
        with open(os.path.join(fpath, 'config_pcorr.json'), 'w') as jsfid:
            jsfid.write(jsonObj)

        ### run calibration
        if corrtype == 'acorr':
            if batchlv == 'spools':
                ACData, ACSumm, ACFits, ACFitsErr = acorr_batch(fpath, ndim, redges, fit, funcname, version=version)
                for i in range(4):
                    if len(ACSumm[i]['roiname']) > 0:
                        for key in ACData[i].keys():
                            ACData[i][key] = np.hstack((rcenters[...,np.newaxis], ACData[i][key]))
                            fname = os.path.join(fpath, os.path.basename(fpath)+'_{}_fl{}.csv'.format(key, i))
                            np.savetxt(fname, ACData[i][key], delimiter=',', comments='')
                        df = pd.DataFrame(ACSumm[i])
                        df.to_csv(os.path.join(fpath, os.path.basename(fpath)+'_{}_Summ_fl{}.csv'.format(corrtype, i)))
                        if fit:
                            df = pd.DataFrame(ACFits[i])
                            df.to_csv(os.path.join(fpath, os.path.basename(fpath)+'_{}_Fits_fl{}.csv'.format(corrtype, i)))
                            df = pd.DataFrame(ACFitsErr[i])
                            df.to_csv(os.path.join(fpath, os.path.basename(fpath)+'_{}_FitsErr_fl{}.csv'.format(corrtype, i))) 
                        
            elif batchlv == 'samples':
                sample_fpaths = [sample_fpath for sample_fpath in os.listdir(fpath) if os.path.isdir(os.path.join(fpath, sample_fpath))]
                for sample_fpath in sample_fpaths:
                    sample_fpath = os.path.join(fpath, sample_fpath)
                    ACData, ACSumm, ACFits, ACFitsErr = acorr_batch(sample_fpath, ndim, redges, fit, funcname, version=version)
                    for i in range(4):
                        if len(ACSumm[i]['roiname']) > 0:
                            for key in ACData[i].keys():
                                ACData[i][key] = np.hstack((rcenters[...,np.newaxis], ACData[i][key]))
                                fname = os.path.join(sample_fpath, os.path.basename(sample_fpath)+'_{}_fl{}.csv'.format(key, i))
                                np.savetxt(fname, ACData[i][key], delimiter=',', comments='')
                            df = pd.DataFrame(ACSumm[i])
                            df.to_csv(os.path.join(sample_fpath, os.path.basename(sample_fpath)+'_{}_Summ_fl{}.csv'.format(corrtype, i)))
                            if fit:
                                df = pd.DataFrame(ACFits[i])
                                df.to_csv(os.path.join(sample_fpath, os.path.basename(sample_fpath)+'_{}_Fits_fl{}.csv'.format(corrtype, i)))
                                df = pd.DataFrame(ACFitsErr[i])
                                df.to_csv(os.path.join(sample_fpath, os.path.basename(sample_fpath)+'_{}_FitsErr_fl{}.csv'.format(corrtype, i))) 
                            
        else:
            if batchlv == 'spools':
                CCData, CCSumm, CCFits, CCFitsErr = ccorr_batch(fpath, ndim, redges, flID_A, flID_B, fit, funcname, version=version)
                for key in CCData.keys():
                    CCData[key] = np.hstack((rcenters[...,np.newaxis], CCData[key]))
                    fname = os.path.join(fpath, os.path.basename(fpath)+'_{}_fl{}_{}.csv'.format(key, flID_A, flID_B))
                    np.savetxt(fname, CCData[key], delimiter=',', comments='')
                df = pd.DataFrame(CCSumm)
                df.to_csv(os.path.join(fpath, os.path.basename(fpath)+'_{}_Summ_fl{}_{}.csv'.format(corrtype, flID_A, flID_B)))    
                df = pd.DataFrame(CCFits)
                df.to_csv(os.path.join(fpath, os.path.basename(fpath)+'_{}_Fits_fl{}_{}.csv'.format(corrtype, flID_A, flID_B)))
                df = pd.DataFrame(CCFitsErr)
                df.to_csv(os.path.join(fpath, os.path.basename(fpath)+'_{}_FitsErr_fl{}_{}.csv'.format(corrtype, flID_A, flID_B)))
            
            elif batchlv == 'samples':
                sample_fpaths = [sample_fpath for sample_fpath in os.listdir(fpath) if os.path.isdir(os.path.join(fpath, sample_fpath))]
                for sample_fpath in sample_fpaths:
                    sample_fpath = os.path.join(fpath, sample_fpath)
                    CCData, CCSumm, CCFits, CCFitsErr = ccorr_batch(sample_fpath, ndim, redges, flID_A, flID_B, fit, funcname, version=version)
                    for key in CCData.keys():
                        CCData[key] = np.hstack((rcenters[...,np.newaxis], CCData[key]))
                        fname = os.path.join(sample_fpath, os.path.basename(sample_fpath)+'_{}_fl{}_{}.csv'.format(key, flID_A, flID_B))
                        np.savetxt(fname, CCData[key], delimiter=',', comments='')
                    df = pd.DataFrame(CCSumm)
                    df.to_csv(os.path.join(sample_fpath, os.path.basename(sample_fpath)+'_{}_Summ_fl{}_{}.csv'.format(corrtype, flID_A, flID_B)))    
                    df = pd.DataFrame(CCFits)
                    df.to_csv(os.path.join(sample_fpath, os.path.basename(sample_fpath)+'_{}_Fits_fl{}_{}.csv'.format(corrtype, flID_A, flID_B)))
                    df = pd.DataFrame(CCFitsErr)
                    df.to_csv(os.path.join(sample_fpath, os.path.basename(sample_fpath)+'_{}_FitsErr_fl{}_{}.csv'.format(corrtype, flID_A, flID_B)))