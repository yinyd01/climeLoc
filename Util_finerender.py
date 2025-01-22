import os
import json
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

from main_smlmpp import main_finer_render 


FLUOR_OPTION = ['DY634', 'AF647', 'CF660C', 'CF680']   
CMAP_OPTION = [ 'turbo', 'viridis', 'plasma', 'inferno', 'magma', 'cividis',
                'Red', 'Green', 'Blue', 'Cyan', 'Magenta', 'Yellow', 
                'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
                'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper']


def _tkfopen_pkl_file(StrVar):
    fname = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
    StrVar.set(fname)

def _tkfopen_roi_file(StrVar):
    fname = filedialog.askopenfilename(filetypes=[("Zip Files", "*.zip"), ("Roi Files", "*.roi")])
    StrVar.set(fname)

def _flID_chk(flID_vals, flID_name_option, flID_cmin_entry, flID_cmax_entry, flID_colormap_option):
    if flID_vals.get():
        flID_name_option.config(state = 'normal')
        flID_cmin_entry.config(state = 'normal')
        flID_cmax_entry.config(state = 'normal')
        flID_colormap_option.config(state = 'normal')
    if not flID_vals.get():
        flID_name_option.config(state = 'disable')
        flID_cmin_entry.config(state = 'disable')
        flID_cmax_entry.config(state = 'disable')
        flID_colormap_option.config(state = 'disable')
    

def _tkclose(win):
    win.destroy()



if __name__ == '__main__':
    
    config_GUI = tk.Tk()
    config_GUI.title("Finer Render")
    config_GUI.geometry("450x700")
    ft = 'Times New Roman'
    row_track = -1
    
    ## ##### LOCALIZATION FILES ##### ##
    row_track += 1
    label = ttk.Label(config_GUI, text="Localization File", font=(ft, 14))
    label.grid(column=0, columnspan=6, row=row_track, padx=10, pady=(20, 0), sticky='w')
    label.config(foreground='gray')

    # localization file
    row_track += 1
    ttk.Label(config_GUI, text="_locsnm.pkl:", font=(ft, 10)).grid(column=0, columnspan=2, row=row_track, padx=10, pady=0, sticky='e')
    loc_fname_vals = tk.StringVar(config_GUI)
    loc_fname_entry = ttk.Entry(config_GUI, width=36, textvariable=loc_fname_vals, state='readonly')
    loc_fname_entry.grid(column=2, columnspan=3, row=row_track, sticky='w')
    fopen_button = tk.Button(config_GUI, text='Browse', font=(ft, 8), height=1, command=lambda : _tkfopen_pkl_file(loc_fname_vals))
    fopen_button.grid(column=5, row=row_track, sticky='w')

    # roi file
    row_track += 1
    ttk.Label(config_GUI, text="_.roi/zip:", font=(ft, 10)).grid(column=0, columnspan=2, row=row_track, padx=10, pady=0, sticky='e')
    roi_fname_vals = tk.StringVar(config_GUI)
    roi_fname_entry = ttk.Entry(config_GUI, width=36, textvariable=roi_fname_vals, state='readonly')
    roi_fname_entry.grid(column=2, columnspan=3, row=row_track, sticky='w')
    fopen_button = tk.Button(config_GUI, text='Browse', font=(ft, 8), height=1, command=lambda : _tkfopen_roi_file(roi_fname_vals))
    fopen_button.grid(column=5, row=row_track, sticky='w')

    # roi pxsz
    row_track += 1
    ttk.Label(config_GUI, text="roi_pxsz:", font=(ft, 10)).grid(column=0, columnspan=2, row=row_track, padx=10, pady=0, sticky='e')
    roi_pxsz_vals = tk.StringVar(config_GUI)
    roi_pxsz_vals.set('10.0')
    roi_pxsz_entry = ttk.Entry(config_GUI, width=20, textvariable=roi_pxsz_vals)
    roi_pxsz_entry.grid(column=2, columnspan=2, row=row_track, sticky='w')

    # intensity mode
    row_track += 1
    ttk.Label(config_GUI, text="top/side:", font=(ft, 10)).grid(column=0, columnspan=2, row=row_track, padx=10, pady=0, sticky='e')
    tos_vals = tk.StringVar(config_GUI)
    tos_vals.set('xy')
    tos_option = ttk.Combobox(config_GUI, width=20, textvariable=tos_vals, state='readonly')
    tos_option['values'] = ['xy', 'xz']
    tos_option.grid(column=2, columnspan=2, row=row_track, sticky='w')

    # thickness
    row_track += 1
    ttk.Label(config_GUI, text="thickness(nm):", font=(ft, 10)).grid(column=0, columnspan=2, row=row_track, padx=10, pady=0, sticky='e')
    roi_thickness_vals = tk.StringVar(config_GUI)
    roi_thickness_vals.set('50.0')
    roi_thickness_entry = ttk.Entry(config_GUI, width=20, textvariable=roi_thickness_vals)
    roi_thickness_entry.grid(column=2, columnspan=2, row=row_track, sticky='w')

    ## ##### BASIC RENDER ##### ##
    row_track += 4
    label = ttk.Label(config_GUI, text="Visualization", font=(ft, 14))
    label.grid(column=0, columnspan=6, row=row_track, padx=10, pady=(20, 0), sticky='w')
    label.config(foreground='gray')

    # tar_pxsz
    row_track += 1
    ttk.Label(config_GUI, text="tar_pxsz:", font=(ft, 10)).grid(column=0, columnspan=2, row=row_track, padx=10, pady=0, sticky='e')
    tar_pxsz_vals = tk.StringVar(config_GUI)
    tar_pxsz_vals.set('10.0')
    tar_pxsz_entry = ttk.Entry(config_GUI, width=20, textvariable=tar_pxsz_vals)
    tar_pxsz_entry.grid(column=2, columnspan=2, row=row_track, sticky='w')

    # frm_range
    row_track += 1
    ttk.Label(config_GUI, text="frm_range:", font=(ft, 10)).grid(column=0, columnspan=2, row=row_track, padx=10, pady=0, sticky='e')
    frm_op_vals = tk.IntVar(config_GUI)
    frm_ed_vals = tk.IntVar(config_GUI)
    frm_op_vals.set(0)
    frm_ed_vals.set(-1)
    frm_op_entry = ttk.Entry(config_GUI, width=10, textvariable=frm_op_vals, state='normal')
    frm_op_entry.grid(column=2, row=row_track, sticky='w')
    frm_ed_entry = ttk.Entry(config_GUI, width=10, textvariable=frm_ed_vals, state='normal')
    frm_ed_entry.grid(column=3, row=row_track, sticky='w')
    
    # non-singlet
    row_track += 1
    ttk.Label(config_GUI, text="non-singlet:", font=(ft, 10)).grid(column=0, columnspan=2, row=row_track, padx=10, pady=0, sticky='e')
    nosinglet_vals = tk.BooleanVar(config_GUI)
    nosinglet_vals.set(True)
    nosinglet_option = ttk.Checkbutton(config_GUI, takefocus=0, variable=nosinglet_vals, offvalue=False, onvalue=True)
    nosinglet_option.grid(column=2, row=row_track, padx=0, pady=0, sticky='w')
    rlim_nm_vals = tk.StringVar(config_GUI)
    rlim_nm_vals.set('50.0')
    rlim_nm_entry = ttk.Entry(config_GUI, width=10, textvariable=rlim_nm_vals)
    rlim_nm_entry.grid(column=3, columnspan=1, row=row_track, sticky='w')
    ttk.Label(config_GUI, text="(nm)", font=(ft, 10)).grid(column=4, row=row_track, padx=0, pady=0, sticky='w')

    # isblur
    row_track += 1
    ttk.Label(config_GUI, text="isblur:", font=(ft, 10)).grid(column=0, columnspan=2, row=row_track, padx=10, pady=0, sticky='e')
    isblur_vals = tk.BooleanVar(config_GUI)
    isblur_vals.set(True)
    isblur_option = ttk.Checkbutton(config_GUI, takefocus=0, variable=isblur_vals, offvalue=False, onvalue=True)
    isblur_option.grid(column=2, row=row_track, padx=0, pady=0, sticky='w')
    sig_vals = tk.StringVar(config_GUI)
    sig_vals.set('-1.0')
    sig_entry = ttk.Entry(config_GUI, width=10, textvariable=sig_vals)
    sig_entry.grid(column=3, columnspan=1, row=row_track, sticky='w')
    ttk.Label(config_GUI, text="(nm)", font=(ft, 10)).grid(column=4, row=row_track, padx=0, pady=0, sticky='w')

    # intensity mode
    row_track += 1
    ttk.Label(config_GUI, text="Intensity:", font=(ft, 10)).grid(column=0, columnspan=2, row=row_track, padx=10, pady=0, sticky='e')
    Intensity_vals = tk.StringVar(config_GUI)
    Intensity_vals.set('blink')
    Intensity_option = ttk.Combobox(config_GUI, width=20, textvariable=Intensity_vals, state='readonly')
    Intensity_option['values'] = ['blink', 'photon']
    Intensity_option.grid(column=2, columnspan=2, row=row_track, sticky='w')

    # alpha
    row_track += 1
    ttk.Label(config_GUI, text="alpha (0~1):", font=(ft, 10)).grid(column=0, columnspan=2, row=row_track, padx=10, pady=0, sticky='e')
    alpha_vals = tk.StringVar(config_GUI)
    alpha_vals.set('0.8')
    alpha_entry = ttk.Entry(config_GUI, width=20, textvariable=alpha_vals)
    alpha_entry.grid(column=2, columnspan=2, row=row_track, sticky='w')

    
    ## ##### RGB PROPERTIES ##### ##
    row_track += 4
    label = ttk.Label(config_GUI, text="RGB Options", font=(ft, 14))
    label.grid(column=0, columnspan=6, row=row_track, padx=10, pady=(20, 0), sticky='w')
    label.config(foreground='gray')

    # feature
    row_track += 1
    ttk.Label(config_GUI, text="feature:", font=(ft, 10)).grid(column=0, columnspan=2, row=row_track, padx=10, pady=0, sticky='e')
    feature_vals = tk.StringVar(config_GUI)
    feature_vals.set('photon')
    feature_option = ttk.Combobox(config_GUI, width=20, textvariable=feature_vals, state='normal')
    feature_option['values'] = ['photon', 'loss', 'locz', 'precision_x', 'precision_y', 'precision_z', 'indf']
    feature_option.grid(column=2, columnspan=2, row=row_track, sticky='w')

    # zmin_nm
    row_track += 1
    ttk.Label(config_GUI, text="zmin:", font=(ft, 10)).grid(column=0, columnspan=2, row=row_track, padx=10, pady=0, sticky='e')
    zmin_vals = tk.StringVar(config_GUI)
    zmin_vals.set('100.0')
    zmin_entry = ttk.Entry(config_GUI, width=10, textvariable=zmin_vals, state='normal')
    zmin_entry.grid(column=2, row=row_track, sticky='w')
    ttk.Label(config_GUI, text="(nm):", font=(ft, 10)).grid(column=3, row=row_track, padx=10, pady=0, sticky='w')

    # zmax_nm
    row_track += 1
    ttk.Label(config_GUI, text="zmax:", font=(ft, 10)).grid(column=0, columnspan=2, row=row_track, padx=10, pady=0, sticky='e')
    zmax_vals = tk.StringVar(config_GUI)
    zmax_vals.set('900.0')
    zmax_entry = ttk.Entry(config_GUI, width=10, textvariable=zmax_vals, state='normal')
    zmax_entry.grid(column=2, row=row_track, sticky='w')
    ttk.Label(config_GUI, text="(nm):", font=(ft, 10)).grid(column=3, row=row_track, padx=10, pady=0, sticky='w')

    # colorbits
    row_track += 1
    colorbits_vals = tk.IntVar(config_GUI)
    colorbits_vals.set(256)
    ttk.Label(config_GUI, text="colorbits:", font=(ft, 10)).grid(column=0, columnspan=2, row=row_track, padx=10, pady=0, sticky='e')
    colorbits_entry = ttk.Entry(config_GUI, width=20, textvariable=colorbits_vals, state='normal')
    colorbits_entry.grid(column=2, columnspan=2, row=row_track, sticky='w')

    row_track += 1
    ttk.Label(config_GUI, text="flID_0:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    flID_0_vals = tk.BooleanVar(config_GUI)
    flID_0_vals.set(False)
    _flID_0_chk = lambda : _flID_chk(flID_0_vals, flID_0_name_option, flID_0_cmin_entry, flID_0_cmax_entry, flID_0_colormap_option)
    flID_0_option = ttk.Checkbutton(config_GUI, takefocus=0, variable=flID_0_vals, offvalue=False, onvalue=True, command=_flID_0_chk)
    flID_0_option.grid(column=1, row=row_track, padx=0, pady=0, sticky='w')
    flID_0_name_vals = tk.StringVar(config_GUI)
    flID_0_name_vals.set('AF647')
    flID_0_name_option = ttk.Combobox(config_GUI, width=10, textvariable=flID_0_name_vals, state='disable')
    flID_0_name_option['values'] = FLUOR_OPTION
    flID_0_name_option.grid(column=2, row=row_track, sticky='e')
    flID_0_cmin_vals = tk.StringVar(config_GUI)
    flID_0_cmin_vals.set('0.1')
    flID_0_cmin_entry = ttk.Entry(config_GUI, width=10, textvariable=flID_0_cmin_vals, state='disable')
    flID_0_cmin_entry.grid(column=3, row=row_track, sticky='w')
    flID_0_cmax_vals = tk.StringVar(config_GUI)
    flID_0_cmax_vals.set('0.9')
    flID_0_cmax_entry = ttk.Entry(config_GUI, width=10, textvariable=flID_0_cmax_vals, state='disable')
    flID_0_cmax_entry.grid(column=4, row=row_track, sticky='w')
    flID_0_colormap_vals = tk.StringVar(config_GUI) 
    flID_0_colormap_vals.set('turbo')
    flID_0_colormap_option = ttk.Combobox(config_GUI, width=10, textvariable=flID_0_colormap_vals, state='disable')
    flID_0_colormap_option['values'] = CMAP_OPTION
    flID_0_colormap_option.grid(column=5, row=row_track, sticky='w')

    row_track += 1
    ttk.Label(config_GUI, text="flID_1:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    flID_1_vals = tk.BooleanVar(config_GUI)
    flID_1_vals.set(False)
    _flID_1_chk = lambda : _flID_chk(flID_1_vals, flID_1_name_option, flID_1_cmin_entry, flID_1_cmax_entry, flID_1_colormap_option)
    flID_1_option = ttk.Checkbutton(config_GUI, takefocus=0, variable=flID_1_vals, offvalue=False, onvalue=True, command=_flID_1_chk)
    flID_1_option.grid(column=1, row=row_track, padx=0, pady=0, sticky='w')
    flID_1_name_vals = tk.StringVar(config_GUI)
    flID_1_name_vals.set('CF660C')
    flID_1_name_option = ttk.Combobox(config_GUI, width=10, textvariable=flID_1_name_vals, state='disable')
    flID_1_name_option['values'] = FLUOR_OPTION
    flID_1_name_option.grid(column=2, row=row_track, sticky='e')
    flID_1_cmin_vals = tk.StringVar(config_GUI)
    flID_1_cmin_vals.set('0.1')
    flID_1_cmin_entry = ttk.Entry(config_GUI, width=10, textvariable=flID_1_cmin_vals, state='disable')
    flID_1_cmin_entry.grid(column=3, row=row_track, sticky='w')
    flID_1_cmax_vals = tk.StringVar(config_GUI)
    flID_1_cmax_vals.set('0.9')
    flID_1_cmax_entry = ttk.Entry(config_GUI, width=10, textvariable=flID_1_cmax_vals, state='disable')
    flID_1_cmax_entry.grid(column=4, row=row_track, sticky='w')
    flID_1_colormap_vals = tk.StringVar(config_GUI) 
    flID_1_colormap_vals.set('turbo')
    flID_1_colormap_option = ttk.Combobox(config_GUI, width=10, textvariable=flID_1_colormap_vals, state='disable')
    flID_1_colormap_option['values'] = CMAP_OPTION
    flID_1_colormap_option.grid(column=5, row=row_track, sticky='w')

    row_track += 1
    ttk.Label(config_GUI, text="flID_2:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    flID_2_vals = tk.BooleanVar(config_GUI)
    flID_2_vals.set(False)
    _flID_2_chk = lambda : _flID_chk(flID_2_vals, flID_2_name_option, flID_2_cmin_entry, flID_2_cmax_entry, flID_2_colormap_option)
    flID_2_option = ttk.Checkbutton(config_GUI, takefocus=0, variable=flID_2_vals, offvalue=False, onvalue=True, command=_flID_2_chk)
    flID_2_option.grid(column=1, row=row_track, padx=0, pady=0, sticky='w')
    flID_2_name_vals = tk.StringVar(config_GUI)
    flID_2_name_vals.set('CF680')
    flID_2_name_option = ttk.Combobox(config_GUI, width=10, textvariable=flID_2_name_vals, state='disable')
    flID_2_name_option['values'] = FLUOR_OPTION
    flID_2_name_option.grid(column=2, row=row_track, sticky='e')
    flID_2_cmin_vals = tk.StringVar(config_GUI)
    flID_2_cmin_vals.set('0.1')
    flID_2_cmin_entry = ttk.Entry(config_GUI, width=10, textvariable=flID_2_cmin_vals, state='disable')
    flID_2_cmin_entry.grid(column=3, row=row_track, sticky='w')
    flID_2_cmax_vals = tk.StringVar(config_GUI)
    flID_2_cmax_vals.set('0.9')
    flID_2_cmax_entry = ttk.Entry(config_GUI, width=10, textvariable=flID_2_cmax_vals, state='disable')
    flID_2_cmax_entry.grid(column=4, row=row_track, sticky='w')
    flID_2_colormap_vals = tk.StringVar(config_GUI) 
    flID_2_colormap_vals.set('turbo')
    flID_2_colormap_option = ttk.Combobox(config_GUI, width=10, textvariable=flID_2_colormap_vals, state='disable')
    flID_2_colormap_option['values'] = CMAP_OPTION
    flID_2_colormap_option.grid(column=5, row=row_track, sticky='w')

    row_track += 1
    ttk.Label(config_GUI, text="flID_3:", font=(ft, 10)).grid(column=0, row=row_track, padx=10, pady=0, sticky='e')
    flID_3_vals = tk.BooleanVar(config_GUI)
    flID_3_vals.set(False)
    _flID_3_chk = lambda : _flID_chk(flID_3_vals, flID_3_name_option, flID_3_cmin_entry, flID_3_cmax_entry, flID_3_colormap_option)
    flID_3_option = ttk.Checkbutton(config_GUI, takefocus=0, variable=flID_3_vals, offvalue=False, onvalue=True, command=_flID_3_chk)
    flID_3_option.grid(column=1, row=row_track, padx=0, pady=0, sticky='w')
    flID_3_name_vals = tk.StringVar(config_GUI)
    flID_3_name_vals.set('DY634')
    flID_3_name_option = ttk.Combobox(config_GUI, width=10, textvariable=flID_3_name_vals, state='disable')
    flID_3_name_option['values'] = FLUOR_OPTION
    flID_3_name_option.grid(column=2, row=row_track, sticky='e')
    flID_3_cmin_vals = tk.StringVar(config_GUI)
    flID_3_cmin_vals.set('0.1')
    flID_3_cmin_entry = ttk.Entry(config_GUI, width=10, textvariable=flID_3_cmin_vals, state='disable')
    flID_3_cmin_entry.grid(column=3, row=row_track, sticky='w')
    flID_3_cmax_vals = tk.StringVar(config_GUI)
    flID_3_cmax_vals.set('0.9')
    flID_3_cmax_entry = ttk.Entry(config_GUI, width=10, textvariable=flID_3_cmax_vals, state='disable')
    flID_3_cmax_entry.grid(column=4, row=row_track, sticky='w')
    flID_3_colormap_vals = tk.StringVar(config_GUI) 
    flID_3_colormap_vals.set('turbo')
    flID_3_colormap_option = ttk.Combobox(config_GUI, width=10, textvariable=flID_3_colormap_vals, state='disable')
    flID_3_colormap_option['values'] = CMAP_OPTION
    flID_3_colormap_option.grid(column=5, row=row_track, sticky='w')


    ## ##### TERMINATE ##### ##
    row_track += 1
    exit_button = tk.Button(config_GUI, text='Save and Run', font=(ft, 12), height=2, command=lambda : _tkclose(config_GUI))
    exit_button.grid(column=4, columnspan=2, row=row_track, pady=(20, 0), sticky='w')



    ## ################################ EXECUTE ################################ ##
    config_GUI.mainloop()

    flIDs = []
    fluornames = []
    colormaps = []
    clims = []
    flID_vals       = [flID_0_vals, flID_1_vals, flID_2_vals, flID_3_vals]
    flID_names      = [flID_0_name_vals, flID_1_name_vals, flID_2_name_vals, flID_3_name_vals]
    flID_colormaps  = [flID_0_colormap_vals, flID_1_colormap_vals, flID_2_colormap_vals, flID_3_colormap_vals]
    flID_cmins      = [flID_0_cmin_vals, flID_1_cmin_vals, flID_2_cmin_vals, flID_3_cmin_vals]
    flID_cmaxs      = [flID_0_cmax_vals, flID_1_cmax_vals, flID_2_cmax_vals, flID_3_cmax_vals]
    for i, (flID_val, flID_name_val, flID_cmap_val, flID_cmin_val, flID_cmax_val) in enumerate(zip(flID_vals, flID_names, flID_colormaps, flID_cmins, flID_cmaxs)):
        if flID_val.get():
            flIDs.append(i)
            fluornames.append(flID_name_val.get())
            colormaps.append(flID_cmap_val.get())
            clims.append([flID_cmin_val.get(), flID_cmax_val.get()])
    
    config_render = {   'fname_smlm':       loc_fname_vals.get(),
                        'fname_roi':        roi_fname_vals.get(),
                        'roi_orient':       tos_vals.get(),
                        'thickness':        float(roi_thickness_vals.get()),
                        'pxsz_i':           float(roi_pxsz_vals.get()),
                        'pxsz_o':           float(tar_pxsz_vals.get()),
                        'frm_range':        [frm_op_vals.get(), frm_ed_vals.get()],
                        'nosinglet':        nosinglet_vals.get(),
                        'rlim_nm':          float(rlim_nm_vals.get()),
                        'isblur':           isblur_vals.get(),
                        'sig':              float(sig_vals.get()) / float(tar_pxsz_vals.get()),
                        'intensity':        Intensity_vals.get(),
                        'alpha':            float(alpha_vals.get()),
                        'feature':          feature_vals.get(),
                        'zmin_nm':          float(zmin_vals.get()),
                        'zmax_nm':          float(zmax_vals.get()),
                        'colorbits':        colorbits_vals.get(),
                        'flID':             flIDs,
                        'fluorname':        fluornames,
                        'clim':             clims,
                        'colormap':         colormaps   }

    fpath_result = '{}_{}_{}'.format(loc_fname_vals.get()[:-11], os.path.splitext(os.path.basename(roi_fname_vals.get()))[0], tos_vals.get())
    if not os.path.exists(fpath_result):
        os.makedirs(fpath_result)

    jsonObj = json.dumps(config_render, indent=4)
    with open(os.path.join(fpath_result, 'render_settings.json'), "w") as jsfid:
        jsfid.write(jsonObj)

    main_finer_render(config_render, fpath_result)