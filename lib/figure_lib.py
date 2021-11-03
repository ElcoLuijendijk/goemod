import numpy as np
import matplotlib.pyplot as pl


def make_h_and_z_plot(ax, times, x, zs, hs, 
                      add_legend=True, leg_position=(0.5, -0.3), 
                      leg_loc='upper center',
                      xlabel='Distance (m)', ylabel='Elevation (m)', 
                      ls_init='--', time_int=10000,
                      time_int_auto=True, plot_int_depth=True):
    
    """
    
    """
    
    cmap = pl.get_cmap('tab20')
    
    land_surf_color = cmap(10.0 * 1/20.)
    watertable_color = cmap(0.0 * 1/20.)
    past_surf_color = cmap(11.0 * 1/20.)
    past_wt_color = cmap(1.0 * 1/20.)
    init_surf_color = cmap(11.0 * 1/20.)
    init_wt_color = cmap(1.0 * 1/20.)
    
    #land_surf_color = 'tab:brown'
    #watertable_color = 'darkblue'
    #past_surf_color = 'beige'
    #past_wt_color = 'tab:blue'
    #init_surf_color = 'tab:orange'
    #init_wt_color = 'tab:blue'
    
    # line width h and z intermediate timesteps
    lw1 = 0.5
    # line width h and z first and last timesteps
    lw2 = 1.5

    # horizontal position of text labels for h and z
    x_text_1 = 1.15
    x_text_2 = 1.01

    # add labels for h at intermediate timesteps
    # (watertable at stats and end of model run are always shown)
    add_text_labels = True
    
    if len(hs) > 7:
        plot_int = int(len(hs) / 7.0)
    else:
        plot_int = 1
    
    year = 365.25 * 24 * 3600
    
    #plot_times = np.linspace(times[0], times[-1], 9)[1:-1]
    if time_int_auto is True:
        if times[-1] / year < 7000:
            time_int = 1000.0
        elif times[-1] / year <= 110000:
            time_int = 10000.0
        elif times[-1] / year < 200000:
            time_int = 25000
        elif times[-1] / year < 500000:
            time_int = 50000.0
        elif times[-1] / year < 2e6:
            time_int = 100000.0
        else:
            time_int = 500000.0
    
    if plot_int_depth is False:
        plot_times = np.arange(times[0], times[-1] + time_int * year, time_int * year)
        plot_inds = np.array([np.argmin(np.abs(p - times)) for p in plot_times])
    else:
        zmins = np.array([zi.min() for zi in zs])
        zplots = np.linspace(zmins[0], zmins[-1], 10)
        print(zplots)
        plot_inds = np.array([np.argmin(np.abs(zmins - zpi)) for zpi in zplots])
        plot_times = times[plot_inds]
    
    for pi in plot_inds:
        leg_s, = ax.plot(x, zs[pi], color=past_surf_color, lw=lw1)
        leg_h, = ax.plot(x, hs[pi], color=past_wt_color, lw=lw1)

        if plot_int_depth is True and pi != plot_inds[0] and pi != plot_inds[-1]:
            tekst = '%0.0f yr' % (times[pi] / year )
            ax.text(x[-1] * x_text_2, hs[pi][-1], tekst, ha='left', color=past_wt_color)
    
    # annotate for land surface first timestep
    tekst1 = '%0.0f kyr' % (times[0] / year / 1e3)
    #ax.text(x[-1] * x_text_1, zs[0][-1], tekst1, ha='left', va='bottom', color='brown', weight='bold')
    ax.annotate(tekst1, (x[-1], zs[0][-1]), (x[-1] * x_text_1, zs[0][-1]), 
                arrowprops=dict(arrowstyle='->', color=init_surf_color),
                ha='left', va='bottom', color=init_surf_color, weight='bold')
    #arrowprops=dict(edgecolor='brown', facecolor='None', shrink=0.01)
    
    # text for watertable first timestep
    ax.text(x[-1] * x_text_2, hs[0][-1], tekst1, ha='left', va='bottom', color=init_wt_color, weight='bold')

    # annotate land surface last timestep
    tekst2 = '%0.0f kyr' % (times[-1] / year / 1e3)
    #ax.text(x[-1] * x_text_1, zs[-1][-1], tekst2, ha='left', va='top', color='brown', weight='bold')
    ax.annotate(tekst2, (x[-1], zs[-1][-1]), (x[-1] * x_text_1, zs[-1][-1]), 
                arrowprops=dict(arrowstyle='->', color=land_surf_color),
                ha='left', va='top', color=land_surf_color, weight='bold')
    
    # text for watertable last timestep
    ax.text(x[-1] * x_text_2, hs[-1][-1], tekst2, ha='left', va='top', color=watertable_color, weight='bold')

    ax.set_xlim(0, x[-1] * 1.2)
    ax.set_ylim(ax.get_ylim()[0] * 1.25, ax.get_ylim()[1])
    
    # initial land surface and h
    leg_sn0, = ax.plot(x, zs[0], color=init_surf_color, ls=ls_init, lw=lw2)
    leg_hn0, = ax.plot(x, hs[0], color=init_wt_color, ls=ls_init, lw=lw2)  

    # final land surface and h
    leg_sn, = ax.plot(x, zs[-1], color=land_surf_color, lw=lw2)
    leg_hn, = ax.plot(x, hs[-1], color=watertable_color, lw=lw2)  
    
    if add_legend is True:

        #ax.plot(x, hs[0], color='lightblue', lw=lw2)  

        ax.legend([leg_sn0, leg_hn0, leg_s, leg_sn, leg_h, leg_hn], 
                  ['initial land surface', 'initial watertable', 
                   'past land surfaces', 'final land surface', 
                   'past watertables', 'final watertable'], 
                  bbox_to_anchor=leg_position, loc=leg_loc,
                 ncol=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return


def vars_over_time_fig(axs, times, width, n_str_of, Q_baseflows, Q_overland_flows, 
                       erosion_of_per_yr, erosion_bf_per_yr, erosion_hd_per_yr, zs,
                      add_legend=True, add_labels_left=True, add_labels_right=True,
                      log_axis_relief=True):
    
    """
    
    """
    
    hour = 60.0 * 60.0
    day = 24.0 * hour
    year = 365.25 * day
    kyr = 1e3 * year
    
    xlabel = 'Distance (m)'
    ylabel = 'Elevation (m)'

    cmap = pl.get_cmap('tab20')

    land_surf_color = cmap(10.0 * 1/20.)
    watertable_color = cmap(0.0 * 1/20.)

    col_overland = cmap(18.0 * 1/20.)
    col_baseflow = cmap(8.0 * 1/20.)
    col_hillslope = cmap(3.0 * 1/20.)
    #col_overland = 'blue'
    #col_baseflow = 'green'
    #col_hillslope = 'orange'

    axs[0].plot(times / kyr, n_str_of, color='black', label='overland flow')
    
    #bf = np.sum(Q_baseflow_2D) * year
    axs[1].plot(times / kyr, Q_baseflows * year / width, 
                color=col_baseflow)
    axs[1].plot(times / kyr, Q_overland_flows * year / width, 
                color=col_overland)
    

    lo, = axs[2].plot(times / kyr, erosion_of_per_yr / width, 
                      color=col_overland, label='overland\nflow')
    lb, = axs[2].plot(times / kyr, erosion_bf_per_yr / width, 
                      color=col_baseflow, label='baseflow')
    lh, = axs[2].plot(times / kyr, np.abs(erosion_hd_per_yr) / width, 
                      color=col_hillslope, label='hillslope\ndiffusion')
    
    if add_legend is True:
        legs = [lo, lb, lh]
        labels = ['Overland flow', 'Baseflow', 'Hillslope diffusion']
        fig = pl.gcf()
        fig.subplots_adjust(bottom=0.12)
        #axs[2].legend(legs, labels, bbox_to_anchor=(1.02, 1.3), loc='upper left')
        fig.legend(legs, labels, loc='lower center', ncol=3)
    zmn = (zs[0].mean() - np.array([np.mean(zi) for zi in zs]))
    axs[3].plot(times / kyr, zmn, color='black')
    
    axr = axs[3].twinx()
    relief = np.array([np.max(zi) - np.min(zi) for zi in zs])
    axr.plot(times / kyr, relief, color='gray')
    axr.spines['right'].set_color('gray')
    axr.yaxis.label.set_color('gray')
    axr.tick_params(axis='y', colors='gray')

    axs[-1].set_xlabel('Time (kyr)')

    axs[0].spines['top'].set_visible(False)

    for ax in axs[:-1]:
        ax.spines['right'].set_visible(False)

    #fig.savefig('fig/overview_variables_vs_time_%s.pdf' % output_file_adj)

    axs[1].set_yscale('log')
    axs[2].set_yscale('log')
    
    if log_axis_relief is True:
        axs[3].set_yscale('log')
        axr.set_yscale('log')
        
    if add_labels_left is True:
        axs[0].set_ylabel('Number of\nstreams')
        axs[1].set_ylabel(r'Discharge ($m\;a^{-1}$)')
        axs[2].set_ylabel(r'Erosion ($m\;a^{-1}$)')
        axs[3].set_ylabel('Difference\nmean elevation (m)')
        
        axs[0].set_ylim(0, n_str_of.max() * 1.1)
    
    if add_labels_right is True:
        axr.set_ylabel('Relief (m)')
        
    return


def diagnostic_plot(x, zs, dzs):

    ts  = -1

    #print(times[ts] / year)
    a = np.argmin(dzs[ts])
    b = np.argmax(dzs[ts])

    xb = 50

    if xb > a:
        xb = a-1
    if xb > b:
        xb = b - 1

    fig, axs = pl.subplots(2, 2)
    axs[0, 0].plot(x[a-xb:a+xb], zs[ts][a-xb:a+xb])
    axs[0, 1].plot(x[b-xb:b+xb], zs[ts][b-xb:b+xb])
    axs[1, 0].plot(x[a-xb:a+xb], dzs[ts][a-xb:a+xb])
    axs[1, 1].plot(x[b-xb:b+xb], dzs[ts][b-xb:b+xb])
    
    return fig


def find_oscillation(ts, x, zs, dzs):
    
    ds = np.sign(np.diff(dzs[ts]))
    b = list(ds.astype(str))
    #if '1-11,-1] in ds:
    #    print('found oscillation')
    a = ','.join(b)

    print('osicillation found: ', '-1.0,1.0,-1.0,1.0' in a)

    dds = np.diff(ds)
    dsum = np.abs(dds[1:]) + np.abs(dds[:-1])
    ddsum = dsum[1:] + dsum[:-1]
    dddsum = ddsum[1:] + ddsum[:-1]
    print('max sum = ', np.max(dddsum))

    o = oscill_loc = np.argmax(dddsum)

    xb = 50

    #if xb > a:
    #    xb = a-1
    #if xb > b:
    #    xb = b - 1

        
    if xb > o:
        xb = o-1

    fig, axs = pl.subplots(2, 1)

    axs[0].plot(x[o-xb:o+xb], zs[ts][o-xb:o+xb])
    axs[1].plot(x[o-xb:o+xb], dzs[ts][o-xb:o+xb])
    
    return fig