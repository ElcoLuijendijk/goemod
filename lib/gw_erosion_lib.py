import itertools
import string
import datetime
import pdb
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

from IPython.core.debugger import set_trace


def find_gev_dist_params(D):
    
    """
    eqs. 8, 9 and 10 in Stowa (2019)
    """
    gev_loc = 1.02 * (0.239 - 0.0250 * np.log10(D/60.))**(-1/0.512)
    gev_disp = 0.478 - 0.0681 * np.log10(D)
    gev_shape = 0.118 - 0.266 *np.log10(D) + 0.0586 * (np.log10(D)**2)
    
    return gev_shape, gev_loc, gev_disp


def percipitation_return_time_func(gev_shape, gev_loc, gev_disp, return_time):
    
    """
    eq. 3 in Stowa (2019)
    
    """
    
    T = return_time
    
    kappa = gev_shape
    eta = gev_loc
    gamma = gev_disp
    
    if type(kappa) is float:
        if kappa != 0.0:
            x_gev = eta * (1 + gamma/kappa *  (1- T**(-kappa)))
        else:
            x_gev = eta * (1 + gamma * np.log(T))
    
    else:
        x_gev = eta * (1.0 + gamma/kappa *  (1.0- T**(-kappa)))
        x_gev0 = eta * (1.0 + gamma * np.log(T))
        x_gev[kappa == 0.0] = x_gev0[kappa == 0.0]
        
    return x_gev


def calculate_precip_events(P, precipitation_duration):
    
    """
    
    note, precipitation frequency curve is hardcoded
    """
    
    hour = 3600.
    day = 24 * hour
    year = 365.25 * day
    #precipitation_duration = 1.0 * day
    max_precip_freq = 30
    precip_freqs = np.arange(1, max_precip_freq)

    Ts = 1./precip_freqs

    gev_shape, gev_loc, gev_disp = find_gev_dist_params(precipitation_duration)

    precip_depths = percipitation_return_time_func(gev_shape, gev_loc, gev_disp, Ts) * 1e-3

    # find cumulative total of all precipitation events
    precip_totals = precip_depths * precip_freqs
    precip_total = np.sum(precip_depths * precip_freqs)

    precip_depth_cum = np.cumsum(precip_totals)
    
    # find n-th precip event where total P exceeds the target P per year
    ind = np.where(precip_depth_cum > (P * year))[0][0]
    
    # select P events and calculate remaining missing precipitation for target P
    remaining_P = P * year - precip_depth_cum[ind-1]
    
    # reduce frequency of last preciptiation event to get up to the total P
    if remaining_P > 0:
        precip_freqs[ind] = precip_freqs[ind] * remaining_P / precip_totals[ind]
        ind += 1
        precip_totals = precip_depths * precip_freqs
        precip_total = np.sum(precip_depths * precip_freqs)

        precip_depth_cum = np.cumsum(precip_totals)
            
    print('including first %i %0.0f hr precipitation events to arrive at total P of %0.2f m / yr with a target of %0.2f m/yr' 
          % (ind, precipitation_duration/hour, precip_depth_cum[ind-1], P * year))

    precip_depths = precip_depths[:ind]
    precip_freqs = precip_freqs[:ind]

    # find cumulative total
    precip_totals = precip_depths * precip_freqs
    precip_total = np.sum(precip_depths * precip_freqs)
    
    return precip_depths, precip_freqs, precip_totals, precip_total


def analytical_solution_h_one_side(rch, T, L, x):
    
    """
    analytical solution steady-state depth-integrated groundwater flow
    in system with uniform recharge and bound by a stream on one side and a groundwater divide on the other
    
    """
    
    dh = rch / T * (L * x - 0.5 * x**2)
    
    return dh


def analytical_solution_h(rch, T, L, x, dH):
    
    """
    analytical solution steady-state depth-integrated groundwater flow
    in system with uniform recharge and bound by a streams on two sides
    
    """
    
    dh = 0.5 * rch/T * x * (L-x) + dH * x /L
    
    return dh
    

def calculate_h(x, z, T, rch, seepage_pts, seepage_xs, seepage_hs):

    """
    use analytical solutions to calculate steady-state hydraulic head 

    """
    
    h = np.ones_like(x)

    q_gw_rch = [0] * len(seepage_pts)

    for xli, xl, hl, xri, xr, hr in zip(seepage_pts[:-1], seepage_xs[:-1], seepage_hs[:-1],
                                        seepage_pts[1:], seepage_xs[1:], seepage_hs[1:]):
        

        dH = hr -  hl
        L = xr - xl
        xloc = x[xli:xri] - x[xli]
        
        if type(rch) is np.float64 or type(rch) is float:
            rch_avg = rch
        else:
            rch_avg = rch[xli:xri].mean()
        
        hloc = analytical_solution_h(rch_avg, T, L, xloc, dH)
        h[xli:xri] = hloc + hl

    # add h for left-hand side
    if seepage_pts[0] > 0:
        xl = x[0]
        xli = 0
        xr = x[seepage_pts[0]]
        xri = seepage_pts[0]
        L = xr - xl
        xloc = L - x[xli:xri]
        
        if type(rch) is np.float64 or type(rch) is float:
            rch_avg = rch
        else:
            rch_avg = rch[xli:xri].mean()
            
        hloc = analytical_solution_h_one_side(rch_avg, T, L, xloc)
        h[xli:xri] = hloc + seepage_hs[0]

    # add h for right-hand side
    if seepage_pts[-1] < (len(x) - 1):
        xl = seepage_xs[-1]
        xli = seepage_pts[-1]
        xr = x[-1]
        xri = len(x)
        xloc = x[xli:xri] - x[xli]
        L = xr - xl
        
        if type(rch) is np.float64 or type(rch) is float:
            rch_avg = rch
        else:
            rch_avg = rch[xli:xri].mean()
            
        hloc = analytical_solution_h_one_side(rch_avg, T, L, xloc)
        h[xli:xri] = hloc + seepage_hs[-1]
    
    h[seepage_pts] = z[seepage_pts]
    
    return h


def find_next_seepage_pt(h, z, seepage_bool, seepage_pts):
    
    """
    
    """
    
    a = len(seepage_pts)
    
    zc = z.copy()

    # find lowest point in model domain where h > z
    zc[seepage_bool == True] = 99999
    zc[h < z] = 99999
    
    if len(seepage_pts) > 0:
        if len(seepage_pts) > 1:
            for sp1, sp2 in zip(seepage_pts[:-1], seepage_pts[1:]):
                
                if sp2 - sp1 > 1:
                    ind_zmin = sp1 + np.argmin(zc[sp1:sp2])
                    if zc[ind_zmin] != 99999:
                        seepage_pts.append(ind_zmin)
        
        if seepage_pts[0] > 0:
            ind_zmin = np.argmin(zc[:seepage_pts[0]])
            if zc[ind_zmin] != 99999:
                seepage_pts.append(ind_zmin)
        
        if seepage_pts[-1] < (len(z) - 1):
            ind_zmin = seepage_pts[-1] + np.argmin(zc[seepage_pts[-1]:])
            if zc[ind_zmin] != 99999:
                seepage_pts.append(ind_zmin)
    
    else:
        ind_zmin = np.argmin(zc)
        seepage_pts.append(ind_zmin)
    
    # remove duplicates
    seepage_pts = list(set(seepage_pts))
    
    # sort streams
    seepage_pts.sort()
    
    if len(seepage_pts) == a:
        print('warning, no improvement in seepage iterations')
        #print('seepage pts: ', seepage_pts)
        #raise ValueError
    
    return seepage_pts


def remove_inflow_seepage_nodes(seepage_pts, x, z, h, T, inflow_lim):
    
    a = len(seepage_pts)
    
    # check if any seepage nodes cause inflow
    dh = np.diff(h)
    dxi = np.diff(x)
    q = -T * dh / dxi
    dq = q[1:] - q[:-1]
    #print(dq.min() * year, dq.max() * year

    dqn = np.zeros_like(z)
    dqn[1:-1] = dq
    rel_q = dqn / inflow_lim

    remove_seepage_nodes = np.where(rel_q > 1)[0]
    
    seepage_pts_mod = seepage_pts
    
    for s in seepage_pts_mod:
        if s in remove_seepage_nodes:
            #print('remove node %i from seepage list' %s)
            seepage_pts_mod.remove(s)
    ns = len(seepage_pts_mod)
    nr = ns - a
    #print('number of seepage pts remaining = %i' % ns)
    #print('%i nodes were removed because they generated more inflow than the recharge rate' % (a - ns))
    
    return seepage_pts_mod, nr


def calculate_h_and_seepage(x, z, T, rch, max_h_error=0.05, return_all=False, verbose=False):
    
    """
    
    """
    
    relief = z.max() - z.min()
    max_h_error = 0.05 * relief
    if max_h_error > 0.05:
        max_h_error = 0.05
    
    # calculate width of cells
    dxc = np.diff(x)

    # calculate width between nodes
    dxn = np.zeros_like(x)
    dxn[1:] += dxc / 2.0
    dxn[:-1] += dxc / 2.0
    
    # set limit for acceptable inflow from seepage nodes
    if type(rch) is float:
        inflow_lim = rch * dxn * 2
    else:
        inflow_lim = rch.mean() * dxn * 2
    
    h = np.ones_like(z) * z.min()
    
    seepage_pts = []
    #seepage_pts = stream_pts.tolist()
    
    max_iter = int(len(x) / 1)
    keep_going = True
    h_converged = False

    n_iter = 0
    
    hs = []
    seepage_pts_all = []
    seepage_bool = np.full((len(x)), False)
    
    while keep_going is True and n_iter < max_iter:
        
        if verbose is True:
            print('iteration %i, seepage pts = %i' % (n_iter, len(seepage_pts)))
        
        if len(seepage_pts) > 0:
        
            seepage_pts_corr, nr = remove_inflow_seepage_nodes(seepage_pts, x, z, h, T, inflow_lim)

            #if nr < -1:
            seepage_pts = seepage_pts_corr
            seepage_xs =  x[seepage_pts]
            seepage_hs = z[seepage_pts]

            h = calculate_h(x, z, T, rch, seepage_pts, seepage_xs, seepage_hs)
            
        seepage_pts = find_next_seepage_pt(h, z, seepage_bool, seepage_pts)

        seepage_xs =  x[seepage_pts]
        seepage_hs = z[seepage_pts]
        seepage_bool[np.array(seepage_pts)] = True
        
        h = calculate_h(x, z, T, rch, seepage_pts, seepage_xs, seepage_hs)
                
        # check if h is below the surface everywhere
        if np.any(h > (z + max_h_error)) == False:
            keep_going = False
            h_converged = True

        n_iter += 1
        
        if return_all is True:
            hs.append(h)
            seepage_pts_all.append(seepage_pts.copy())
    
    # correct values of h to avoid issues with precipitation excess calculation
    h[h > z] = z[h > z]
    
    if h_converged is True:
        if verbose is True:
            print('converged after %i iterations' % n_iter)
    else:
        print('warning, convergence h calculation failed after %i iterations' % n_iter)
    
    if return_all is True:
        return h, seepage_pts, hs, seepage_pts_all
    else:
        return h, seepage_pts
    

def find_depressions(z):
    
    """
    Find depressions in a 1D array with elevation values 
    
    z : elevation
    
    
    returns
    z_depr : booelan array with True for depressions
    """
    
    
    ## find depressions
    dz = np.diff(z)
    z_depr = np.zeros_like(z, dtype=bool)
    potential_stream_pts = np.where(np.sign(dz[1:]) > np.sign(dz[:-1]) + 1)[0] + 1
    
    #ddz = np.diff(z, n=2)
    #potential_stream_pts = np.where(ddz > 0)[0] + 1 
    
    z_depr[potential_stream_pts] = True
    
    # add node on left-hand and right-hand sides
    if dz[0] > 0:
        z_depr[0] = True
    if dz[-1] < 0:
        z_depr[-1] = True
        
    return z_depr


def calculate_contributing_area(x, u, stream_pts):

    """
    calculate the area contributing gw flow to each stream

    """


    contributing_area = np.zeros(len(stream_pts))
    contributing_area_pts = []
    for i, stream_pt in enumerate(stream_pts):
        if stream_pt == stream_pts[0]:
            contributing_area_left = x[stream_pt]
            contributing_area_pt1 = 0
        else:
            last_stream_pt = stream_pts[i-1]
            divide_loc = last_stream_pt + np.argmax(u[last_stream_pt:stream_pt])
            contributing_area_left = x[stream_pt] - x[divide_loc]
            contributing_area_pt1 = divide_loc
            
        if stream_pt == stream_pts[-1]:
            contributing_area_right = x.max() - x[stream_pt]
            contributing_area_pt2 = -1
        else:
            next_stream_pt = stream_pts[i+1]
            divide_loc = stream_pt + np.argmax(u[stream_pt:next_stream_pt])
            contributing_area_right = x[divide_loc] - x[stream_pt]
            contributing_area_pt2 = divide_loc

        contributing_area[i] = contributing_area_left + contributing_area_right
        
        contributing_area_pts.append([contributing_area_pt1, contributing_area_pt2])
        
    return contributing_area_pts, contributing_area


def calculate_baseflow(x, h, T, stream_pts_bf):
    
    """
    calculate steady-state baseflow to streams 
    using data on hydraulic head, transmissivity and stream locations
    """
    
    #Q_baseflow_2D_old = rch * contributing_area_bf

    dhi = h[1:] - h[:-1]
    dxi = x[1:] - x[:-1]

    q = - T * dhi / dxi

    dq = (q[1:] - q[:-1])

    dQ = np.zeros_like(x)
    dQ[1:-1] = dq
    dQ[0] = dq[0]
    dQ[-1] = -dq[-1]

    Q_baseflow_2D = -dQ[stream_pts_bf]
    
    # take out stream pts with negative baseflow (losing streams)
    Q_baseflow_2D[Q_baseflow_2D <= 0] = 1e-12
    
    return Q_baseflow_2D


def calculate_excess_precipitation(P_event, h, z, specific_yield, dxc, dx):
    
    """
    calculate saturation excess using data on elevation, watertable depth, specific yield and precipitation
    
    returns:
        Pe: array
            excess precipitation / saturation overland flow generated per event, per grid cell (m2)
    """
    

    # calculate temporary watertable
    he = h + P_event / specific_yield

    # calculate precipitation excess above storage
    Pe_nodes = he - z
    Pe = (Pe_nodes[1:] + Pe_nodes[:-1]) / 2.0 * dxc * specific_yield

    Pe[Pe < 0] = 0

    # for each cell where watertable cross surface find intersection he and z
    dhe = np.diff(he)
    dze = np.diff(z)
    he_slope = dhe / dxc
    z_slope = dze / dxc

    xi = (z[:-1] - he[:-1]) / (he_slope - z_slope)
    in_cell = np.logical_and(xi > 0, xi < dx)

    ind_int = np.where(in_cell)[0]
    if np.any(in_cell):
        for i in ind_int:
            x1, x2 = i, i+1
            if he[x1] > z[x1] and he[x2] > z[x2]:
                raise ValueError
            if he[x1] > z[x1]:
                Pe[i] = (he[x1] - z[x1]) * xi[i] * specific_yield 
            elif he[x2] > z[x2]:
                Pe[i] = (he[x2] - z[x2]) * (dxc[i] - xi[i]) * specific_yield 
    
    if Pe.min() < 0:
        raise ValueError
    
    return Pe


def calculate_overland_flow_per_stream(V_overland_2D_per_cell, stream_pts_of, contributing_area_pts_of):
    
    """
    calculate overland flow per stream using data on overland flow generated per cell,
    stream locations and contributing areas
    
    returns:
    
        V_overland_2D_per_stream : array (n_stream_nodes)
            2D volume of overland flow per stream node

    """
    
    V_overland_2D_per_stream = np.zeros(len(stream_pts_of))
    
    for pt_i, stream_pt_i, contributing_area_pt in zip(itertools.count(), 
                                                       stream_pts_of, 
                                                       contributing_area_pts_of):
        x1, x2 = contributing_area_pt
        V_overland_2D_per_stream[pt_i] = np.sum(V_overland_2D_per_cell[x1:x2])
   
    
    return V_overland_2D_per_stream


def calculate_overland_flow_volume(P_event, max_infiltration, dxc, dx, z, h, specific_yield):
    
    """
    calculate infiltration-excess and saturation-excess overland flow
    for a single precipitation event
    
    
    returns:
    
    V_overland_2D_per_node:
        2D volume of overland flow per stream node
    
    
    """
    

    if P_event > max_infiltration:
        n_cells = len(h) - 1
        infiltration_excess = np.ones(n_cells) * (P_event - max_infiltration) * dxc
        infiltrated_P = max_infiltration
    else:
        infiltration_excess = 0.0
        infiltrated_P = P_event
        
    saturation_excess = calculate_excess_precipitation(infiltrated_P, h, z, specific_yield, dxc, dx)
    
    V_overland_2D_per_cell = infiltration_excess + saturation_excess

    #V_overland_2D = np.zeros(len(stream_pts_of))

    #for pt_i, stream_pt_i, contributing_area_pt in zip(itertools.count(), 
    #                                                   stream_pts_of, 
    #                                                   contributing_area_pts_of):
    #    x1, x2 = contributing_area_pt
    #    V_overland_2D[pt_i] = np.sum(P_excess[x1:x2])
        
    return V_overland_2D_per_cell


def calculate_sediment_flux_overland_flow(V0, L, S, St, kf, Km, n, m):
    
    """
    integrated sediment flux over time for a single overland flow event in a stream channel:
    
    V0 : initial volume of water to be discharged (m3)
    L : Length of channel
    S : Slope of channel bed along flow direction
    St : Slope of channel bed perpendicular to flow direction, assuming a triangular channel shape
    kf : empirical coefficient in sediment discharge equation (units...)
    Km : empirical coefficient in the Gauckler-Manning equation (units....)
    n : empirical coefficient in sediment discharge equation (units...)
    m : empirical coefficient in sediment discharge equation (units...)
    """
    
    t0 = np.inf
    t1 = 0.0
    
    # calculate constants in equation:
    a = kf * S**n * (Km *S**0.5 / St)**m
    b = ((V0 * St) / L) ** (-1./3.)
    c = (Km * S**0.5) / (3 * L)
    
    Vsa = a * (b + c * t0) ** (-4*m+1) / (-4*m*c + c) - a * (b + c * t1) ** (-4*m+1) / (-4*m*c + c)
    
    return Vsa


def hillslope_diffusion_old(K_d, z, z_gradient):

    """
    calculate hillslope diffusion flux

    :param K_d:
    :param z:
    :param z_gradient:
    :return:
    """

    q_sh = -K_d * z_gradient

    return q_sh


def hillslope_diffusion(x, z, K_d, dx, dt):

    """
    calculate hillslope diffusion
    """
    
    z_gradient = np.diff(z) / np.diff(x)

    # calculate flux
    q_hillslope = -K_d * z_gradient
    
    #dz_hillslope[:] = 0.0
    # calculate hillslope elevation change
    dz_hillslope = np.zeros_like(z)
    dz_hillslope[1:] -= - dt * q_hillslope / dx
    dz_hillslope[:-1] += - dt * q_hillslope / dx

    return dz_hillslope


def construct_diffusion_matrix_variable_x(
        u, x, dt, K, c,
        fixed_u_left,
        fixed_u_right):

    """
    set up matrix for finite difference eqs.
    """

    nx = len(u)

    # create a diagonal matrix
    A = np.diagflat(np.ones(nx))

    # create the matrix coefficient
    s = np.zeros(nx)
    s[1:-1] = 2.0 / (x[2:] - x[:-2])
    s[0] = 1.0 / (x[1] - x[0])
    s[-1] = 1.0 / (x[-1] - x[-2])

    t = np.zeros(nx)
    t[:-1] = K / (x[1:] - x[:-1])

    w = np.zeros(nx)
    w[1:] = K / (x[1:] - x[:-1])

    v = dt / c

    # fill matrix
    nx = len(u)
    for i in range(1, nx-1):
        A[i, i] = 1.0 + s[i] * t[i] * v[i] + s[i] * w[i] * v[i]
        A[i, i-1] = -s[i] * w[i] * v[i]
        A[i, i+1] = -s[i] * t[i] * v[i]

    # boundary condition left-hand side
    if fixed_u_left is not None:
        A[0, 0] = 1.0
        A[0, 1] = 0.0
    else:
        A[0, 0] = 1.0 + s[0] * t[0] * v[0]
        A[0, 1] = -s[0] * t[0] * v[0]
    
    # boundary condition right-hand side
    if fixed_u_right is not None:
        A[-1, -1] = 1.0
        A[-1, -2] = 0.0
    else:
        #lower bnd
        A[-1, -1] = 1.0 + s[-1] * w[-1] * v[-1]
        A[-1, -2] = -s[-1] * w[-1] * v[-1]

    return A


def create_diffusion_eq_vector_variable_x(
        u, Q, dt, c, dx_left, dx_right,
        left_bnd_flux,
        right_bnd_flux,
        fixed_u_left,
        fixed_u_right):

    """
    create vector for right hand side implicit FD equation
    """

    # create vector
    b = np.zeros_like(u)

    # fill with values
    b[:] = u[:] + Q * dt / c

    # upper bnd
    if fixed_u_left is not None:
        b[0] = fixed_u_left
    elif left_bnd_flux is not None:
        b[0] = u[0] + (Q[0] + left_bnd_flux / dx_left) * dt / c[0]

    # lower bnd
    if fixed_u_right is not None:
        b[-1] = fixed_u_right
    elif right_bnd_flux is not None:
        b[-1] = u[-1] + (Q[-1] + right_bnd_flux / dx_right) * dt / c[-1]

    return b


def solve_1D_diffusion(u, x, dt, K, c, Q,
                       left_bnd_flux,
                       right_bnd_flux,
                       fixed_u_left,
                       fixed_u_right,
                       A=None,
                       verbose=False,
                       return_matrix_too=False):

    """
    solve 1D diffusion equation, with variable diffusion coefficient (K) and node distances

    """

    #c = np.ones_like(phi)
    #Q = np.zeros_like(phi)

    if A is None:

        A = construct_diffusion_matrix_variable_x(
            u, x, dt, K, c,
            fixed_u_left,
            fixed_u_right)

    nx = len(u)

    dx_left = x[1] - x[0]
    dx_right = x[-1] - x[-2]

    b = create_diffusion_eq_vector_variable_x(
        u, Q, dt, c, dx_left, dx_right,
        left_bnd_flux,
        right_bnd_flux,
        fixed_u_left,
        fixed_u_right)

    # use numpy linalg to solve system of eqs.:
    # uses  LAPACK routine _gesv:
    # https://software.intel.com/sites/products/documentation/hpc/mkl/mklman/\
    # GUID-4CC4330E-D9A9-4923-A2CE-F60D69EDCAA8.htm
    # uses LU decomposition and iterative solver
    try:
        u_new = np.linalg.solve(A, b)
    except:
        msg = 'error, solving matrix for diffusion eq. failed'
        raise ValueError(msg)

    # TODO check other linear solvers, such as CG:
    # http://docs.scipy.org/doc/scipy-0.13.0/reference/generated/
    # scipy.sparse.linalg.cg.html
    # and use sparse matrices to conserve memory
    # (not really necessary in 1D case)

    # check that solution is correct:
    check = np.allclose(np.dot(A, u_new), b)

    if verbose is True:
        print('solution is correct = ', check)

    if check is False:
        msg = 'warning, solution is ', check
        raise ValueError(msg)
    
    if return_matrix_too is True:
        return u_new, A
    else:
        return u_new

    
def model_gwflow_and_erosion(width, dx, 
                             total_runtime, dt, 
                             min_rel_dz, max_rel_dz, min_abs_dz,
                             fixed_upstream_length, upstream_length, upstream_aspect_ratio,
                             downstream_length,
                             P, precipitation_duration, precip_return_time, infiltration_capacity, ET, 
                             initial_relief, n_initial_topo_segments, init_z,
                             T, specific_yield, phi,
                             S_init, St, 
                             k_f, K_n, n, m, k_w , omega, K_d,
                             recalculate_slope, U,
                             enforce_CFL_condition=True,
                             max_CFL_number=0.5,
                             average_rch=True,
                             explicit_diffusion_solver=True,
                             variable_dt=False,
                             max_dt=500 * 365.25 * 24 * 3600.,
                             include_stream_erosion_in_hillslope_diff=False,
                             remove_out_of_plane_gw=True,
                             use_relief_from_file=False,
                             relief_input_file='default_topography.csv',
                             save_initial_topography=False, initial_topography_file='initial_topography.csv',
                             save_final_topography=False, final_topography_file='final_topography.csv',
                             debug=False):
    
    """
    
    """
    
    report_interval = 200
    
    year = 365.25 * 24 * 3600.
    
    if use_relief_from_file is True:
        # read csv file with initial topography
        dfr = pd.read_csv(relief_input_file)
        x = dfr['x'].values
        z_init = dfr['z'].values
        width = x.max() - x.min()
        nx = len(x)
        dx = np.diff(x)[0]
    else:
        ## Set up grid
        nx = int(width / dx)
        x = np.linspace(0, width, nx)
    
    xc = (x[1:] + x[:-1]) / 2.0

    # calculate width of cells
    dxc = np.diff(x)

    # calculate width between nodes
    dxn = np.zeros_like(x)
    dxn[1:] += dxc / 2.0
    dxn[:-1] += dxc / 2.0

    if use_relief_from_file is False:  
        ## generate initial topography
        print('generating initial topography with max relief of %0.2f m and %i linear segments' 
              % (initial_relief, n_initial_topo_segments))

        xx = np.linspace(0, nx*dx, n_initial_topo_segments)

        z_depressions = (np.random.random(n_initial_topo_segments) - 0.5) * initial_relief + init_z
        #z_depressions = np.random.randn(n_z_perturbations) * relief / 2.0 + init_z

        z_init = np.interp(x, xx, z_depressions)

        
    if save_initial_topography is True:
        dfr = pd.DataFrame(index=x, columns=['z'])
        dfr['z'] = z_init
        print(f'saving relief initial timestep to {initial_topography_file}')
        dfr.to_csv(initial_topography_file, index_label='x')

    z = z_init.copy()
    
    ## set up rainfall depths array
    precip_depths, precip_freqs, precip_totals, precip_total = \
        calculate_precip_events(P, precipitation_duration)

    ##########
    # set up arrays for loop

    seepage_bool = np.zeros_like(z, dtype=bool)

    dz_bf = np.zeros_like(z)
    dz_of = np.zeros_like(z)
    dz_hillslope = np.zeros_like(z)

    zs = []
    hs = []
    times = []
    streams_bf_all = []
    streams_of_all = []

    time = 0
    ts_report = 0

    ti = 0

    Pes = []
    Vos = []

    # stream locations for overland & baseflow
    strs_of = []
    strs_bf = []

    # number of active streams
    n_str_of = []
    n_str_bf = []

    # upstream length
    uls_of = []
    uls_bf = []

    # contributing areas
    cas_bf = []
    cas_of = []

    # baseflow and overland flow fluxes
    Q_baseflows = []
    Q_overland_flows = []
    Q_baseflow_nodes_i = np.zeros_like(x)
    Q_overland_flow_nodes_i = np.zeros_like(x)
    Q_baseflow_nodes = []
    Q_overland_flow_nodes = []

    # erosion overland flow, baseflow, hillslope diff
    erosion_bf_per_yr = []
    erosion_of_per_yr = []
    erosion_hd_per_yr = []

    dzs = []
    
    ETs = []

    n_str_of_i = 0
    
    max_infiltration_depth = infiltration_capacity * precipitation_duration
    print(f'max infiltration depth = {max_infiltration_depth}')

    print('starting model runs')

    # initial h and rch
    rch_init = P - max_infiltration_depth - ET
    if rch_init < 0:
        rch_init = 1e-15
    h, seepage_pts = calculate_h_and_seepage(x, z, T, rch_init, return_all=False)
    
    while time < total_runtime:

        if ti < 20 or ts_report >= report_interval:
            print('timestep %i, time = %0.1f yrs of %0.0f yrs, active streams = %i, min z = %0.2f' 
                  % (ti, time/year, total_runtime/year, n_str_of_i, z.min()))
            ts_report = 0

        if ti >= 20:
            report_interval = 10
        if ti >= 100:
            report_interval = 100
        if ti >= 500:
            report_interval = 500

        if np.any(np.isnan(z)):
            raise ValueError('error, nan value in elevation array z')

        ## Calculate stream locs for overland flow
        z_depr = find_depressions(z)
        stream_pts_of = np.where(z_depr)[0]    

        ## calc overland flow
        contributing_area_pts_of, contributing_area_of = calculate_contributing_area(x, z, stream_pts_of)
        
        #of_results = [calculate_overland_flow_volume(precip_depth, max_infiltration_depth, 
        #                                             dxc, dx, z, h, specific_yield, 
        #                                             stream_pts_of, contributing_area_pts_of) 
        #                      for precip_depth in precip_depths]
        
        # overland flow volume per event, per cell
        P_excess = np.array([calculate_overland_flow_volume(precip_depth, max_infiltration_depth, 
                                                    dxc, dx, z, h, specific_yield) 
                              for precip_depth in precip_depths])
        
        
        # overland flow volume per event, per stream
        # note, this is the value used later on to calculate erosion by overland flow
        #V_of_2D = np.array([a for a, b in of_results])
        V_of_2D = np.array([calculate_overland_flow_per_stream(Pei, stream_pts_of, contributing_area_pts_of) 
                            for Pei in P_excess])
        
        # multiply with precipitation frequencies
        V_of_2D_all = np.array([Vi * f for Vi, f in zip(V_of_2D, precip_freqs)])
        # overland flow discharge per time period (=usuallly per year)
        Q_of_2D = np.sum(V_of_2D_all, axis=0) / precip_return_time
        
        # total overland flow per year
        Q_overland_flow_sum = np.array([Pei * f for Pei, f in zip(P_excess, precip_freqs)])
        Q_overland_flow = np.sum(Q_overland_flow_sum) / precip_return_time

        Q_overland_flow_nodes_i[:] = 0
        Q_overland_flow_nodes_i[stream_pts_of] = Q_of_2D 

        ## calculate partitioning overland flow and recharge
        #pot_rch_per_event = np.array([Pdi * pfi - (Poi/dxc) 
        #                            for Pdi, pfi, Poi in zip(precip_depths, precip_freqs, P_excess)])
        pot_rch_per_event = np.array([(Pdi - (Pei/dxc)) * pfi  
                                    for Pdi, pfi, Pei in zip(precip_depths, precip_freqs, P_excess)])
        
        if debug is True and time == 0:
            
            print('precip depth, precip freq, overland flow, potential rch')
            for Pdi, pfi, Poi, prch in zip(precip_depths, precip_freqs, P_excess, pot_rch_per_event):
                
                print(f'{Pdi:0.2e}, {pfi:0.0f}, {np.mean(np.array(Poi / dxc)):0.2e}, {np.mean(np.array(prch)):0.2e}')
        
        # sum all recharge events to get a total potential recharge per year
        # note this is a 1D value of pot recharge per cell
        pot_rch = np.sum(pot_rch_per_event, axis=0)
        
        if debug is True:
            # check waterbalance
            P_total = np.sum([Pdi * Pfi for Pdi, Pfi in zip(precip_depths, precip_freqs)])
            P_excess_total = np.sum(np.array([Pei / dxc * Pfi for Pei, Pfi in zip(P_excess, precip_freqs)]), axis=0)

            wb_err = P_total - P_excess_total - pot_rch
            wb_err_rel = np.abs(wb_err / P_total) * 100.0

            if np.max(wb_err_rel) > 1.0:

                print(f'warning, max cell waterbalance error = {wb_err_rel:0.3f} %')

                set_trace()

        # calculate ET rate from potential rch
        actual_ET = pot_rch.copy()
        actual_ET[pot_rch > (ET * year)] = ET * year
        
        # substract actual ET from recharge nodes
        rch_all_cells_per_year = pot_rch - actual_ET 
        rch_all_cells = rch_all_cells_per_year / year

        # make sure no negative rch values
        rch_all_cells[rch_all_cells < 0] = 0.0
        
        if average_rch is True:
            rch = rch_all_cells.mean()
        else:
            rch = rch_all_cells
            rch[rch < 0] = 0
        
        if debug is True:
            # another wb check
            if average_rch is True:
                wb_err = np.sum(pot_rch * dxc) - rch * width * year - np.sum(actual_ET * dxc)
            else:
                wb_err = np.sum(pot_rch * dxc) - np.sum(rch * dxc * year) - np.sum(actual_ET * dxc)

            wb_err_rel = np.abs(wb_err / np.sum(pot_rch * dxc)) * 100.0

            if np.max(wb_err_rel) > 1.0:
                print(f'warning, max cell waterbalance error = {np.max(wb_err_rel):0.3f} %')
                
                set_trace()
                

        if remove_out_of_plane_gw is True:
            
            # calculating out of plane groundwater flow 
            # note this does not work with spatially distributed recharge yet, only with averaged value
            
            if time == 0:
                S_min = S_init
            else:
                # pick lowest stream gradient
                S_min = np.min(S_bf)
            
            # gw flow leaving the model domain (m3/sec)
            gw_flow_out = T * S_min * width

            if average_rch is True:
                rch_vol = rch * width * upstream_length
            else:
                rch_vol = rch * dxc * upstream_length

            #
            net_rch = rch_vol - gw_flow_out
            
            rch_old = rch
            
            if average_rch is True:
                rch = net_rch / (width * upstream_length)
                gw_loss = rch_old - rch
            else:
                rch = net_rch / (dxc * upstream_length)
                gw_loss = np.sum(rch_old - rch)
            
            if debug is True and time == 0:
                print('gw flow out of the model domain = %0.2e m3/yr' % (gw_flow_out * year))
                print('total rch volume in model domain = %0.2e m3/yr' % (rch_vol * year))
                print('original rch in model domain = %0.2e m/yr' % (rch_init * year))
                print('net rch in model domain = %0.2e m/yr' % (rch * year))

            if rch <= 0:
                if debug is True:
                    msg = 'Warning: Negative effective in-plane recharge due to high rate' 
                    msg += ' of out-of-plane groundwater flow.\n'
                    msg += 'Reduce transmissivity of stream slope to limit the amount of out-of plane groundwater flow.'
                    #raise ValueError(msg)
                    msg += '\nsetting in-plane rch to zero'
                    print(msg)
                rch = 0.0
                
        ## Calculate h and seepage pts
        h, seepage_pts = calculate_h_and_seepage(x, z, T, rch, return_all=False)
        #print('done calculating h')

        seepage_bool[:] = False
        seepage_bool[seepage_pts] = True
        stream_pts_bf = np.where(np.logical_and(seepage_bool, z_depr))[0]

        ## calc contributing area and baseflow
        contributing_area_pts_bf, contributing_area_bf = calculate_contributing_area(x, h, stream_pts_bf)
        #Q_baseflow_2D = rch * contributing_area_bf
        Q_baseflow_2D = calculate_baseflow(x, h, T, stream_pts_bf)
        
        # copy calculated baseflow to streams to nodes
        Q_baseflow_nodes_i[:] = 0
        Q_baseflow_nodes_i[stream_pts_bf] = Q_baseflow_2D

        if fixed_upstream_length is True:
            upstream_length_bf = np.ones(len(contributing_area_bf)) * upstream_length
        else:
            upstream_length_bf = contributing_area_bf * upstream_aspect_ratio
            upstream_length_bf[upstream_length_bf == 0.0] = dx
        
        #
        baseflow = Q_baseflow_2D * upstream_length_bf

        # store data for baseflow:
        uls_bf.append(upstream_length_bf)
        cas_bf.append(contributing_area_pts_bf)
        n_str_bf.append(len(stream_pts_bf))
        strs_bf.append(stream_pts_bf)
        Q_baseflows.append(np.sum(Q_baseflow_2D))
        Q_baseflow_nodes.append(Q_baseflow_nodes_i.copy())
        
        if recalculate_slope is True:
            ## recalculate stream slope for baseflow streams
            #z_bf = z[stream_pts_bf]
            #z_origin_bf = upstream_length_bf * S_init
            #S_bf = (z_origin_bf - z_bf) / upstream_length_bf
            
            # downstream elevation
            # downstream length is assumed to equal upstream length
            z_downstream_init = -downstream_length * S_init
            
            # adjust for baselevel change
            z_downstream = z_downstream_init + U * time
            
            #
            z_bf = z[stream_pts_bf]
            S_bf = (z_bf - z_downstream) / downstream_length
            
            # make sure no negative slopes
            S_bf[S_bf < 0] = 0
        else:
            S_bf = np.ones(len(stream_pts_bf)) * S_init

        
        # calculate stream width during baseflow
        W_bf = k_w * baseflow ** omega
        
        # calc baseflow erosion rate
        Cb = W_bf * k_f * (baseflow / W_bf)**m * S_bf**n
        
        # calculate baseflow erosion during one timestep
        Cbt = Cb * dt
        
        # calculate eroded volume by dividing by (1 - porosity)
        VEb = Cbt / (1 - phi)

        # calculate resultig incision per timestep. 
        # assuming a uniform ditribution over the channel and a linear
        # increase from orogin to the position of the cross-section
        z_change_baseflow = -2 * VEb / W_bf / upstream_length_bf
        dz_bf[:] = 0.0
        dz_bf[stream_pts_bf] = z_change_baseflow

        if np.any(np.isnan(dz_bf)):
            msg = 'error, nan value in elevation change array dz_bf'
            raise ValueError(msg)
        
        # record baseflow erosion per year
        erosion_bf_per_yr_i = np.sum(VEb / (dt / year) / upstream_length_bf)
        erosion_bf_per_yr.append(erosion_bf_per_yr_i)
        
        # calculate upstream length for streams with overland flow
        if fixed_upstream_length is True:
            upstream_length_of = np.ones(len(contributing_area_of)) * upstream_length
        else:
            upstream_length_of = contributing_area_of * upstream_aspect_ratio
        
        # calculate total overland flow volume to be discharged per precipitation event
        V_of_3D = np.array([V2D * upstream_length_of for V2D in V_of_2D])

        # store data for overland flow
        n_str_of.append(np.sum(V_of_2D[0] > 0.0))
        #Pes.append(np.sum(P_excess))
        Q_overland_flows.append(Q_overland_flow)
        Vos.append(np.sum(V_of_2D))
        uls_of.append(upstream_length_of)
        cas_of.append(contributing_area_pts_of)
        strs_of.append(stream_pts_of)
        Q_overland_flow_nodes.append(Q_overland_flow_nodes_i.copy())

        ## recalculate stream slope for overland flow streams
        if recalculate_slope is True:
            #z_of = z[stream_pts_of]
            #z_origin_of = upstream_length_of * S_init
            #S_of = (z_origin_of - z_of) / upstream_length_of
            
            # downstream elevation
            # downstream length is assumed to equal upstream length
            z_downstream_init = -downstream_length * S_init
            
            z_downstream = z_downstream_init + U * time
            
            #
            z_of = z[stream_pts_of]
            S_of = (z_of - z_downstream) / downstream_length
            
            # make sure no negative slopes
            S_of[S_of < 0] = 0.0
            
        else:
            S_of = np.ones(len(stream_pts_of)) * S_init

        ## calc overland flow erosion
        n_precip = len(precip_depths)
        n_of_streams = len(stream_pts_of)
        V_erosion_of = np.zeros((n_precip, n_of_streams))
        
        # calculate eroded volume per precipitation event
        for precip_j in range(n_precip):
            for pt_i in range(n_of_streams):
                Vi = calculate_sediment_flux_overland_flow(V_of_3D[precip_j, pt_i], 
                                                           upstream_length_of[pt_i], 
                                                           S_of[pt_i], St, k_f, K_n, n, m)
                if np.isnan(Vi) == False:
                    V_erosion_of[precip_j, pt_i] = Vi

        # calculate channel width during overland flow event
        Q_of = V_of_3D / precipitation_duration
        channel_width_of = k_w * Q_of ** omega
        V_erosion_of_rock = V_erosion_of / (1.0 - phi)

        # distribute erosion over the entire channel (inc. the upstream reaches)
        # assuming a linear increase in erosion up the the position of the cross sect.
        dz_of_event = -2.0 * V_erosion_of_rock / (upstream_length_of * channel_width_of)
        dz_of_event[np.isnan(dz_of_event)] = 0.0
        
        # calculate erosion over 1 year per event
        dz_of_1year_corrected = np.array([dzi * pfi for dzi, pfi in zip(dz_of_event, precip_freqs)])
        
        # calculate erosion timestep for all events
        dz_of_str_all_events = dz_of_1year_corrected * dt / precip_return_time
        #dz_of_str_all_events = dz_of_1year_corrected / precip_return_time
        
        # sum all events to get overland flow erosion in one timestep
        dz_of_str = np.sum(dz_of_str_all_events, axis=0)
        
        # get overland flow erosion in stream pts only
        dz_of[:] = 0.0
        try:
            dz_of[stream_pts_of] = dz_of_str
        except:
            pdb.set_trace()
            
        erosion_of_per_yr_i = np.sum(V_erosion_of / upstream_length_of)
        erosion_of_per_yr.append(erosion_of_per_yr_i)

        if np.any(np.isnan(dz_of)):
            msg = 'error, nan value in elevation change array dz_of'
            raise ValueError(msg)
        
        if enforce_CFL_condition is True or explicit_diffusion_solver is True:
            # calculate max CFL timestep for hillslope diffusion
            q_hs = K_d * np.diff(z) / dx 
            dt_max_hs = max_CFL_number * dx / np.max(np.abs(q_hs))
      
        if explicit_diffusion_solver is True:
            # old explicit solver for hillslope diffusion:
            # fast, but potentially unstable:
            dz_hillslope = hillslope_diffusion(x, z, K_d, dx, dt)
        else:
            # new implicit solver:
            cd = np.ones_like(x)
            Qd = np.zeros_like(x)
            
            # include stream erosion as a source term in hillslope diffusion eq.
            if include_stream_erosion_in_hillslope_diff is True:
                Qd = (dz_of + dz_bf) / dt
            
            left_bnd_flux = 0.0
            right_bnd_flux = 0.0
            fixed_u_left = None
            fixed_u_right = None
            z_hs_new = solve_1D_diffusion(z, x, dt, K_d, cd, Qd,
                                          left_bnd_flux,
                                          right_bnd_flux,
                                          fixed_u_left,
                                          fixed_u_right)
            dz_hillslope = z_hs_new - z

        if np.any(np.isnan(dz_hillslope)):
            msg = 'error, nan value in elevation change array dz_hillslope'
            raise ValueError(msg)
            
        VE_hs = (1.0 - phi) * dz_hillslope / (dt / year) * dxn
        erosion_hd_per_yr_i = np.sum(VE_hs)
        erosion_hd_per_yr.append(erosion_hd_per_yr_i)

        ## Sum erosion and calculate new surface elevation
        # note checked timesteps. all erosion terms are given per timestep
        if include_stream_erosion_in_hillslope_diff is False:
            dz = dz_bf + dz_of + dz_hillslope
        else:
            dz = dz_hillslope

        # store elevation change for default timestep size
        dzs.append(dz)

        if np.any(np.isnan(dz)):
            msg = 'error, nan value in elevation change array dz'
            raise ValueError(msg)
        
        if variable_dt is True:
            # calculate min and max allowable elevation change in one timestep
            relief = z.max() - z.min()
            max_dz = np.max(np.abs(dz))
            
            if min_rel_dz is not None:
                min_allowed_dz = relief * min_rel_dz
            else:
                min_allowed_dz = 0.0
            if max_rel_dz is not None:
                max_allowed_dz = relief * max_rel_dz
            else:
                max_allowed_dz = 1e3
            
            if min_abs_dz is not None and min_allowed_dz < min_abs_dz:
                min_allowed_dz = min_abs_dz

            # change timestep size if elevation change is too low or high:
            if  max_dz < min_allowed_dz:
                dt_corr = dt * min_allowed_dz / max_dz               
            elif max_dz > max_allowed_dz:
                dt_corr = dt * max_allowed_dz / max_dz    
            else:
                dt_corr = dt
                dz_corr = dz   
                
            #
            if max_dt is not None and dt_corr > max_dt:
                dt_corr = max_dt

            # enforce CFL limit when explicit solver for hillslope diffusion is used:
            if (enforce_CFL_condition is True or explicit_diffusion_solver is True) and dt_corr > dt_max_hs:
                dt_corr = dt_max_hs

            # correct elevation change for change in timestep size
            dz_corr = dz * dt_corr / dt
        
        else:
            dz_corr = dz
            dt_corr = dt
        
        # update elevation & time
        z += dz_corr
        time += dt_corr
        
        if np.any(np.isnan(z)):
            raise ValueError('error, nan value in elevation array z')
        
        # store elevation, watertable, time and number of active streams
        zs.append(z.copy())
        hs.append(h)

        times.append(time)
        ti += 1
        ts_report += 1

        n_str_of_i = n_str_of[-1]
        
        # check if waterbalance is closed
        # note, time unit for waterbalance terms is one sec, all units should be 2D (m2/sec)
        #P_total = P * dxc.sum() 
        P_total = np.sum([Pdi * Pfi for Pdi, Pfi in zip(precip_depths, precip_freqs)]) * width
        ET_total = np.sum(actual_ET * dxc)
        Q_of_total = np.sum(np.array(Q_overland_flow)) * year
        Q_bf_total = np.sum(np.array(Q_baseflow_2D)) * year
        rch_total = rch * width * year
        pot_rch_total = np.sum(pot_rch * dxc)
        #wb = P_total - ET_total - Q_of_total - Q_bf_total
        wb = P_total - ET_total - Q_of_total - rch_total - gw_loss * width * year
        
        wb_err = np.abs(wb) / P_total * 100.0
        
        if debug is True and wb_err > 1.0:
            msg = f'warning, waterbalance error is {wb_err} %\n'
            msg += f'precip total = {P_total:0.0f}\n'
            msg += f'ET = {ET_total:0.0f}\n'
            msg += f'overland flow = {Q_of_total:0.0f}\n'
            msg += f'baseflow = {Q_bf_total:0.0f}\n'
            msg += f'total rch = {rch_total:0.0f}\n'
            msg += f'out of plane gw loss = {gw_loss * width * year:0.0f}\n'
            msg += f'potential rch = {pot_rch_total:0.0f}\n'
            print(msg)
            
            print(type(pot_rch))
            
            set_trace()
            
            #raise(ValueError)

    # end loop
    print('done with gw flow and erosion model loop')
    print('final timestep %i, time = %0.1f yrs of %0.0f yrs, active streams = %i, min z = %0.2f' 
      % (ti, time/year, total_runtime/year, n_str_of_i, z.min()))
    
    times = np.array(times)

    #                             save_initial_topography=False, initial_topography_file='initial_topography.csv',
                            # save_final_topography=False, final_topography_file='final_topography.csv',

    #
    if save_final_topography is True:
        dfr = pd.DataFrame(index=x, columns=['z'])
        dfr['z'] = zs[-1]
        print(f'saving relief final timestep to {final_topography_file}')
        dfr.to_csv(final_topography_file, index_label='x')

    # convert output to arrays
    Q_baseflows = np.array(Q_baseflows)
    Q_overland_flows = np.array(Q_overland_flows)
    n_str_of = np.array(n_str_of)

    erosion_of_per_yr = np.array(erosion_of_per_yr)
    erosion_bf_per_yr = np.array(erosion_bf_per_yr)
    erosion_hd_per_yr = np.array(erosion_hd_per_yr)

    dzs = np.array(dzs)
    Q_baseflow_nodes = np.array(Q_baseflow_nodes)
    Q_overland_flow_nodes = np.array(Q_overland_flow_nodes)
    
    model_results = (times, x, zs, hs, dzs, Q_baseflows, Q_overland_flows, n_str_of, 
                     erosion_of_per_yr, erosion_bf_per_yr, erosion_hd_per_yr,
                     Q_baseflow_nodes, Q_overland_flow_nodes)
    
    #print(m, n)
    
    return model_results