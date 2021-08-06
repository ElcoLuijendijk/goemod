# GOEMod: Groundwater outcrop and erosion model

################################################
## import modules:
################################################
import itertools
import string
import datetime
import inspect
import time
import pickle
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.gridspec import GridSpec

import pandas as pd

# import gw and erosion functions
#from lib.gw_erosion_lib import *
import lib.gw_erosion_lib as gwe


################################################
## Some fixed values
################################################
hour = 60.0 * 60.0
day = 24.0 * hour
year = 365.25 * day
kyr = 1e3 * year

################################################
## Set up model default parameters
################################################
class ModelParams:
    
    hour = 60.0 * 60.0
    day = 24.0 * hour
    year = 365.25 * day
    kyr = 1e3 * year

    # timestep
    dt = 1.0 * year

    #n_timesteps = 1000
    total_runtime = 1e4 * year

    # dimensions of model domain in x direction, perpendicular to streams
    width = 20000

    #
    dx = 5.0

    #
    fixed_upstream_length = True

    # upstream length: length of contributing area to streams, upstream from the modeled 2D cross section
    upstream_length = 10e3

    # downstream length: length to downstream boundary with fixed elevation
    # used to calculate stream slope if recalculate_slope = True
    downstream_length = 10e3

    # initial relief
    initial_relief = 0.5
    n_initial_topo_segments = 200
    
    #
    use_relief_from_file = True
    #relief_input_file = 'saved_topographies/default_topography_esurf_paper_revision.csv'
    # uncomment the next lines to use an incised topography as initial topography
    # this was used for the model runs to test the persistence of streams in the revised manuscript
    relief_input_file = 'saved_topographies/saved_final_topography_base_case_1e4yr.csv'
    
    # save initial topography
    save_initial_topography = False

    # name of file with initial topography data
    initial_topography_file = 'saved_initial_topography.csv'
    
    # save the generated final topography to a new file
    save_final_topography = False

    # name of file with final topography data
    final_topography_file = 'saved_final_topography.csv'

    #
    #max_z_change_limit = 0.1

    # aspect ratio to calculate size of each sub-catchment in the 3rd dimension
    # aspect ratio of 2.0 means that the upstream contributing length 
    # is 2x the length of the contributing area
    # in the model domain / cross-section
    upstream_aspect_ratio = 1.0

    # precipitation (=precip - evapotranspiration)
    P = 0.75 / year
    
    #
    precip_return_time = 1.0 * year
    
    # infiltration capacity
    infiltration_capacity = 1.0e-4

    # evapotranspiration
    ET = 0.375 / year

    # overland flow vs recharge parameters
    phi = 0.20
    specific_yield = 0.20

    # average recharge or calculate spatially distributed values
    average_rch = False

    # transmissivity (m2/sec)
    T = 1.0e-2

    # erosion parameters:
    #rho_s = 2650.0

    ## Erosion parameters in Boogaart et al. (2002, Geomorphology)
    ## These are nicely theoretically justified for an average sand with diameter of 0.3 mm
    ## note that the manuscript contains an inconsistency, if kf=0.011 is used as suggested in eq. 19
    ## then the term 1 / (rho_s * (1- porosity)) drops out of eq. 4.
    k_f = 10**(3.1)
    m = 1.8
    n = 2.1
    
    # hillslope diffusion
    K_d = 10**-2.0 / year

    # Manning coefficient
    # 20-30 in de Vries (1995)
    #K_n = 1.0 / 0.035
    K_n = 25.0

    # channel slope
    S_init = S = 4.0e-4
    
    #
    recalculate_slope = True

    # baselevel change (m/s)
    U = -2e-5 / year
    # fast baselevel drop scenario
    #U = -1e-3 / year
    
    # parameters relating channel width and discharge, based on compilation by van den Berg (1995)
    k_w = 3.65
    omega = 0.5

    # slope of stream bed perpendicular to flow direction
    St = 0.002
    
    # variable timestepping
    variable_dt = True
    
    # maximum timestep
    max_dt = 1000 * year
    
    # maximum relative relief change in one timestep. expressed as fraction of total relief
    max_rel_dz = 0.005

    # minimum relative relief change in one timestep
    min_rel_dz = 0.001

    # minimum absolte change in relief in on timestep
    min_abs_dz = 0.001

    #
    init_z = 0.0

    # precipitation duration
    hour = 3600.
    day = 24 * hour
    precipitation_duration = 3 * hour


################################################
## Set up parameter ranges for sensitivity analysis or parameter space exploration
################################################
model_name = 'sensitivity'


class ParameterRanges:

    """
    parameter ranges for sensitivity or uncertainty analysis

    the model will look for any variable ending with _s below and then look for the
    corresponding variable in model_parameters.py

    each _s variable should be a list of values, beo.py will replace the variable
    in ModelParams with each item in the list consecutively
    """

    year = 365.25 * 24 * 60 * 60.0

    # option whether to vary one model parameter at a time
    # (ie for a sensitivtiy analysis)
    # or to run all parameter combinations, using the parameter ranges specified below
    parameter_combinations = False

    # option to add a first base run with unchanged parameters to the list of model
    # runs
    initial_base_run = True
    
    nsteps = 10
    
    # parameter ranges:
    P_s = np.linspace(0.25, 1.5, nsteps) / year
    specific_yield_s = np.linspace(0.1, 0.5, nsteps)
    K_d_s = np.linspace(4.0e-5, 4e-2, nsteps) / year
    k_f_s = 10**np.linspace(2.3, 4.2, nsteps)
    U_s = -10**np.linspace(-5, -3, nsteps) / year
        

################################################
## Generate an id number for this particular model run
################################################
model_id = int(np.random.rand(1) * 1000)

today = datetime.datetime.now()
today_str = '%i-%i-%i' % (today.day, today.month, today.year)

output_file_adj = '%s_id_%i_%s' % (model_name, model_id, today_str)

print('appending this to each figure and output dataset filename: ', output_file_adj)

################################################
## Set up parameter sets for multiple model runs
################################################
pr = ParameterRanges

# create list with param values for each model run
scenario_param_names_raw = dir(pr)
scenario_param_names = [m for m in scenario_param_names_raw
                        if '__' not in m and '_s' in m]

scenario_parameter_list = [getattr(pr, p)
                           for p in scenario_param_names]

# construct list with all parameter combinations
if pr.parameter_combinations is True:
    scenario_parameter_combinations = \
        list(itertools.product(*scenario_parameter_list))
else:
    nscens = np.sum(np.array([len(sp) for sp in scenario_parameter_list
                              if sp is not None]))
    nparams = len(scenario_parameter_list)
    scenario_parameter_combinations = []

    if pr.initial_base_run is True:
        scenario_parameter_combinations.append([None] * nparams)

    for j, sl in enumerate(scenario_parameter_list):
        if sl[0] is not None:
            sc = [None] * nparams
            for sli in sl:
                sci = list(sc)
                sci[j] = sli
                scenario_parameter_combinations.append(sci)

param_list = scenario_parameter_combinations

################################################
## Get parameter names
################################################
mp = ModelParams
Parameters = ModelParams

# get attributes
attributes = inspect.getmembers(
    Parameters, lambda attribute: not (inspect.isroutine(attribute)))
attribute_names = [attribute[0] for attribute in attributes
                   if not (attribute[0].startswith('__') and
                           attribute[0].endswith('__'))]

# set up pandas dataframe to store model input params
n_model_runs = len(param_list)

## Set up pandas dataframe to store results
# set up pandas dataframe to store model input params
n_model_runs = len(param_list)
#n_ts = np.sum(np.array(mp.N_outputs))
n_ts = 1
n_rows = n_model_runs * n_ts

ind = np.arange(n_rows)
columns = ['model_run', 'model_error', 'timestep', 'runtime_yr', 'computational_time'] + attribute_names

df = pd.DataFrame(index=ind, columns=columns)

################################################
## Run all model experiments
################################################

n_runs = len(param_list)

# set up arrays to store results final timestep
n_streams_final = np.zeros(n_runs)
erosion_rates = np.zeros(n_runs)
incision_rates = np.zeros(n_runs)
wt_depth_avg = np.zeros(n_runs)
ratio_overland_and_baseflows = np.zeros(n_runs)
ratio_overland_and_baseflow_erosion = np.zeros(n_runs)
end_times = np.zeros(n_runs)

# list for storing the variables that vary over time:
zs_all = []
hs_all = []
times_all = []
n_str_all = []
#Q_baseflows, Q_overland_flows, erosion_of_per_yr, erosion_bf_per_yr, erosion_hd_per_yr

Q_baseflows_all = []
Q_overland_flows_all = []
erosion_of_per_yr_all = []
erosion_bf_per_yr_all = []
erosion_hd_per_yr_all = []

# loop through all model experiments
for model_run, param_set in enumerate(param_list):

    print('-' * 20)

    # reload default params
    Parameters = mp()

    # update default parameters in Parameter class
    for scenario_param_name, scenario_parameter in \
            zip(scenario_param_names, param_set):

        if scenario_parameter is not None:
            # find model parameter name to adjust
            model_param_name = scenario_param_name[:-2]

            print('updating parameter %s from %s to %s' 
                  % (model_param_name,
                     str(getattr(Parameters, model_param_name)),
                     str(scenario_parameter)))

            # update model parameter
            setattr(Parameters, model_param_name, scenario_parameter)

    print('-' * 20)

    # store input parameters in dataframe
    attributes = inspect.getmembers(
        Parameters, lambda attribute: not (inspect.isroutine(attribute)))
    attribute_dict = [attribute for attribute in attributes
                      if not (attribute[0].startswith('__') and
                              attribute[0].endswith('__'))]

    for a in attribute_dict:
        if a[0] in df.columns:
            if type(a[1]) is list or type(a[1]) is np.ndarray:
                df.loc[model_run, a[0]] = str(a[1])
            else:
                df.loc[model_run, a[0]] = a[1]
                
    print('running single model')
    
    print('model run %i of %i' % (model_run +1 , len(param_list)))
    
    # run the model once
    model_results = gwe.model_gwflow_and_erosion(
        Parameters.width, Parameters.dx, 
        Parameters.total_runtime, Parameters.dt, 
        Parameters.min_rel_dz, Parameters.max_rel_dz, Parameters.min_abs_dz,
        Parameters.fixed_upstream_length, Parameters.upstream_length,
        Parameters.upstream_aspect_ratio,
        Parameters.downstream_length,
        Parameters.P, mp.precipitation_duration, Parameters.precip_return_time, 
        Parameters.infiltration_capacity, Parameters.ET, 
        Parameters.initial_relief, Parameters.n_initial_topo_segments, Parameters.init_z,
        Parameters.T, Parameters.specific_yield, Parameters.phi,
        Parameters.S_init, Parameters.St, 
        Parameters.k_f, Parameters.K_n, 
        Parameters.n, Parameters.m, Parameters.k_w, 
        Parameters.omega, Parameters.K_d, Parameters.recalculate_slope, Parameters.U,
        variable_dt=mp.variable_dt, max_dt=mp.max_dt, 
        use_relief_from_file=Parameters.use_relief_from_file, relief_input_file=Parameters.relief_input_file)

    # copy model results
    (times, x, zs, hs, dzs, Q_baseflows, Q_overland_flows, n_str_of, 
     erosion_of_per_yr, erosion_bf_per_yr, erosion_hd_per_yr,
     Q_baseflow_nodes, Q_overland_flow_nodes) = model_results
    
    # record erosion and incision rates and n streams at final step
    erosion_rate = np.mean(zs[-1] - zs[0]) / times[-1]
    erosion_rates[model_run] = erosion_rate
    
    max_incision_loc = np.argmin(zs[-1])
    incision_rate = (zs[-1][max_incision_loc] - zs[0][max_incision_loc]) / times[-1]
    incision_rates[model_run] = incision_rate
    
    n_streams_final[model_run] = n_str_of[-1]
    
    wt_depth_avg[model_run] = np.mean(zs[-1] - hs[-1])

    ratio_overland_and_baseflows[model_run] = Q_overland_flows[-1] / Q_baseflows[-1]
    
    ratio_overland_and_baseflow_erosion[model_run] = erosion_of_per_yr[-1] / erosion_bf_per_yr[-1]
    
    end_times = times[-1]
    
    times_all.append(times)
    zs_all.append(zs)
    hs_all.append(hs)
    n_str_all.append(n_str_of)
    
    Q_baseflows_all.append(Q_baseflows)
    Q_overland_flows_all.append(Q_overland_flows)
    erosion_of_per_yr_all.append(erosion_of_per_yr)
    erosion_bf_per_yr_all.append(erosion_bf_per_yr)
    erosion_hd_per_yr_all.append(erosion_hd_per_yr)

################################################
## Store the result in a pandas dataframe
################################################
#cols = ['model_run', 'T', 'time', 'n_streams', 'stream_density', 'erosion_rate', 'incision_rate', 
#        'avg_watertable_depth', 'ratio_overland_flow_baseflow', 'ratio_overland_and_baseflow_erosion']
#index = np.arange(len(Ts))

#df = pd.DataFrame(columns=cols, index=index)

#df['model_run'] = index
#df['T_m2s-1'] = Ts
df['time_yr'] = end_times / year
df['n_streams'] = n_streams_final
df['stream_density_str_per_km'] = n_streams_final / mp.width * 1e3
df['erosion_rate_m_per_yr'] = -erosion_rates * year
df['incision_rate_m_per_yr'] = -incision_rates * year
df['avg_watertable_depth_m'] = wt_depth_avg 
df['ratio_overland_flow_baseflow'] = ratio_overland_and_baseflows
df['ratio_overland_and_baseflow_erosion'] = ratio_overland_and_baseflow_erosion
df['min_elevation'] = zs[-1].min()

fn = 'model_results/model_results_%i_runs_%0.0fyrs_%s.csv' \
    % (len(df), mp.total_runtime / year, output_file_adj)
print('saving model result summary as %s' % fn)
df.to_csv(fn)

#######
# store results in a pickle file

output_data = [df, times_all, x, zs_all, hs_all, 
               n_str_all, Q_baseflows_all, Q_overland_flows_all, 
               erosion_of_per_yr_all, erosion_bf_per_yr_all,
               erosion_hd_per_yr_all]

fn = 'model_results/model_results_%i_runs_%0.0fyrs_%s.pck' \
    % (len(df), mp.total_runtime / year, output_file_adj)
print(f'saving model results as {fn}')
fout = open(fn, 'wb')
pickle.dump(output_data, fout)
fout.close()

print('done')
