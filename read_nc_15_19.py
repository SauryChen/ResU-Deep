# This file is used to read nc data.
# for global test, the areas should have the same size. ( 121 * 280 )
# delete + 1 after lon_ed * 4
#import time
import netCDF4 as nc
import numpy as np

def read_nc(filename, variable, level, loc, t):
    lat_st = loc[0]
    lat_ed = loc[1]
    lon_st = loc[2]
    lon_ed = loc[3]
    
    dir_path = '/ceph-data/cmx/ERA5_15_19/'
    file = dir_path + filename + '.nc'
    data_file = nc.Dataset(file)
    
    if (level == -2):
        data_a = data_file[variable][0,:,(lon_st*4):(lon_ed*4):1]
        return data_a
    
    elif (level == -1):
        data_a = data_file[variable][3*t,:,(lon_st*4):(lon_ed*4):1]
        return data_a
    
    elif (level == 0):
        if (filename == 'rh' or filename == 'sh' or filename == 'temperature' or filename == 'u_wind' or filename == 'v_wind'):
            data_a = data_file[variable][3*t,0,:,(lon_st*4):(lon_ed*4):1]
            data_b = data_file[variable][3*t,1,:,(lon_st*4):(lon_ed*4):1]
            data_c = data_file[variable][3*t,2,:,(lon_st*4):(lon_ed*4):1]
            return data_a, data_b, data_c
        
        elif (filename == 'vertical_velocity'):
            data_a = data_file[variable][3*t,0,:,(lon_st*4):(lon_ed*4):1]
            data_b = data_file[variable][3*t,1,:,(lon_st*4):(lon_ed*4):1]
            return data_a, data_b
    
    else:
        print("Error: Wrong Document!")
        
def load_variables(loc, t):
    #start = time.time()
    geopotential = read_nc('geopotential', 'z', -2, loc, t)
    geoheight = geopotential / 9.80665
    cape = read_nc('cape', 'cape', -1, loc, t)
    cin = read_nc('cin', 'cin', -1, loc, t)
    tciw = read_nc('cloud', 'tciw', -1 , loc, t)
    tclw = read_nc('cloud', 'tclw', -1 , loc, t)
    tcwv = read_nc('cloud', 'tcwv', -1 , loc, t)
    #p84_162 = read_nc('moisture', 'p84.162', -1 , loc)
    ie = read_nc('inst_moist', 'ie', -1, loc, t)
    blh = read_nc('soil_boundary', 'blh', -1, loc, t)
    st = read_nc('soil_boundary', 'stl1', -1, loc, t)
    ssrd = read_nc('solar_radiation', 'ssrd', -1, loc, t)
    #st = np.load(numpy_file)
    surlh = read_nc('surlh', 'slhf', -1, loc, t)
    sursh = read_nc('sursh', 'sshf', -1, loc, t)
    rh_300, rh_500, rh_700 = read_nc('rh', 'r', 0, loc, t)
    sh_300, sh_500, sh_700 = read_nc('sh', 'q', 0, loc, t)
    t_300, t_500, t_700 = read_nc('temperature', 't', 0, loc, t)
    u_300, u_500, u_700 = read_nc('u_wind', 'u', 0, loc, t)
    v_300, v_500, v_700 = read_nc('v_wind', 'v', 0, loc, t)
    u_925 = read_nc('wind_925', 'u', -1, loc, t)
    v_925 = read_nc('wind_925', 'v', -1, loc, t)
    ver_500, ver_700 = read_nc('vertical_velocity', 'w', 0, loc, t)
    #end = time.time()
    #duration = end - start
    #print('Reading Data: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))

    # prepare dataset:
    data_x = np.dstack((cape, cin, tciw, tclw, tcwv, blh, st, surlh, sursh, ie, ssrd,\
                             rh_300, rh_500, rh_700, \
                             sh_300, sh_500, sh_700, \
                             t_300, t_500, t_700, \
                             u_300, u_500, u_700, u_925,\
                             v_300, v_500, v_700, v_925,\
                             ver_500, ver_700, geoheight))
    #print("variable dataset has the shape: ", data_x.shape) # should be [121, 280, 31]
    
    # correction check
    #print(np.sum(np.abs(data_x[:,:,0] - cape)))
    #print(np.sum(np.abs(data_x[:,:,1] - cin)))
    
    return data_x
    
#load_variables([30,0,240,310], 100)

# This function is too slow. Neglected.
def load_label(loc, t):
    lon_st = loc[2]
    lon_ed = loc[3]
    label = np.load('/ceph-data/cmx/TRMM/deep_label.npy')
    label = label[t, :, (lon_st*4):(lon_ed*4):1]
    return label
