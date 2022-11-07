import pandas as pd
import numpy as np
import h5py
import os
import datetime
from matplotlib import pyplot as plt 
import seaborn as sns
import s3fs
import pdb

# Converting factors
J_to_kWh = 1/3600000
area = 4982.22 # m2
kBtu_to_kWh = 0.293071
ft2_to_m2 = 0.092903
kBtu_per_ft2_to_kWh_per_m2 = kBtu_to_kWh/ft2_to_m2

# Utility functions
def get_df_from_hdf(hdf, climate='1A', efficiency='High', year='TMY3', str_run='run_1', data_key='ZonePeopleOccupantCount'):
    '''
    This function extracts the dataframe from the all run hdf5 file
    '''
    ts_root = hdf.get('3. Data').get('3.2. Timeseries')
    sub = ts_root.get(climate).get(efficiency).get(year).get(str_run).get(data_key)
    cols = np.array(sub.get('axis0'))[1:].astype(str)
    data = np.array(sub.get('block1_values'))
    df = pd.DataFrame(data, columns = cols)
    df.index = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in pd.date_range('2006-01-01', '2007-01-01', freq='10min')[:-1]]
    return df

def ls_to_text_file(ls_lines, dir_txt_out, decode='UTF-8'):
    '''
    This function writes a list of strings to a text file
    '''
    with open(dir_txt_out, 'w') as filehandle:
        for line in ls_lines:
            filehandle.write(line.decode(decode))
        filehandle.close()

        
def get_monthly_sum(hdf, str_clm, str_eff, str_yr, str_run, var='ElectricityFacility', convert=J_to_kWh):
    '''
    This function calculates the monthly sum of a specific variable from a yearly simulation
    '''
    df = get_df_from_hdf(
        hdf, 
        str_clm, 
        str_eff, 
        str_yr, 
        str_run, 
        var
    )
    df['datetime'] = pd.to_datetime(df.index)
    df = df.reset_index(drop=True)
    df['month'] = df['datetime'].dt.month
    df_temp = df.groupby('month').sum()*J_to_kWh
    return np.array(df_temp.iloc[:, 0])
        
        
def get_all_monthly_site_energy(hdf, str_clm, str_eff):
    '''
    This function getd the monthly site energy consumptions for all simulations in for a specific climate and efficiency level
    '''
    ts_root = hdf.get('3. Data').get('3.2. Timeseries')
    dict_all_years = {}
    ls_yrs = list(ts_root.get(str_clm).get(str_eff).keys())
    for str_yr in ls_yrs:
        ls_runs = list(ts_root.get(str_clm).get(str_eff).get(str_yr).keys())  
        for i, str_run in enumerate(ls_runs):
            if i == 0:
                arr_monthly_sums = get_monthly_sum(hdf, str_clm, str_eff, str_yr, str_run, 'ElectricityFacility') + get_monthly_sum(hdf, str_clm, str_eff, str_yr, str_run, 'GasFacility')
            else:
                arr_monthly_sums += get_monthly_sum(hdf, str_clm, str_eff, str_yr, str_run, 'ElectricityFacility') + get_monthly_sum(hdf, str_clm, str_eff, str_yr, str_run, 'GasFacility')
        dict_all_years[str_yr] = (arr_monthly_sums/5)
    return dict_all_years


def get_all_weather(str_clm, hdf, start_year):
    '''
    This function gets the outdoor air temperature data for all years of the specific climate zone starting from a certain year.
    '''
    str_eff = 'Standard'
    str_run = 'run_1'
    count = 0
    ts_root = hdf.get('3. Data').get('3.2. Timeseries')
    ls_yrs = list(ts_root.get(str_clm).get(str_eff).keys())
    for str_yr in ls_yrs:
        if str_yr != 'TMY3':
            dd = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'SiteOutdoorAirDrybulbTemperature')
            if count == 0:
                arr_all_t = dd.to_numpy()
            else:
                arr_all_t = np.concatenate((arr_all_t, dd.to_numpy()), axis=0)
        count += 1
    df_all_OAT = pd.DataFrame(arr_all_t, 
                              columns=['OAT'], 
                              index=pd.date_range(f'{start_year}-01-01', periods=1576800, freq='10Min'))
    df_all_OAT['datetime'] = df_all_OAT.index
    df_all_OAT['month'] = df_all_OAT['datetime'].dt.strftime('%Y-%m')
    return df_all_OAT


def get_ts_data(hdf, str_clm = '5A', str_eff = 'Standard', str_yr = '1992', str_run = 'run_1'):
    '''
    This function extracts the time series data from the HDF5 file
    '''
    ts_root = hdf.get('3. Data').get('3.2. Timeseries')
    ts_root.get(str_clm).get(str_eff).get(str_yr).get(str_run).keys()
    
    df_cool = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'CoolingElectricity')
    df_ele = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'ElectricityFacility')
    df_havc_ele = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'ElectricityHVAC')
    df_ext_ele = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'ExteriorLightsElectricity')
    df_fans = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'FansElectricity')
    df_gas = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'GasFacility')
    df_gas_hvac = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'GasHVAC')

    df_heat_ele = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'HeatingElectricity')
    df_mels = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'InteriorEquipmentElectricity')
    df_lights = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'InteriorLightsElectricity')

    df_pumpep = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'PumpElectricPower')
    df_pumpe = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'PumpsElectricity')

    df_oat = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'SiteOutdoorAirDrybulbTemperature')
    df_occs = pd.DataFrame(get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, 'ZonePeopleOccupantCount').sum(axis=1), columns=['Total Occupant Count'])

    df_col = pd.concat([df_occs, df_cool, df_lights, df_mels, df_ele, df_ext_ele, df_fans, df_heat_ele, df_gas, df_gas_hvac, df_havc_ele, df_pumpep, df_pumpe, df_oat], axis=1)
    df_col['datetime'] = pd.to_datetime(df_col.index)
    df_col['climate'] = str_clm
    df_col['efficiency'] = str_eff
    df_col['weekday'] = np.where(df_col['datetime'].dt.dayofweek < 5, True, False)
    df_col['Interior Lighting (kWh)'] = df_col['InteriorLights:Electricity[J]'] * J_to_kWh
    df_col['MELs (kWh)'] = df_col['InteriorEquipment:Electricity[J]'] * J_to_kWh
    df_col['Site Electricity (kWh)'] = df_col['Electricity:Facility[J]'] * J_to_kWh
    df_col['Site Gas (kWh)'] = df_col['Gas:Facility[J]']* J_to_kWh
    df_col['Site Total Energy (kWh)'] = df_col['Site Electricity (kWh)'] + df_col['Site Gas (kWh)']
    df_col['HVAC Electricity (kWh)'] = df_col['Electricity:HVAC[J]'] * J_to_kWh
    df_col['Outdoor Air Temperature (degC)'] = df_col['Environment:Site Outdoor Air Drybulb Temperature[C]']
    df_col = df_col.drop(['InteriorLights:Electricity[J]', 
                          'InteriorEquipment:Electricity[J]', 
                          'Electricity:Facility[J]',
                          'Gas:Facility[J]',
                          'Environment:Site Outdoor Air Drybulb Temperature[C]'
                         ], axis=1)
    df_col['Operating Time'] = np.where(
        (df_col['datetime'].dt.hour > 6) & 
        (df_col['datetime'].dt.hour < 20) &
        (df_col['weekday'] == True), 'Yes', 'No')
    df_col['hour'] = df_col['datetime'].dt.hour
    return df_col

def save_zone_data(hdf, str_clm = '5A', str_eff = 'Standard', str_yr = '1992', str_run = 'run_1'):
    
    list_vars = [
        'AirSystemOutdoorAirEconomizerStatus',
        'FanAirMassFlowRate',
        'PumpMassFlowRate',
        'SystemNodeMassFlowRate',
        'SystemNodePressure',
        'SystemNodeRelativeHumidity',
        'SystemNodeTemperature', 
        'ZoneAirRelativeHumidity',
        'ZoneAirTerminalVAVDamperPosition',
        'ZoneElectricEquipmentElectricPower',
        'ZoneLightsElectricPower', 
        'ZoneMeanAirTemperature',
        'ZoneMechanicalVentilationMassFlowRate',
        'ZonePeopleOccupantCount',
        'ZoneThermostatCoolingSetpointTemperature', 
        'ZoneThermostatHeatingSetpointTemperature'
    ]

    for var in list_vars:
        df = get_df_from_hdf(hdf, str_clm, str_eff, str_yr, str_run, var)
        df.to_csv(f'{var}.csv')
        print(f'saved {var}')

def main(filename='building_data.csv'):

    # Access the HDF5 file from AWS S3 bucket
    print('Extracting from S3....')
    dir_aws_s3 = 's3://oedi-data-lake/building_synthetic_dataset/A_Synthetic_Building_Operation_Dataset.h5'
    s3 = s3fs.S3FileSystem(anon=True)
    s3file = s3.open(dir_aws_s3, "rb")
    hdf = h5py.File(s3file)

    print('Extracting system and zone data from S3....')
    save_zone_data(hdf)

    print('Reading from HDF file....')
    df_col = get_ts_data(hdf)
    df_col.to_csv(filename)

    print(f'Saved to {filename}')

    return 0

if __name__ == "__main__":
    
    main()