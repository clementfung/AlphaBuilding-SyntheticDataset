import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt 
import seaborn as sns
import pickle
import pdb
import networkx as nx
import utils

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, WhiteKernel
from sklearn.ensemble import GradientBoostingRegressor

def load_csv(filename='building_data.csv'):
    df = pd.read_csv(f'dataframes/{filename}')
    return df

def load_boosted_regression(name):
    load_obj = pickle.load(open(f'models/{name}.pkl', 'rb'))
    return load_obj['regressor_low'], load_obj['regressor_mid'], load_obj['regressor_high']

def train_boosted_regression_ci(X_train, Y_train, save_model=False, name='sample'):

    regressor = GradientBoostingRegressor(loss="quantile", alpha=0.5)
    regressor.fit(X_train, Y_train)

    regressor_low = GradientBoostingRegressor(loss="quantile", alpha=0.025)
    regressor_low.fit(X_train, Y_train)

    regressor_high = GradientBoostingRegressor(loss="quantile", alpha=0.975)
    regressor_high.fit(X_train, Y_train)
    
    if save_model:

        save_obj = {
            'regressor_low': regressor_low, 
            'regressor_mid': regressor,
            'regressor_high': regressor_high
        }
        
        pickle.dump(save_obj, open(f'models/{name}.pkl', 'wb'))

    return regressor, regressor_low, regressor_high

def train_gaussian_proc(X_train, Y_train):

    #kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=5, length_scale_bounds=(1e-2, 1e2))
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
    gp = GaussianProcessRegressor(kernel=kernel, alpha=100, n_restarts_optimizer=9)
    gp.fit(X_train, Y_train)

    # mean_pred, std_pred = gp.predict(X_space, return_std=True)
    return gp

def main_massflow_model():

    df_massflow = load_csv('FanAirMassFlowRate.csv')
    df_damper = load_csv('ZoneAirTerminalVAVDamperPosition.csv')
    df_flow = load_csv('ZoneMechanicalVentilationMassFlowRate.csv')
    df_occ = load_csv('ZonePeopleOccupantCount.csv')
    df_rawtemp = load_csv('ZoneTemperature.csv')

    df = load_csv('building_data.csv')
    outside_temp = df.iloc[:, 20]

    pdb.set_trace()

    # xmin = config['xmin']
    # xmax = config['xmax']
    # exp_name = config['exp_name']

    # open_idx = np.where(df['Operating Time'] == 'Yes')[0]
    # X_space = np.linspace(start=xmin, stop=xmax, num=1_000).reshape(-1, 1) 

def main_pairwise_models(config):

    df_massflow = load_csv('FanAirMassFlowRate.csv')
    df_damper = load_csv('ZoneAirTerminalVAVDamperPosition.csv')
    df_flow = load_csv('ZoneMechanicalVentilationMassFlowRate.csv')
    df_occ = load_csv('ZonePeopleOccupantCount.csv')
    df_rawtemp = load_csv('ZoneTemperature.csv')

    df = load_csv('building_data.csv')
    outside_temp = df.iloc[:, 20]
    exp_name = config['exp_name']

    open_idx = np.where(df['Operating Time'] == 'Yes')[0]

    for i in range(1, len(df_damper.columns)):
        
        if i == 45:
            # Skip one defective zone
            continue
        
        sanitized_name = df_damper.columns[i][:-40].upper()

        df_train = df_damper.iloc[open_idx,i]
        #df_train = df_occ.iloc[open_idx, i-12]
        #df_out = df_rawtemp.iloc[open_idx,i] 
        df_out = (df_rawtemp.iloc[:,i] - outside_temp)[open_idx]

        # For damper inputs: filter out closed
        # open_damper_idx = df_train > 0.201
        # df_train = df_train[open_damper_idx]
        # df_out = df_out[open_damper_idx]

        X_train = df_train.values.reshape(-1, 1)
        Y_train = df_out.values

        corr = utils.get_corr(X_train.T, Y_train)
        print(f'{sanitized_name}: corr={corr}')

        reg, reg_low, reg_high = train_boosted_regression_ci(X_train, Y_train, save_model=False, name=f'model-col{i}')

        xmin = np.min(X_train)
        xmax = np.max(X_train)
        X_space = np.linspace(start=xmin, stop=xmax, num=1_000).reshape(-1, 1) 
        mean_pred = reg.predict(X_space)
        low_pred = reg_low.predict(X_space)
        high_pred = reg_high.predict(X_space)

        plt.title(f'{sanitized_name}: corr={corr:.4f}')
        plt.scatter(df_train, df_out)
        plt.plot(X_space, mean_pred, color='red')
        plt.plot(X_space, low_pred, color='red')
        plt.plot(X_space, high_pred, color='red')
        
        plt.close()

def attack_pairwise_models(config):

    df_massflow = load_csv('FanAirMassFlowRate.csv')
    df_damper = load_csv('ZoneAirTerminalVAVDamperPosition.csv')
    df_flow = load_csv('ZoneMechanicalVentilationMassFlowRate.csv')
    df_occ = load_csv('ZonePeopleOccupantCount.csv')
    df_rawtemp = load_csv('ZoneTemperature.csv')

    df = load_csv('building_data.csv')
    outside_temp = df.iloc[:, 20]
    exp_name = config['exp_name']

    open_idx = np.where(df['Operating Time'] == 'Yes')[0]
    attack_point = 14000

    for i in range(1, len(df_damper.columns)):
        
        if i == 45:
            # Skip one defective zone
            continue
        
        sanitized_name = df_damper.columns[i][:-40].upper()
        df_train = df_damper.iloc[open_idx,i]
        df_out = (df_rawtemp.iloc[:,i] - outside_temp)[open_idx]

        X_train = df_train.values.reshape(-1, 1)
        Y_train = df_out.values

        pdb.set_trace()

        corr = utils.get_corr(X_train.T, Y_train)
        print(f'{sanitized_name}: corr={corr}')

        reg, reg_low, reg_high = load_boosted_regression(name=f'model-col{i}')

        xmin = np.min(X_train)
        xmax = np.max(X_train)
        X_space = np.linspace(start=xmin, stop=xmax, num=1_000).reshape(-1, 1) 
        mean_pred = reg.predict(X_space)
        low_pred = reg_low.predict(X_space)
        high_pred = reg_high.predict(X_space)

        plt.title(f'{sanitized_name}: corr={corr:.4f}')
        plt.scatter(df_train, df_out)
        plt.plot(X_space, mean_pred, color='red')
        plt.plot(X_space, low_pred, color='red')
        plt.plot(X_space, high_pred, color='red')
    
        plt.scatter(X_train[attack_point], Y_train[attack_point], color='green')
        plt.scatter(X_train[attack_point], Y_train[attack_point] * 0.5, color='green')
        plt.show()
        plt.close()

if __name__ == '__main__':

    # main_massflow_model()
    main_pairwise_models({'exp_name': 'damper-tempdiff'})

