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
from sklearn import tree, linear_model
import scipy.stats as ss

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

def process_zone_data():

    df_massflow = load_csv('FanAirMassFlowRate.csv')

    df_damper = load_csv('ZoneAirTerminalVAVDamperPosition.csv')
    df_flow = load_csv('ZoneMechanicalVentilationMassFlowRate.csv')
    df_occ = load_csv('ZonePeopleOccupantCount.csv')
    df_rawtemp = load_csv('ZoneTemperature.csv')
    df_heatset = load_csv('ZoneThermostatHeatingSetpointTemperature.csv')
    df_coolset = load_csv('ZoneThermostatCoolingSetpointTemperature.csv')
    df_rawtemp_atk = load_csv('ZoneTemperature_Atk.csv')

    df = load_csv('building_data.csv')
    outside_temp = df.iloc[:, 20]
    open_idx = np.where(df['Operating Time'] == 'Yes')[0]

    df_sub = dict()

    for i in range(1, len(df_occ.columns)):

        sanitized_name = df_occ.columns[i][:-29].upper()
        #df_sub['out_temp'] = outside_temp
        df_sub['occ'] = df_occ.iloc[open_idx,i]

        cidx = utils.get_column_idx_match(sanitized_name, df_flow.columns)
        print(df_flow.columns[cidx])
        df_sub['flow'] = df_flow.iloc[open_idx,cidx]

        cidx = utils.get_column_idx_match(sanitized_name, df_damper.columns)
        print(df_damper.columns[cidx])
        df_sub['damper'] = df_damper.iloc[open_idx,cidx]

        cidx = utils.get_column_idx_match(sanitized_name, df_heatset.columns)
        print(df_heatset.columns[cidx])
        df_sub['heatset'] = df_heatset.iloc[open_idx,cidx]

        cidx = utils.get_column_idx_match(sanitized_name, df_coolset.columns)
        print(df_flow.columns[cidx])
        df_sub['coolset'] = df_coolset.iloc[open_idx,cidx]

        cidx = utils.get_column_idx_match(sanitized_name, df_rawtemp.columns)
        print(df_rawtemp.columns[cidx])
        df_sub['tempdiff'] = df_rawtemp.iloc[open_idx,cidx] - outside_temp[open_idx]
        df_sub['tempdiff_atk'] = df_rawtemp_atk.iloc[open_idx,cidx] - outside_temp[open_idx]

        df = pd.DataFrame(df_sub)

        save_name = sanitized_name.replace(' ', '')
        df.to_csv(f'{save_name}_train.csv')

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
        plt.scatter(df_train, df_out, color='black')
        
        plt.ylabel('Relative temperature difference')
        plt.xlabel('Damper Position')
        
        plt.plot(X_space, mean_pred, color='green')
        plt.plot(X_space, low_pred, color='green')
        plt.plot(X_space, high_pred, color='green')
        
        plt.close()

def main_mass_models():

    df_massflow = load_csv('FanAirMassFlowRate.csv')
    df_damper = load_csv('ZoneAirTerminalVAVDamperPosition.csv')
    df_rawtemp = load_csv('ZoneTemperature.csv')
    df_rawtemp_atk = load_csv('ZoneTemperature_Atk.csv')

    df = load_csv('building_data.csv')
    open_idx = np.where(df['Operating Time'] == 'Yes')[0]
    outside_temp = df.iloc[:, 20]
    z_idx = 1

    for subname in ['TOP', 'MID', 'BOT']:

        flow_top = dict()
        temp_top = dict()
        temp_atk = dict()

        for i in range(1, len(df_damper.columns)):
            sanitized_name = df_damper.columns[i][:-40].upper()

            if subname in sanitized_name:
                flow_top[sanitized_name] = df_damper.iloc[:,i].values
                temp_top[sanitized_name] = df_rawtemp.iloc[:,i].values
                temp_atk[sanitized_name] = df_rawtemp_atk.iloc[:,i].values

        df_damp = pd.DataFrame(flow_top)
        df_temp = pd.DataFrame(temp_top)
        df_temp_atk = pd.DataFrame(temp_atk)

        mass_flow = df_massflow.iloc[open_idx, z_idx]
        damp_pts = np.mean(df_damp, axis=1)[open_idx]
        temp_pts = (np.mean(df_temp, axis=1) - outside_temp)[open_idx]
        temp_atk_pts = (np.mean(df_temp_atk.iloc[19930:19960], axis=1) - outside_temp[19930:19960])

        X_train = damp_pts.values.reshape(-1, 1)
        Y_train = temp_pts.values

        corr = utils.get_corr(X_train.T, Y_train)
        print(f'Top corr={corr}')

        reg, reg_low, reg_high = train_boosted_regression_ci(X_train, Y_train, save_model=False, name=f'model-mass{i}')

        xmin = np.min(X_train)
        xmax = np.max(X_train)
        X_space = np.linspace(start=xmin, stop=xmax, num=1_000).reshape(-1, 1) 
        mean_pred = reg.predict(X_space)
        low_pred = reg_low.predict(X_space)
        high_pred = reg_high.predict(X_space)

        plt.title(f'Aggregate {subname}: corr={corr:.5f}')
        
        plt.scatter(X_train, Y_train, color='black')
        plt.scatter(np.mean(df_damp, axis=1)[19930:19960], temp_atk_pts, color='red')

        plt.plot(X_space, mean_pred, color='green')
        plt.plot(X_space, low_pred, color='green')
        plt.plot(X_space, high_pred, color='green')

        plt.ylabel('Average temperature difference')
        plt.xlabel('Average damper position')
        #plt.xlabel('Total PVAV flow')

        #plt.show()
        plt.savefig(f'attack-massmodel-damp-{subname}.png')
        plt.close()

        z_idx += 1

    pdb.set_trace()

def main_train_dtrees():

    df_occ = load_csv('ZonePeopleOccupantCount.csv')

    for depth in range(2, 7):

        tree_errs = np.zeros((len(df_occ.columns), 3))

        for i in range(1, len(df_occ.columns)):

            sanitized_name = df_occ.columns[i][:-29].upper()
            save_name = sanitized_name.replace(' ', '')
            df_train = pd.read_csv(f'{save_name}_train.csv', index_col=0)

            X_train = df_train.iloc[:,:5].values
            Y_train = df_train.iloc[:,5].values

            clf = tree.DecisionTreeRegressor(max_depth=depth)
            clf = clf.fit(X_train, Y_train)

            mse = np.sqrt(np.mean((Y_train - clf.predict(X_train))**2))
            threshold = np.sqrt(np.quantile((Y_train - clf.predict(X_train))**2, 0.95))
            #print(f'{sanitized_name} train error: {mse}, 95th at {threshold}')
            
            X_atk = df_train.iloc[7738:7768,:5].values
            Y_atk = df_train.iloc[7738:7768,6].values
            atk_mse = np.sqrt(np.mean((Y_atk - clf.predict(X_atk))**2))
            #print(f'{sanitized_name} atk error: {atk_mse}')

            tree_errs[i, 0] = mse
            tree_errs[i, 1] = threshold
            tree_errs[i, 2] = atk_mse

            # fig, ax = plt.subplots(2, 1, figsize=(10, 10))
            # ax[0].plot(Y_train)
            # ax[1].plot(clf.predict(X_train))
            # fig.tight_layout()
            # plt.savefig(f'tree-prediction-{save_name}.pdf')
            # plt.close()

            # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            # ax.scatter(clf.predict(X_train), Y_train)
            # fig.tight_layout()
            # plt.savefig(f'tree-scatter-{save_name}.pdf')
            # plt.close()

            tree.plot_tree(clf, feature_names=df_train.columns[:6], fontsize=6)
            plt.savefig(f'tree-struct-{save_name}.pdf')
            plt.close()

        print(f'Average benign error: {np.mean(tree_errs[1:,0])}')
        print(f'Average benign threshold: {np.mean(tree_errs[1:,1])}')
        print(f'Average attack error: {np.mean(tree_errs[1:,2])}')
        
        num_higher = np.sum(tree_errs[1:,2] > tree_errs[1:,0])
        num_caught = np.sum(tree_errs[1:,2] > tree_errs[1:,1])
        print(f'Number caught: {num_caught} ({num_caught/(len(tree_errs)-1)}%)')
        print(f'Number where attack is higher: {num_higher} ({num_higher/(len(tree_errs)-1)}%)')

    pdb.set_trace()

def create_attack_trace(tempdiff):

    df_rawtemp = load_csv('ZoneTemperature.csv')
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    ax.plot(df_rawtemp.iloc[19910:19990, 33], color='blue', lw=3, label='normal')
    df_rawtemp.iloc[19930:19960, 1:] = df_rawtemp.iloc[19930:19960, 1:] + tempdiff
    ax.plot(df_rawtemp.iloc[19929:19961, 33], color='red', lw=3, label='spoofed')
    ax.set_title('Attack Region', fontsize=24)
    ax.set_xlabel('Time', fontsize=16)
    ax.set_ylabel('Zone Temperature', fontsize=16)
    ax.legend()
    plt.savefig('attack_example.pdf')

    df_rawtemp.to_csv('dataframes/ZoneTemperature_Atk.csv', index=False)

    pdb.set_trace()

def attack_models(config):

    df_massflow = load_csv('FanAirMassFlowRate.csv')
    df_damper = load_csv('ZoneAirTerminalVAVDamperPosition.csv')
    df_flow = load_csv('ZoneMechanicalVentilationMassFlowRate.csv')
    df_occ = load_csv('ZonePeopleOccupantCount.csv')
    df_rawtemp = load_csv('ZoneTemperature.csv')
    df_rawtemp_atk = load_csv('ZoneTemperature_Atk.csv')

    df = load_csv('building_data.csv')
    outside_temp = df.iloc[:, 20]
    exp_name = config['exp_name']

    building_open_idx = np.where(df['Operating Time'] == 'Yes')[0]
    detect_obj = np.zeros((len(df_damper.columns), 2))

    for i in range(1, len(df_damper.columns)):
        
        if i == 45:
            # Skip one defective zone
            continue
        
        open_idx = np.where(df_damper.iloc[:,i] > 0.201)[0]

        sanitized_name = df_damper.columns[i][:-40].upper()
        save_name = sanitized_name.replace(' ', '')

        df_train = df_damper.iloc[open_idx,i]
        df_out = (df_rawtemp.iloc[:,i] - outside_temp)[open_idx]

        X_train = df_train.values.reshape(-1, 1)
        Y_train = df_out.values

        corr = utils.get_corr(X_train.T, Y_train)
        print(f'{sanitized_name}: corr={corr}')

        reg_low, reg, reg_high = load_boosted_regression(name=f'model-col{i}')

        xmin = np.min(X_train)
        xmax = np.max(X_train)
        X_space = np.linspace(start=xmin, stop=xmax, num=1_000).reshape(-1, 1) 
        mean_pred = reg.predict(X_space)
        low_pred = reg_low.predict(X_space)
        high_pred = reg_high.predict(X_space)

        atk_pred = reg_high.predict(df_damper.iloc[19930:19960, i].values.reshape(-1, 1))
        atk_true = (df_rawtemp_atk.iloc[19930:19960, i] - outside_temp[19930:19960])
        print(f'Detected: {np.sum(atk_true.values > atk_pred)} out of 30')

        detect_obj[i, 0] = corr
        detect_obj[i, 1] = np.sum(atk_true.values > atk_pred)

        plt.title(f'{sanitized_name}: corr={corr:.4f}')
        plt.scatter(df_damper.iloc[building_open_idx, i], (df_rawtemp.iloc[:,i] - outside_temp)[building_open_idx], color='black')
        plt.plot(X_space, mean_pred, color='green')
        plt.plot(X_space, low_pred, color='green')
        plt.plot(X_space, high_pred, color='green')
    
        plt.ylabel('Relative temperature difference')
        plt.xlabel('Damper Position')

        #plt.scatter(df_damper.iloc[19930:19960, i], atk_true, color='red')
        plt.tight_layout()
        #plt.show()
        plt.savefig(f'model-{save_name}.png')
        plt.close()

    print(f'Total detected: {np.sum(detect_obj[:, 1] > 0)}')
    
    pop1 = np.abs(detect_obj[np.where(detect_obj[:, 1] > 0)[0]][:,0]) 
    pop2 = np.abs(detect_obj[np.where(detect_obj[:, 1] == 0)[0]][:,0]) 
    stat, pval = ss.f_oneway(pop1, pop2)
    print(f'ANOVA={stat:.3f} pval={pval:.5f}')

    pdb.set_trace()

if __name__ == '__main__':

    # main_pairwise_models({'exp_name': 'damper-tempdiff'})

    # create_attack_trace(tempdiff=3)
    #attack_models({'exp_name': 'damper-tempdiff'})
    # main_mass_models()

    # process_zone_data()
    main_train_dtrees()

    # main_linreg()

