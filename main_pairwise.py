import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt 
import utils
import seaborn as sns
import pdb
import networkx as nx

from utils import load_csv

WINTER = np.arange(0, 13140)
SPRING = np.arange(13140, 26280)
SUMMER = np.arange(26280, 39420)
FALL = np.arange(39420, 52560)
YEAR = np.arange(0, 52560)

def main():

    df_vav = load_csv('ZoneAirTerminalVAVDamperPosition.csv')
    df_flow = load_csv('ZoneMechanicalVentilationMassFlowRate.csv')

    'ZoneMechanicalVentilationMassFlowRate.csv'
    'ZoneAirTerminalVAVDamperPosition.csv'
    'ZoneMeanAirTemperature.csv'

    df = load_csv('building_data.csv')
    open_idx = np.where(df['Operating Time'] == 'Yes')[0]
    
    fig, ax = plt.subplots(3, 2, figsize=(12, 16))

    for i in range(6):
        
        starter = 20 # 14 or 20
        col_idx = starter + i
        
        x = i // 2
        y = i % 2

        print(f'Collecting {df_vav.columns[col_idx]} and {df_flow.columns[col_idx]}')
        
        location = df_vav.columns[col_idx][:14]
        ax[x, y].set_title(f'{location}', fontsize=14)
        ax[x, y].scatter(df_vav.iloc[open_idx,col_idx].values, df_flow.iloc[open_idx,col_idx].values)
        #ax[x, y].set_xlabel('Zone VAV Damper', fontsize=14)
        #ax[x, y].set_ylabel('Zone Airflow', fontsize=14)
    
    ax[0, 0].set_ylabel('Zone Airflow', fontsize=14)
    ax[1, 0].set_ylabel('Zone Airflow', fontsize=14)
    ax[2, 0].set_ylabel('Zone Airflow', fontsize=14)

    ax[2, 0].set_xlabel('Zone VAV Damper', fontsize=14)
    ax[2, 1].set_xlabel('Zone VAV Damper', fontsize=14)

    fig.tight_layout()
    plt.savefig(f'corridor-flow.pdf')
    plt.close()

def explore_graph():

    df_massflow = load_csv('FanAirMassFlowRate.csv')
    df_damper = load_csv('ZoneAirTerminalVAVDamperPosition.csv')
    df_flow = load_csv('ZoneMechanicalVentilationMassFlowRate.csv')
    df = load_csv('building_data.csv')
    open_idx = np.where(df['Operating Time'] == 'Yes')[0]

    graph = nx.read_gml('brick-graph.gml')

    print('========= LEVEL 1 ======================')

    for i in range(1, len(df_massflow.columns)):
        sanitized_name = df_massflow.columns[i][:-33].upper()
        print(f'For node: {sanitized_name}')
        print(f'Below: {graph[sanitized_name]}')
        print(f'Above: {graph.pred[sanitized_name]}')

    print('========= LEVEL 2 ======================')

    for i in range(1, len(df_damper.columns)):
        sanitized_name = df_damper.columns[i][:-40].upper()
        print(f'For node: {sanitized_name}')
        print(f'Below: {graph[sanitized_name]}')
        print(f'Above: {graph.pred[sanitized_name]}')

    print('========= LEVEL 3 ======================')

    for i in range(1, len(df_flow.columns)):
        sanitized_name = df_flow.columns[i][:-49].upper()
        print(f'For node: {sanitized_name}')
        print(f'Below: {graph[sanitized_name]}')
        print(f'Above: {graph.pred[sanitized_name]}')

def explore_pairwise_relations():

    df_damper = load_csv('ZoneAirTerminalVAVDamperPosition.csv')
    df_rawtemp = load_csv('ZoneTemperature.csv')
    df = load_csv('building_data.csv')
    open_idx = np.where(df['Operating Time'] == 'Yes')[0]
    outside_temp = df.iloc[:, 20]

    # line_idx = np.where((df_damper.iloc[:,24] > 0.4) & (df_flow.iloc[:,24] > 0.03)) 

    for i in range(1, len(df_damper.columns)):
        
        sanitized_name = df_damper.columns[i][:-40].upper()
        temp_name = df_rawtemp.columns[i].upper()

        temp_diff = (df_rawtemp.iloc[:,i] - outside_temp)[open_idx]

        corr = utils.get_corr(df_damper.iloc[open_idx,i], temp_diff)
        print(f'{sanitized_name} col-{i}: {corr:.5f}')

        fig, ax = plt.subplots(1, 1)
        ax.set_title(f'{sanitized_name}: corr={corr:.3f}')
        ax.scatter(df_damper.iloc[open_idx,i], temp_diff, color='black')
        plt.savefig(f'zone-damper-tempdiff-col{i}.png')
        plt.close()

        print(f'{sanitized_name} vs {temp_name}: corr={corr:.5f}')

def explore_setpoints():

    df_heatset = load_csv('ZoneThermostatHeatingSetpointTemperature.csv') 
    df_coolset = load_csv('ZoneThermostatCoolingSetpointTemperature.csv') 
    df_damper = load_csv('ZoneAirTerminalVAVDamperPosition.csv')
    df_rawtemp = load_csv('ZoneTemperature.csv')
    df = load_csv('building_data.csv')
    open_idx = np.where(df['Operating Time'] == 'Yes')[0]
    outside_temp = df.iloc[:, 20]

    #for i in range(1, len(df_damper.columns)):
    for i in range(26, 30):
        
        sanitized_name = df_damper.columns[i][:-40].upper()
        temp_name = df_rawtemp.columns[i].upper()

        temp_set = (df_rawtemp.iloc[:,i] - df_heatset.iloc[:,i])[open_idx]
        corr = utils.get_corr(df_damper.iloc[open_idx,i], temp_set)

        fig, ax = plt.subplots(1, 1)
        ax.set_title(f'{sanitized_name}: corr={corr:.3f}')
        ax.scatter(df_damper.iloc[open_idx,i], temp_set, color='black')
        plt.show()
        plt.close()

        print(f'{sanitized_name} vs {temp_name}: corr={corr:.5f}')

if __name__ == '__main__':
    
    #explore_graph()
    explore_pairwise_relations()
    explore_setpoints()