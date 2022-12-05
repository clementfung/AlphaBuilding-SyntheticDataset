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


def explore_flow_sums():

    df_massflow = load_csv('FanAirMassFlowRate.csv')
    df_damper = load_csv('ZoneAirTerminalVAVDamperPosition.csv')
    df_flow = load_csv('ZoneMechanicalVentilationMassFlowRate.csv')
    df_zonetemp = load_csv('ZoneMechanicalVentilationMassFlowRate.csv')
    
    df = load_csv('building_data.csv')
    open_idx = np.where(df['Operating Time'] == 'Yes')[0]

    flow_top = dict()
    flow_mid = dict()
    flow_bot = dict()

    for i in range(1, len(df_flow.columns)):
        sanitized_name = df_flow.columns[i][:-49].upper()

        if 'TOP' in sanitized_name:
            flow_top[sanitized_name] = df_damper.iloc[:,i].values
        if 'MID' in sanitized_name:
            flow_mid[sanitized_name] = df_damper.iloc[:,i].values
        if 'BOT' in sanitized_name:
            flow_bot[sanitized_name] = df_damper.iloc[:,i].values

    df_top = pd.DataFrame(flow_top)
    df_mid = pd.DataFrame(flow_mid)
    df_bot = pd.DataFrame(flow_bot)

    top_mass_normed = df_massflow.iloc[open_idx,1]
    sum_mass_normed = np.mean(df_top, axis=1)[open_idx]

    plt.plot(np.arange(len(open_idx)), top_mass_normed, label='Total VAV')
    plt.plot(np.arange(len(open_idx)), sum_mass_normed, label='Mean dampers')
    plt.legend()
    plt.show()
    plt.close()

    corr = utils.get_corr(top_mass_normed, sum_mass_normed)
    print(corr) 
    plt.scatter(top_mass_normed, sum_mass_normed)
    plt.show()
    plt.close()

    plt.plot(np.arange(len(open_idx)), df_massflow.iloc[open_idx,2], label='Total VAV')
    plt.plot(np.arange(len(open_idx)), np.mean(df_mid, axis=1)[open_idx], label='Mean dampers')
    plt.savefig('total_mid.png')
    plt.close()

    plt.plot(np.arange(len(open_idx)), df_massflow.iloc[open_idx,3], label='Total VAV')
    plt.plot(np.arange(len(open_idx)), np.mean(df_bot, axis=1)[open_idx], label='Mean dampers')
    plt.savefig('total_bot.png')
    plt.close()

    pdb.set_trace()

def explore_temp_sums():

    df_massflow = load_csv('FanAirMassFlowRate.csv')
    df_damper = load_csv('ZoneAirTerminalVAVDamperPosition.csv')
    df_flow = load_csv('ZoneMechanicalVentilationMassFlowRate.csv')
    df_zonetemp = load_csv('ZoneTemperature.csv')
    
    df = load_csv('building_data.csv')
    open_idx = np.where(df['Operating Time'] == 'Yes')[0]

    flow_top = dict()
    flow_mid = dict()
    flow_bot = dict()

    for i in range(1, len(df_flow.columns)):
        sanitized_name = df_flow.columns[i][:-49].upper()

        if 'TOP' in sanitized_name:
            flow_top[sanitized_name] = df_zonetemp.iloc[:,i].values
        if 'MID' in sanitized_name:
            flow_mid[sanitized_name] = df_zonetemp.iloc[:,i].values
        if 'BOT' in sanitized_name:
            flow_bot[sanitized_name] = df_zonetemp.iloc[:,i].values

    df_flow_top = pd.DataFrame(flow_top)
    df_flow_mid = pd.DataFrame(flow_mid)
    df_flow_bot = pd.DataFrame(flow_bot)
    outside_temp = df.iloc[:, 20]

    top_mass_normed = df_massflow.iloc[open_idx,1]
    mean_top_normed = (outside_temp - np.mean(df_flow_top, axis=1))[open_idx]

    plt.plot(np.arange(len(open_idx)), top_mass_normed, label='Total VAV')
    plt.plot(np.arange(len(open_idx)), mean_top_normed, label='Mean temperature difference')
    plt.legend()
    plt.show()
    #plt.savefig('total_top.png')
    plt.close()

    plt.scatter(top_mass_normed, mean_top_normed)
    plt.show()
    plt.close()

    corr = utils.get_corr(top_mass_normed, mean_top_normed)
    print(corr) 

    mid_mass_normed = df_massflow.iloc[open_idx,2]
    mean_mid_normed = (outside_temp - np.mean(df_flow_top, axis=1))[open_idx]

    plt.plot(np.arange(len(open_idx)), mid_mass_normed, label='Total VAV')
    plt.plot(np.arange(len(open_idx)), mean_mid_normed, label='Mean temperature difference')
    plt.show()
    #plt.savefig('total_mid.png')
    plt.close()

    corr = utils.get_corr(mid_mass_normed, mean_mid_normed)
    print(corr) 
    plt.scatter(mid_mass_normed, mean_mid_normed)
    plt.show()
    plt.close()

    bot_mass_normed = df_massflow.iloc[open_idx,3]
    mean_bot_normed = (outside_temp - np.mean(df_flow_bot, axis=1))[open_idx]

    plt.plot(np.arange(len(open_idx)), bot_mass_normed, label='Total VAV')
    plt.plot(np.arange(len(open_idx)), mean_bot_normed, label='Mean temperature difference')
    plt.show()
    #plt.savefig('total_bot.png')
    plt.close()

    corr = utils.get_corr(bot_mass_normed, mean_bot_normed)
    print(corr)
    plt.scatter(bot_mass_normed, mean_bot_normed)
    plt.show()
    plt.close() 

    pdb.set_trace()

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

def explore_first():

    df_massflow = load_csv('FanAirMassFlowRate.csv')
    df_damper = load_csv('ZoneAirTerminalVAVDamperPosition.csv')
    df_flow = load_csv('ZoneMechanicalVentilationMassFlowRate.csv')
    df_heatset = load_csv('ZoneThermostatHeatingSetpointTemperature.csv')
    df_coolset = load_csv('ZoneThermostatCoolingSetpointTemperature.csv')

    df = load_csv('building_data.csv')
    open_idx = np.where(df['Operating Time'] == 'Yes')[0]

    pdb.set_trace()

    flow_top = dict()
    flow_mid = dict()
    flow_bot = dict()

    for i in range(1, len(df_flow.columns)):
        sanitized_name = df_flow.columns[i][:-49].upper()

        if 'TOP' in sanitized_name:
            flow_top[sanitized_name] = df_flow.iloc[:,i].values
        if 'MID' in sanitized_name:
            flow_mid[sanitized_name] = df_flow.iloc[:,i].values
        if 'BOT' in sanitized_name:
            flow_bot[sanitized_name] = df_flow.iloc[:,i].values

    df_flow_top = pd.DataFrame(flow_top)
    df_flow_mid = pd.DataFrame(flow_mid)
    df_flow_bot = pd.DataFrame(flow_bot)

    fig, ax = plt.subplots(2, 1, figsize=(15, 6))

    ax[0].plot(np.sum(df_flow_top.iloc[open_idx].iloc[:2000], axis=1))
    ax[1].plot(np.sum(df_damper.iloc[open_idx].iloc[:2000], axis=1))
    ax[2].plot(df_massflow.iloc[open_idx].iloc[:2000, 1])

    ax[0].plot(np.sum(df_flow_bot.iloc[open_idx].iloc[:2000], axis=1))
    ax[1].plot(np.sum(df_damper.iloc[open_idx].iloc[:2000], axis=1))
    ax[2].plot(df_massflow.iloc[open_idx].iloc[:2000, 3])

    ax[0].plot(np.sum(df_flow_mid.iloc[open_idx].iloc[:2000], axis=1))
    ax[1].plot(np.sum(df_damper.iloc[open_idx].iloc[:2000], axis=1))
    ax[2].plot(df_massflow.iloc[open_idx].iloc[:2000, 2])

    pdb.set_trace()

def explore_pairwise_relations():

    df_massflow = load_csv('FanAirMassFlowRate.csv')
    df_flow = load_csv('ZoneMechanicalVentilationMassFlowRate.csv')

    df_damper = load_csv('ZoneAirTerminalVAVDamperPosition.csv')
    df_rawtemp = load_csv('ZoneTemperature.csv')
    df_airtemp = load_csv('ZoneMeanAirTemp65.csv')
    df = load_csv('building_data.csv')

    # line_idx = np.where((df_damper.iloc[:,24] > 0.4) & (df_flow.iloc[:,24] > 0.03)) 

    for i in range(1, len(df_damper.columns)):
        sanitized_name = df_damper.columns[i][:-40].upper()
        print(sanitized_name)

        fig, ax = plt.subplots(2, 2)

        ax[0, 0].set_title(sanitized_name)
        ax[0, 1].set_title(sanitized_name)

        winter_open = utils.get_open_idx(df, WINTER)
        spring_open = utils.get_open_idx(df, SPRING)
        summer_open = utils.get_open_idx(df, SUMMER)
        fall_open = utils.get_open_idx(df, FALL)
        open_idx = utils.get_open_idx(df, YEAR)

        ax[0, 0].scatter(df_damper.iloc[WINTER, i].iloc[winter_open], df_rawtemp.iloc[WINTER, i].iloc[winter_open], color='blue')
        ax[0, 1].scatter(df_damper.iloc[SPRING, i].iloc[spring_open], df_rawtemp.iloc[SPRING, i].iloc[spring_open], color='green')
        ax[1, 0].scatter(df_damper.iloc[SUMMER, i].iloc[summer_open], df_rawtemp.iloc[SUMMER, i].iloc[summer_open], color='red')
        ax[1, 1].scatter(df_damper.iloc[FALL, i].iloc[fall_open], df_rawtemp.iloc[FALL, i].iloc[fall_open], color='orange')
        
        plt.savefig(f'pairwise{sanitized_name}-raw-season.png')
        plt.close()

        fig, ax = plt.subplots(1, 1)
        ax.set_title(sanitized_name)
        ax.scatter(df_damper.iloc[open_idx,i], df_rawtemp.iloc[open_idx,i], color='black')        
        plt.savefig(f'pairwise{sanitized_name}-raw-full.png')
        plt.close()

if __name__ == '__main__':
    
    #explore_pairwise_relations()
    #explore_graph()
    explore_flow_sums()
