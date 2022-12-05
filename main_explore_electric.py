import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt 
import seaborn as sns
import pdb
import networkx as nx

def load_csv(filename='building_data.csv'):
    df = pd.read_csv(f'dataframes/{filename}')
    return df

def plot_annual(column, title='Sample Title', filename='temp.pdf'):

    # Each row is a day, columns are hours
    daily = column.reshape((365, 144))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    img = ax.imshow(daily.T)
    ax.set_yticks(np.arange(0, 144, 6))
    ax.set_xticks(np.arange(0, 350, 30))
    ax.set_yticklabels(np.arange(1, 25))
    ax.set_xticklabels(np.arange(1, 13))
    ax.set_title(title, fontsize=24)

    fig.colorbar(img, location='bottom', orientation='horizontal')
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():

    # Plot some exploratory information
    df = load_csv('ZonePeopleOccupantCount.csv')
    print(f'Plotting {df.columns[6]}')
    plot_annual(df.iloc[:,6].values, title='Occupants', filename='occupants.pdf')

    df = load_csv('ZoneAirTerminalVAVDamperPosition.csv')
    print(f'Collecting {df.columns[18]}')
    df_heatset = df.copy()

    'ZoneMechanicalVentilationMassFlowRate.csv'
    'ZoneAirTerminalVAVDamperPosition.csv'
    'ZoneMeanAirTemperature.csv'

    df = load_csv('ZoneMechanicalVentilationMassFlowRate.csv')
    print(f'Collecting {df.columns[18]}')
    df_airtemp = df.copy()

    df = load_csv('building_data.csv')
    print(f'Plotting building data')
    plot_annual(df.iloc[:,19].values, title='Total HVAC Electricity Usage', filename='total-hvac.pdf')
    plot_annual(df.iloc[:,1].values, title=df.columns[1], filename='total-occ.pdf')
    plot_annual(df.iloc[:,2].values, title=df.columns[2], filename='total-cool.pdf')
    plot_annual(df.iloc[:,5].values, title=df.columns[5], filename='total-heat.pdf')
    plot_annual(df.iloc[:,6].values, title=df.columns[6], filename='total-gas.pdf')

    open_idx = np.where(df['Operating Time'] == 'Yes')[0]
    plt.scatter(df_heatset.iloc[open_idx,18].values, df_airtemp.iloc[open_idx,18].values)
    plt.show()
    plt.close()
    
    pdb.set_trace()

def explore_graph():

    df = load_csv('building_data.csv')
    open_idx = np.where(df['Operating Time'] == 'Yes')[0]
    graph = nx.read_gml('brick-graph.gml')

def explore_pairwise_relations():

    df_zone_lights = load_csv('ZoneLightsElectricPower.csv')
    
    df_zone_eq = load_csv('ZoneElectricEquipmentElectricPower.csv')
    df_zone_temp = load_csv('ZoneTemperature.csv')

    df = load_csv('building_data.csv')
    open_idx = np.where(df['Operating Time'] == 'Yes')[0]
    outside_temp = df.iloc[:, 20]
    total_electric = df.iloc[:, 16]
    total_energy = df.iloc[:, 18]

    plt.scatter(df_zone_eq.iloc[open_idx, 15], df_zone_temp.iloc[open_idx, 15] - outside_temp[open_idx])
    plt.show()
    plt.close()

    plt.scatter(np.sum(df_zone_eq.iloc[open_idx, 1:], axis=1), total_electric.iloc[open_idx])
    plt.show()
    plt.close()

    plt.scatter(np.sum(df_zone_eq.iloc[open_idx, 1:], axis=1), total_energy.iloc[open_idx])
    plt.show()
    plt.close()

    pdb.set_trace()

if __name__ == '__main__':
    explore_pairwise_relations()