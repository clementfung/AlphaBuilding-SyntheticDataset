import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt 
import seaborn as sns
import pdb
import networkx as nx

def load_csv(filename='building_data.csv'):
    df = pd.read_csv(filename)
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

def airflowplots():

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

def plot_pngs():
    df_col = load_csv()
    col_names = df_col.columns

    # Skip first column
    for i in range(1, 9):
        column = col_names[i]
        
        plt.plot(df_col[column])
        plt.ylabel(column)
        plt.savefig(f'{column.replace(" ", "")}.png')
        plt.close()

def explore_graph():

    df_massflow = load_csv('FanAirMassFlowRate.csv')
    df_damper = load_csv('ZoneAirTerminalVAVDamperPosition.csv')
    df_flow = load_csv('ZoneMechanicalVentilationMassFlowRate.csv')

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

    pdb.set_trace()

if __name__ == '__main__':
    explore_graph()