import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt 
import seaborn as sns
import pdb

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

def main():

    # Plot some exploratory information
    df = load_csv('ZonePeopleOccupantCount.csv')
    print(f'Plotting {df.columns[6]}')
    plot_annual(df.iloc[:,6].values, title='Occupants', filename='occupants.pdf')

    df = load_csv('ZoneThermostatCoolingSetpointTemperature.csv')
    print(f'Plotting {df.columns[18]}')
    plot_annual(df.iloc[:,18].values, title='Cooling Setpoint', filename='cooling-setpoint.pdf')
    
    df = load_csv('ZoneThermostatHeatingSetpointTemperature.csv')
    print(f'Plotting {df.columns[18]}')
    plot_annual(df.iloc[:,18].values, title='Heating Setpoint', filename='heating-setpoint.pdf')
    
    df = load_csv('ZoneAirTerminalVAVDamperPosition.csv')
    print(f'Plotting {df.columns[18]}')
    plot_annual(df.iloc[:,18].values, title='VAV Damper', filename='vav-damper.pdf')

    df = load_csv('ZoneMeanAirTemperature.csv')
    print(f'Plotting {df.columns[18]}')
    plot_annual(df.iloc[:,18].values, title='Air Temperature', filename='air-temp.pdf')

    df = load_csv('ZoneElectricEquipmentElectricPower.csv')
    print(f'Plotting {df.columns[6]}')
    plot_annual(df.iloc[:,6].values, title='Electric Power', filename='elec-power.pdf')

    df = load_csv('building_data.csv')
    print(f'Plotting {df.columns[19]}')
    plot_annual(df.iloc[:,19].values, title='Total HVAC Electricity Usage', filename='hvac-power.pdf')

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

if __name__ == '__main__':
    main()