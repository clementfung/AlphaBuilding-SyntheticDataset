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

def flow_total():

    df_massflow = load_csv('FanAirMassFlowRate.csv')
    df_damper = load_csv('ZoneAirTerminalVAVDamperPosition.csv')
    df_flow = load_csv('ZoneMechanicalVentilationMassFlowRate.csv')
    
    df = load_csv('building_data.csv')
    open_idx = np.where(df['Operating Time'] == 'Yes')[0]

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

    df_top = pd.DataFrame(flow_top)
    df_mid = pd.DataFrame(flow_mid)
    df_bot = pd.DataFrame(flow_bot)

    top_mass_normed = df_massflow.iloc[open_idx,1]
    sum_mass_normed = np.mean(df_top, axis=1)[open_idx]
    corr = utils.get_corr(top_mass_normed, sum_mass_normed)
    print(corr) 
    plt.scatter(top_mass_normed, sum_mass_normed)
    plt.title(f'Top Zone: corr={corr:.5f}')
    plt.ylabel('Sum of zone flow')
    plt.xlabel('Zone Mass Flow')
    plt.savefig('total-flow-top.png')
    plt.close()

    mid_mass_normed = df_massflow.iloc[open_idx,2]
    mean_mid_normed = np.mean(df_mid, axis=1)[open_idx]
    corr = utils.get_corr(mid_mass_normed, mean_mid_normed)
    print(corr) 
    plt.scatter(mid_mass_normed, mean_mid_normed)
    plt.title(f'Mid Zone: corr={corr:.5f}')
    plt.ylabel('Sum of zone flow')
    plt.xlabel('Zone Mass Flow')
    plt.savefig('total-flow-mid.png')
    plt.close()

    bot_mass_normed = df_massflow.iloc[open_idx,3]
    mean_bot_normed = np.mean(df_bot, axis=1)[open_idx]
    corr = utils.get_corr(bot_mass_normed, mean_bot_normed)
    print(corr) 
    plt.scatter(bot_mass_normed, mean_bot_normed)
    plt.title(f'Bottom Zone: corr={corr:.5f}')
    plt.ylabel('Sum of zone flow')
    plt.xlabel('Zone Mass Flow')
    plt.savefig('total-flow-bot.png')
    plt.close()

    pdb.set_trace()

def mean_damper():

    df_massflow = load_csv('FanAirMassFlowRate.csv')
    df_damper = load_csv('ZoneAirTerminalVAVDamperPosition.csv')
    df = load_csv('building_data.csv')
    open_idx = np.where(df['Operating Time'] == 'Yes')[0]

    flow_top = dict()
    flow_mid = dict()
    flow_bot = dict()

    for i in range(1, len(df_damper.columns)):
        sanitized_name = df_damper.columns[i][:-40].upper()

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
    corr = utils.get_corr(top_mass_normed, sum_mass_normed)
    print(corr) 
    plt.scatter(top_mass_normed, sum_mass_normed)
    plt.title(f'Top Zone: corr={corr:.5f}')
    plt.ylabel('Average damper')
    plt.xlabel('Zone Mass Flow')
    plt.savefig('total-damper-top.png')
    plt.close()

    mid_mass_normed = df_massflow.iloc[open_idx,2]
    mean_mid_normed = np.mean(df_mid, axis=1)[open_idx]
    corr = utils.get_corr(mid_mass_normed, mean_mid_normed)
    print(corr) 
    plt.scatter(mid_mass_normed, mean_mid_normed)
    plt.title(f'Mid Zone: corr={corr:.5f}')
    plt.ylabel('Average damper')
    plt.xlabel('Zone Mass Flow')
    plt.savefig('total-damper-mid.png')
    plt.close()

    bot_mass_normed = df_massflow.iloc[open_idx,3]
    mean_bot_normed = np.mean(df_bot, axis=1)[open_idx]
    corr = utils.get_corr(bot_mass_normed, mean_bot_normed)
    print(corr) 
    plt.scatter(bot_mass_normed, mean_bot_normed)
    plt.title(f'Bottom Zone: corr={corr:.5f}')
    plt.ylabel('Average damper')
    plt.xlabel('Zone Mass Flow')
    plt.savefig('total-damper-bot.png')
    plt.close()

    pdb.set_trace()

def mean_tempdiff():

    df_massflow = load_csv('FanAirMassFlowRate.csv')
    #df_damper = load_csv('ZoneAirTerminalVAVDamperPosition.csv')
    df_rawtemp = load_csv('ZoneTemperature.csv')
    df = load_csv('building_data.csv')
    open_idx = np.where(df['Operating Time'] == 'Yes')[0]
    outside_temp = df.iloc[:, 20]

    flow_top = dict()
    flow_mid = dict()
    flow_bot = dict()

    for i in range(1, len(df_rawtemp.columns)):
        sanitized_name = df_rawtemp.columns[i][:-40].upper()

        if 'TOP' in sanitized_name:
            flow_top[sanitized_name] = df_rawtemp.iloc[:,i].values
        if 'MID' in sanitized_name:
            flow_mid[sanitized_name] = df_rawtemp.iloc[:,i].values
        if 'BOT' in sanitized_name:
            flow_bot[sanitized_name] = df_rawtemp.iloc[:,i].values

    df_top = pd.DataFrame(flow_top)
    df_mid = pd.DataFrame(flow_mid)
    df_bot = pd.DataFrame(flow_bot)

    top_mass_normed = df_massflow.iloc[open_idx,1]
    sum_mass_normed = (outside_temp - np.mean(df_top, axis=1))[open_idx]
    corr = utils.get_corr(top_mass_normed, sum_mass_normed)
    print(corr) 
    plt.scatter(top_mass_normed, sum_mass_normed)
    plt.title(f'Top Zone: corr={corr:.5f}')
    plt.ylabel('Temperature difference')
    plt.xlabel('Zone Mass Flow')
    plt.savefig('total-tempdiff-top.png')
    plt.close()

    mid_mass_normed = df_massflow.iloc[open_idx,2]
    mean_mid_normed = (outside_temp - np.mean(df_mid, axis=1))[open_idx]
    corr = utils.get_corr(mid_mass_normed, mean_mid_normed)
    print(corr) 
    plt.scatter(mid_mass_normed, mean_mid_normed)
    plt.title(f'Mid Zone: corr={corr:.5f}')
    plt.ylabel('Temperature difference')
    plt.xlabel('Zone Mass Flow')
    plt.savefig('total-tempdiff-mid.png')
    plt.close()

    bot_mass_normed = df_massflow.iloc[open_idx,3]
    mean_bot_normed = (outside_temp - np.mean(df_bot, axis=1))[open_idx]
    corr = utils.get_corr(bot_mass_normed, mean_bot_normed)
    print(corr) 
    plt.scatter(bot_mass_normed, mean_bot_normed)
    plt.title(f'Bottom Zone: corr={corr:.5f}')
    plt.ylabel('Temperature difference')
    plt.xlabel('Zone Mass Flow')
    plt.savefig('total-tempdiff-bot.png')
    plt.close()

    pdb.set_trace()

def damper_tempdiff():

    df_massflow = load_csv('FanAirMassFlowRate.csv')
    df_damper = load_csv('ZoneAirTerminalVAVDamperPosition.csv')
    df_rawtemp = load_csv('ZoneTemperature.csv')
    df = load_csv('building_data.csv')
    open_idx = np.where(df['Operating Time'] == 'Yes')[0]
    outside_temp = df.iloc[:, 20]

    damp_top = dict()
    damp_mid = dict()
    damp_bot = dict()
    temp_top = dict()
    temp_mid = dict()
    temp_bot = dict()

    for i in range(1, len(df_rawtemp.columns)):
        sanitized_name = df_rawtemp.columns[i][:-40].upper()

        if 'TOP' in sanitized_name:
            damp_top[sanitized_name] = df_damper.iloc[:,i].values
            temp_top[sanitized_name] = df_rawtemp.iloc[:,i].values
        if 'MID' in sanitized_name:
            damp_mid[sanitized_name] = df_damper.iloc[:,i].values
            temp_mid[sanitized_name] = df_rawtemp.iloc[:,i].values
        if 'BOT' in sanitized_name:
            damp_bot[sanitized_name] = df_damper.iloc[:,i].values
            temp_bot[sanitized_name] = df_rawtemp.iloc[:,i].values

    tf_top = pd.DataFrame(temp_top)
    tf_mid = pd.DataFrame(temp_mid)
    tf_bot = pd.DataFrame(temp_bot)

    df_top = pd.DataFrame(damp_top)
    df_mid = pd.DataFrame(damp_mid)
    df_bot = pd.DataFrame(damp_bot)

    top_damp_normed = np.mean(df_top, axis=1)[open_idx]
    top_temp_normed = (outside_temp - np.mean(tf_top, axis=1))[open_idx]

    corr = utils.get_corr(top_damp_normed, top_temp_normed)
    print(corr) 
    plt.scatter(top_damp_normed, top_temp_normed)
    # plt.scatter(top_damp_normed.iloc[attack_point], top_temp_normed.iloc[attack_point], color='green')
    # plt.scatter(top_damp_normed.iloc[attack_point], top_temp_normed.iloc[attack_point] * 0.5, color='green')
    plt.title(f'Top Zone: corr={corr:.5f}')
    plt.ylabel('Temperature difference')
    plt.xlabel('Average Damper')
    plt.savefig('total-damper-tempdiff-top.png')
    plt.close()

    mid_damp_normed = np.mean(df_mid, axis=1)[open_idx]
    mid_temp_normed = (outside_temp - np.mean(tf_mid, axis=1))[open_idx]
    corr = utils.get_corr(mid_damp_normed, mid_temp_normed)
    print(corr) 
    plt.scatter(mid_damp_normed, mid_temp_normed)
    plt.title(f'Mid Zone: corr={corr:.5f}')
    plt.ylabel('Temperature difference')
    plt.xlabel('Average Damper')
    plt.savefig('total-damper-tempdiff-mid.png')
    plt.close()

    bot_damp_normed = np.mean(df_bot, axis=1)[open_idx]
    bot_temp_normed = (outside_temp - np.mean(tf_bot, axis=1))[open_idx]
    corr = utils.get_corr(bot_damp_normed, bot_temp_normed)
    print(corr) 
    plt.scatter(bot_damp_normed, bot_temp_normed)
    plt.title(f'Bottom Zone: corr={corr:.5f}')
    plt.ylabel('Temperature difference')
    plt.xlabel('Average Damper')
    plt.savefig('total-damper-tempdiff-bot.png')
    plt.close()

    pdb.set_trace()

def damper_coupling():

    df_massflow = load_csv('FanAirMassFlowRate.csv')
    df_damper = load_csv('ZoneAirTerminalVAVDamperPosition.csv')
    df_rawtemp = load_csv('ZoneTemperature.csv')
    df = load_csv('building_data.csv')
    open_idx = np.where(df['Operating Time'] == 'Yes')[0]
    outside_temp = df.iloc[:, 20]

    damp_top = dict()
    damp_mid = dict()
    damp_bot = dict()

    for i in range(1, len(df_rawtemp.columns)):
        sanitized_name = df_rawtemp.columns[i][:-40].upper()

        if 'TOP' in sanitized_name:
            damp_top[sanitized_name] = df_damper.iloc[:,i].values
        if 'MID' in sanitized_name:
            damp_mid[sanitized_name] = df_damper.iloc[:,i].values
        if 'BOT' in sanitized_name:
            damp_bot[sanitized_name] = df_damper.iloc[:,i].values

    df_top = pd.DataFrame(damp_top)
    df_mid = pd.DataFrame(damp_mid)
    df_bot = pd.DataFrame(damp_bot)

    # Test each feature as support
    is_open = df_top > 0.2001

    for i in range(len(is_open.columns)):
        support = is_open[is_open.iloc[:, i]]
        print(f'Support for VAV column {is_open.columns[i]}: {len(support)} samples')
        print(np.mean(support, axis=0))

        pdb.set_trace()


if __name__ == '__main__':
    
    damper_coupling()

    flow_total()
    mean_damper()
    mean_tempdiff()
    damper_tempdiff()
