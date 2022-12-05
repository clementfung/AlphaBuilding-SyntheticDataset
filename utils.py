import numpy as np
import pandas as pd
import pickle
import pdb
import matplotlib.pyplot as plt

def load_csv(filename='building_data.csv'):
    df = pd.read_csv(f'dataframes/{filename}')
    return df

def get_open_idx(building_data_df, idx_range):
    return np.where(building_data_df.iloc[idx_range, -2] == 'Yes')[0]

def get_corr(X1, X2):
    corr = np.corrcoef(X1, X2)[0, 1]
    return corr

# Scale vector
def scale_vec(vec):
    new_vec = vec - np.min(vec)
    new_vec /= np.max(new_vec)
    return new_vec

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

if __name__ == '__main__':

    df = load_csv('SystemNodeTemperature.csv')
    dftemp = load_csv('ZoneMeanAirTemp65.csv')
    nodes = pickle.load(open('nodes.pkl', 'rb'))
    node_idxs = [0]

    for i in range(len(df.columns)):
        col_name = df.columns[i]
        node_name = col_name[:-27]
        
        if node_name in nodes:
            print(node_name)
            node_idxs.append(i)

    df2 = df.iloc[:, node_idxs]
    df2.to_csv('SystemNodeTemp65.csv', index=False)

    pdb.set_trace()
