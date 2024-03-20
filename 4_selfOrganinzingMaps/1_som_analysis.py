# %%
import warnings
import numpy as np
import pandas as pd
from modules.bootstrap import *
import matplotlib.pyplot as plt
from modules.minisom import MiniSom
from modules.bootstrap import bootstrap
from sklearn.preprocessing import MinMaxScaler

# ignore warnings
warnings.filterwarnings('ignore')

# %%
# Load the data
data = pd.read_csv('../0_data/som_studies/data_rotaA.csv', index_col=[0])

# %%
# define the columns that will be used in the PCA
x_cols = ['x_glic_et', 'x_cell_glic', 'eta_cell_orgsolv', 'capex_f1_A',
          'raw_mat_price', 'enzyme_load', 'lign_price', 'et_price']

# define the target column
target = 'msp'

# sample x and y
x = data[x_cols]
y = data[[target]]

# %%
# scale the features and fit SOM
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(x)

x_sc = scaler.transform(x)
# %%
# determine the grid size
N = int(np.sqrt(5 * np.sqrt(x_sc.shape[0]))) 
print(f'Lado do Mapa: {N}')

# %%
 # fit the model - the size of the map was chosen based 
# on the recommendations of the creators of the packages
som = MiniSom(x=N, y=N, input_len=x_sc.shape[1],
              sigma=N-1, learning_rate=0.5)
som.random_weights_init(x_sc)
som.train_random(x_sc, num_iteration=200)

# %%
# calculate the distance map and find the 
# coordinates of the winning neurons
dists = som.distance_map()

# get winning neurons for each sample
dist_list = []
winning = []
for i in range(x_sc.shape[0]):

    # get winning neuron coordinates
    coord = som.winner(x=x_sc[i])
    winning.append(coord)

    # get distance
    dist_list.append(dists[coord])

# %%
# add the winning neurons to the data
x_data = pd.DataFrame(scaler.inverse_transform(x_sc), columns=x_cols)
x_data['winning'] = winning

# append the target column
x_data[target] = y

# %%
# analyze the distribution of the winning neurons
x_data['winning'].value_counts()

# %%
# extract TOP N winning neurons
top_n = 3
top_n_neurons = x_data['winning'].value_counts().index[:top_n]
# %%
# sample data for top winning neurons
x_data_top = x_data[x_data['winning'].isin(top_n_neurons)]
# %%
# calculate the bootstrapped mean and 95% confidence interval
# for the target variable
boot_dict = {}
for col in x_cols+[target]:
    boot_dict[col] = {}
    for c in x_data_top['winning'].unique():
        boot_dict[col][c] = bootstrap(x_data_top[x_data_top['winning'] == c][col])

# %%
# plot all the bootstrapped samples
for col in x_cols+[target]:
    plot_bootstrap_samples(boot_dict, col)

# %%
# analyze the effect of the winning neuron influence
influence = range(1, N)
lowest_msp = []
for n in influence:
    msp_i = []
    for _ in range(100):

        som = MiniSom(x=N, y=N, input_len=x_sc.shape[1],
                sigma=n, learning_rate=0.5)
        som.random_weights_init(x_sc)
        som.train_random(x_sc, num_iteration=200)

        # calculate the distance map and find the 
        # coordinates of the winning neurons
        dists = som.distance_map()

        # get winning neurons for each sample
        dist_list = []
        winning = []
        for i in range(x_sc.shape[0]):

            # get winning neuron coordinates
            coord = som.winner(x=x_sc[i])
            winning.append(coord)

            # get distance
            dist_list.append(dists[coord])

        # add the winning neurons to the data
        x_data = pd.DataFrame(scaler.inverse_transform(x_sc), columns=x_cols)
        x_data['winning'] = winning

        # append the target column
        x_data[target] = y

        top_n_neurons = x_data['winning'].value_counts().index[:top_n]

        # sample data for top winning neurons
        x_data_top = x_data[x_data['winning'].isin(top_n_neurons)]

        # append the mean of msp to the list
        msp_i.append(x_data_top[target].mean())

    # get the winning neurons with the lowest msp
    lowest_msp.append(np.mean(msp_i))

# %%
# plot the influence of the winning neurons
plt.plot(influence, lowest_msp, 'o-')
plt.xlabel('Influence Radius', size=14)
plt.ylabel('Lowest MSP', size=14)
plt.title('Influence Radius vs. Lowest MSP', size=16)
plt.grid(True, alpha=0.2)
plt.show()

# %%
# define the final model of the SOM
min_msp_list = []
x_list = {}
for _ in range(100):
    som = MiniSom(x=N, y=N, input_len=x_sc.shape[1],
                sigma=1, learning_rate=0.5)
    som.random_weights_init(x_sc)
    som.train_random(x_sc, num_iteration=200)

    # calculate the distance map and find the 
    # coordinates of the winning neurons
    dists = som.distance_map()

    # get winning neurons for each sample
    dist_list = []
    winning = []
    for i in range(x_sc.shape[0]):

        # get winning neuron coordinates
        coord = som.winner(x=x_sc[i])
        winning.append(coord)

        # get distance
        dist_list.append(dists[coord])

    # add the winning neurons to the data
    x_data = pd.DataFrame(scaler.inverse_transform(x_sc), columns=x_cols)
    x_data['winning'] = winning

    # append the target column
    x_data[target] = y

    top_n_neurons = x_data['winning'].value_counts().index[:top_n]

    # sample data for top winning neurons
    x_data_top = x_data[x_data['winning'].isin(top_n_neurons)]

    # determine the winning neuron
    agg = x_data_top.groupby(['winning'])[target].mean()
    min_msp = agg.min()
    winning_neuron = agg[agg == min_msp].index[0]

    # append the data to the list
    min_msp_list.append(bootstrap(x_data_top[x_data_top['winning'] == winning_neuron][target]))
    for col in x_cols:
        x_list[col] = []
        x_list[col].append(bootstrap(x_data_top[x_data_top['winning'] == winning_neuron][col]))


# %%
# transform the data into a single array
min_msp_list = np.array(min_msp_list)
min_msp_list = min_msp_list.reshape(-1, 1)

for col in x_cols:
    x_list[col] = np.array(x_list[col])
    x_list[col] = x_list[col].reshape(-1, 1)

# %%
# plot distribution of the lowest MSP
plot_bootstrap_samples({'msp': {1: min_msp_list}}, 'msp')

# %%
for col in x_cols:
    plot_bootstrap_samples({col: {1: x_list[col]}}, col)