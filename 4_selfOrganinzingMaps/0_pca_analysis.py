# %%
import warnings
import warnings
import pandas as pd
from modules.bootstrap import *
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

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

# define the PCA pipeline
pca = Pipeline([
    ('scaler', RobustScaler()),
    ('pca', PCA(n_components=2, random_state=2))
])

# fit PCA
pca.fit(x)

# %%

# analyze the explained variance
plt.plot(range(2), pca['pca'].explained_variance_ratio_.cumsum(), 'o-')
plt.xlabel('No. Component', size=14)
plt.ylabel('Cumulated Explained Variance', size=14)
plt.title('Variance Retention in each Component', size=16)
plt.grid(True, alpha=0.2)
plt.xticks(ticks=range(2), labels=['1', '2'])
plt.show()

# %%
# transform the data
data_pca = pd.DataFrame(
    pca.transform(x), columns=['PC1', 'PC2'], index=x.index
)

# append the target
data_pca[target] = y.values

# %%
# plot the components
plt.figure(figsize=(10, 10))
plt.scatter(x=data_pca['PC1'], y=data_pca['PC2'], c=data[target])
plt.xlabel('PC1', size=14)
plt.ylabel('PC2', size=14)
plt.colorbar()
plt.show()

# %%
# lets try to cluster the data - select the number of clusters
# using the silhouette score
s_score = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=2)
    kmeans.fit(data_pca[['PC1', 'PC2']])
    s_score.append(silhouette_score(data_pca[['PC1', 'PC2']], kmeans.labels_))

# %%
# plot the silhouette score
plt.plot(range(2, 11), s_score, 'o-')
plt.xlabel('No. Cluster', size=14)
plt.ylabel('Silhouette Score', size=14)
plt.title('Cluster No. Selection - Silhouette Score', size=16)
plt.grid(True, alpha=0.2)
plt.xticks(ticks=range(2, 11), labels=range(2, 11))
plt.show()

# %%
# define the final model of the 
kmeans = KMeans(n_clusters=3, random_state=2)
kmeans.fit(data_pca[['PC1', 'PC2']])

# predict the clusters for each data point
data_pca['cluster'] = kmeans.labels_

# %%
# let's analyze the clusters
print(data_pca.groupby('cluster')[target].agg(['mean', 'std', 'count']))

# %%
# append x variables to each cluster
data_pca[x_cols] = x

# analyze distributions
print(data_pca.groupby('cluster').agg(['mean', 'std', 'count']))

# %%
# calculate the bootstrapped mean and 95% confidence interval
# for the target variable
boot_dict = {}
for col in x_cols+[target]:
    boot_dict[col] = {}
    for c in data_pca['cluster'].unique():
        boot_dict[col][c] = bootstrap(data_pca[data_pca['cluster'] == c][col])

# %%
# plot all the bootstrapped samples
for col in x_cols+[target]:
    plot_bootstrap_samples(boot_dict, col)