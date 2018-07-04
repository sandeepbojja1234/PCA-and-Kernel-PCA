
# coding: utf-8
#Author: Sandeep
# # Delta Flight Exploration Dataset

# In[77]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
from sklearn.model_selection import cross_val_score
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools


# In[78]:


delta = pd.read_csv('C:\\Users\\msasi\\Desktop\\CSC 529\\Final pRoject 29\\delta.csv')
print(delta.describe())


# In[79]:


delta_names = delta.loc[:,'Aircraft']
print(delta_names)


# In[80]:


delta.head()


# In[81]:


delta.shape


# In[82]:


corr = delta.corr()


# In[49]:


mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[212]:


delta_imp = delta[['Accommodation','Cruising Speed (mph)','Range (miles)','Engines','Wingspan (ft)','Tail Height (ft)','Length (ft)']]


# In[213]:


sns.pairplot(delta_imp)


# In[83]:


#To see whether there are any missing values
(len(delta)-delta.count())/len(delta)


# In[84]:


from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
delta_noname = delta.iloc[:,1:]


# In[85]:


delta_noname.head()


# In[86]:


col_names = list(delta_noname.columns.values)
print(col_names)


# In[87]:


#scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
#scaler.fit(delta_noname)
#delta_scaled = scaler.transform(delta_noname)
#delta_scaled = preprocessing.scale(delta_noname)

scaler = MinMaxScaler()
delta_noname[['Seat Width (Club)', 'Seat Pitch (Club)', 'Seat (Club)', 'Seat Width (First Class)', 'Seat Pitch (First Class)', 'Seats (First Class)', 'Seat Width (Business)', 'Seat Pitch (Business)', 'Seats (Business)', 'Seat Width (Eco Comfort)', 'Seat Pitch (Eco Comfort)', 'Seats (Eco Comfort)', 'Seat Width (Economy)', 'Seat Pitch (Economy)', 'Seats (Economy)', 'Accommodation', 'Cruising Speed (mph)', 'Range (miles)', 'Engines', 'Wingspan (ft)', 'Tail Height (ft)', 'Length (ft)', 'Wifi', 'Video', 'Power', 'Satellite', 'Flat-bed', 'Sleeper', 'Club', 'First Class', 'Business', 'Eco Comfort', 'Economy']] = scaler.fit_transform(delta_noname[['Seat Width (Club)', 'Seat Pitch (Club)', 'Seat (Club)', 'Seat Width (First Class)', 'Seat Pitch (First Class)', 'Seats (First Class)', 'Seat Width (Business)', 'Seat Pitch (Business)', 'Seats (Business)', 'Seat Width (Eco Comfort)', 'Seat Pitch (Eco Comfort)', 'Seats (Eco Comfort)', 'Seat Width (Economy)', 'Seat Pitch (Economy)', 'Seats (Economy)', 'Accommodation', 'Cruising Speed (mph)', 'Range (miles)', 'Engines', 'Wingspan (ft)', 'Tail Height (ft)', 'Length (ft)', 'Wifi', 'Video', 'Power', 'Satellite', 'Flat-bed', 'Sleeper', 'Club', 'First Class', 'Business', 'Eco Comfort', 'Economy']])


# In[88]:


delta_noname.head


# In[89]:


pca = PCA(n_components=4)
pca.fit(delta_noname)
delta_noname_components = pca.transform(delta_noname)


# In[99]:


#converting to dataframe
delta_noname_components_df = pd.DataFrame(delta_noname_components, columns=['PC1','PC2','PC3','PC4'] )


# In[214]:


print(delta_noname_components_df)


# In[107]:


var= pca.explained_variance_ratio_


# In[108]:


var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)


# In[215]:


print(var)


# In[110]:


plt.plot(var1)


# In[111]:


from sklearn.cluster import KMeans
num_clusters = 4
km = KMeans(n_clusters=num_clusters)
get_ipython().magic('time km.fit(delta_noname)')
clusters = km.labels_.tolist()


# In[62]:


delta_names_list = pd.Series.tolist(delta_names)


# In[63]:


import pandas as pd

#aircrafts = { 'Aircraft': delta_names, 'cluster': clusters }
aircrafts = { 'Aircraft': delta_names_list, 'cluster': clusters }
frame = pd.DataFrame(aircrafts, index = [clusters] , columns = ['Aircraft','cluster'])


# In[70]:


frame.shape


# In[64]:


frame['cluster'].value_counts()


# In[65]:


result = frame.groupby(by=clusters)


# In[67]:


#to print clusters
for key, item in result:
    print( result.get_group(key), "\n\n")


# In[253]:


#to plot the clusters
y1 = km.fit_predict(delta_noname_components)
plt.figure(figsize=(8, 8), dpi=80)
plt.title('First 2 principal components after Linear PCA and 4 Clusters')
plt.scatter(delta_noname_components[:,0],delta_noname_components[:,1],c=y1)


# In[119]:


cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a'}

#set up cluster names using a dict
cluster_names = {0: 'Long Range', 
                 1: 'Domestic Short Range', 
                 2: 'VIP Jets', 
                 3: 'Private Jets',
                }


# In[127]:


delta_noname_components.shape


# In[258]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(delta_noname_components[:, 0], delta_noname_components[:, 1], delta_noname_components[:, 2],cmap=plt.cm.Paired)
titel="First three directions in PCA"
ax.set_title(titel)
ax.set_xlabel("1st eigenvector")
ax.set_ylabel("2nd eigenvector")
ax.set_zlabel("3rd eigenvector")

plt.show()


# In[217]:


#Elbow mwthod to choose the optimal number of K for clusteringp
import numpy
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

distortion = []
K = range(1,10)
for k in K:
    kmeans = KMeans(n_clusters = k).fit(delta_noname)
    kmeans.fit(delta_noname)
    distortion.append(sum(numpy.min(cdist(delta_noname, kmeans.cluster_centers_, 'euclidean'), axis=1)) / delta_noname.shape[0])
    
plt.plot(K, distortion, 'bx-')
plt.title('The Elbow Method showing the optimal k')
plt.show()    


# In[277]:


#PCA with RBF
from sklearn.decomposition import PCA, KernelPCA

kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10,n_components=4)
kpca.fit(delta_noname)
delta_noname_components_rbf = kpca.transform(delta_noname)


# In[278]:


#to get the variance being explained by the components
import numpy
explained_variance = numpy.var(delta_noname_components_rbf, axis=0)
explained_variance_ratio = explained_variance / numpy.sum(explained_variance)
print(explained_variance_ratio)


# In[279]:


delta_noname_components_rbf.shape


# In[280]:


delta_noname_components_df_rbf = pd.DataFrame(delta_noname_components_rbf, columns=['RBF-PC1','RBF-PC2','RBF-PC3','RBF-PC4'] )


# In[282]:


from sklearn.cluster import KMeans
num_clusters = 4
km = KMeans(n_clusters=num_clusters)


# In[283]:


y1 = km.fit_predict(delta_noname_components_rbf)
plt.figure(figsize=(8, 8), dpi=80)
plt.title('First 2 principal components after Linear PCA and 4 Clusters')
plt.scatter(delta_noname_components_rbf[:,0],delta_noname_components_rbf[:,1],c=y1)


# In[284]:


clusters = km.labels_.tolist()


# In[285]:


import pandas as pd

#aircrafts = { 'Aircraft': delta_names, 'cluster': clusters }
aircrafts = { 'Aircraft': delta_names_list, 'cluster': clusters }
frame_rbf = pd.DataFrame(aircrafts, index = [clusters] , columns = ['Aircraft','cluster'])


# In[286]:


frame_rbf['cluster'].value_counts()


# In[287]:


result_rbf = frame_rbf.groupby(by=clusters)


# In[288]:


#to print clusters
for key, item in result_rbf:
    print( result_rbf.get_group(key), "\n\n")


# In[275]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(delta_noname_components_rbf[:, 0], delta_noname_components_rbf[:, 1], delta_noname_components_rbf[:, 2],cmap=plt.cm.Paired)
titel="First three directions in PCA"
ax.set_title(titel)
ax.set_xlabel("1st eigenvector")
ax.set_ylabel("2nd eigenvector")
ax.set_zlabel("3rd eigenvector")

plt.show()


# In[298]:


#Polynomial Kernel with degree 2
from sklearn.cluster import KMeans
num_clusters = 4
km = KMeans(n_clusters=num_clusters)
kpca_poly = KernelPCA(kernel="poly", gamma=10,n_components=4, degree = 2)
kpca_poly.fit(delta_noname)
delta_noname_components_poly = kpca_poly.transform(delta_noname)


# In[299]:


delta_noname_components_poly.shape


# In[300]:


delta_noname_components_df_poly = pd.DataFrame(delta_noname_components_poly, columns=['POLY-PC1','POLY-PC2','POLY-PC3','POLY-PC4'])


# In[301]:


y1 = km.fit_predict(delta_noname_components_poly)
plt.figure(figsize=(8, 8), dpi=80)
plt.title('First 2 principal components after Linear PCA and 4 Clusters')
plt.scatter(delta_noname_components_poly[:,0],delta_noname_components_poly[:,1],c=y1)


# In[307]:


clusters = km.labels_.tolist()


# In[308]:


import pandas as pd

#aircrafts = { 'Aircraft': delta_names, 'cluster': clusters }
aircrafts = { 'Aircraft': delta_names_list, 'cluster': clusters }
frame_poly = pd.DataFrame(aircrafts, index = [clusters] , columns = ['Aircraft','cluster'])


# In[309]:


frame_poly['cluster'].value_counts()


# In[310]:


result_poly = frame_poly.groupby(by=clusters)


# In[311]:


#to print clusters
for key, item in result_poly:
    print( result_poly.get_group(key), "\n\n")


# In[312]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
cluster_col = {'#1b9e77', '#d95f02', '#7570b3', '#e7298a'}
# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(delta_noname_components_poly[:, 0], delta_noname_components_poly[:, 1], delta_noname_components_poly[:, 2],cmap=plt.cm.Paired)
titel="First three directions in PCA"
ax.set_title(titel)
ax.set_xlabel("1st eigenvector")
ax.set_ylabel("2nd eigenvector")
ax.set_zlabel("3rd eigenvector")

plt.show()

