# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 17:07:32 2018

@author: SWC
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy import linalg
from sklearn import mixture
from sklearn.model_selection import KFold
from learning_funcs import cross_validate_GMM


file_loc = 'C:\Drive\Video Analysis\data\\'
date = '05.02.2018\\'
mouse_session = '202-1a\\'
file_loc = file_loc + date + mouse_session

cross_validation = False
add_velocity = True
vel_scaling_factor = 3 #scale down velocity's importance relative to PC1
dims_used = 6

pca_coeffs = np.load(file_loc + 'PCA_coeffs_sampledata_3_5.npy')
velocity = np.load(file_loc + 'analyze_3_5_velocity.npy')
plt.close('all')


#%% Start with Gaussian Mixture Model -- assign each point probability for each source gaussian


dim_range = [6]
K_range = [10]
seed_range = [0]
num_components_range = [3,10]
improvement_thresh = .01 #.005 - .02; higher threshold means lower number of components may be used in model
tol = .001

data_for_cross_val = pca_coeffs[0:18000,:]

if cross_validation:
    if add_velocity:
        max_vel = np.max(abs(velocity[0:18000,0:2]))
        max_pc = np.max(pca_coeffs[0:18000,0:6])
        velocity_for_model = velocity[0:18000,0:2] / max_vel * max_pc / vel_scaling_factor
        velocity_for_model[:,1] = np.abs(velocity_for_model[:,1]) - np.mean(np.abs(velocity_for_model[:,1]))
        data_for_cross_val = np.append(data_for_cross_val,velocity_for_model,axis=1)
    
    xval_scores = cross_validate_GMM(data_for_cross_val, dim_range, K_range, seed_range, num_components_range, improvement_thresh, tol)
    #in cluster w/ high confidence; 10th percentile of confidence; score; bayesian info content
   

#%%   Generate mixture model with optimal number of latent variables
num_components = 4
filter_pcs = True
#gmm = mixture.GaussianMixture(n_components=num_components,tol=.00001,covariance_type='full',random_state=0)
data_to_generate_gmm = pca_coeffs[0:18000,0:dims_used]

if add_velocity:
    max_vel = np.max(abs(velocity[0:18000,0:2]))
    max_pc = np.max(pca_coeffs[0:18000,0:6])
    velocity_for_model = velocity[0:18000,0:2] / max_vel * max_pc / vel_scaling_factor
    velocity_for_model[:,1] = np.abs(velocity_for_model[:,1]) - np.mean(np.abs(velocity_for_model[:,1]))
    data_to_generate_gmm = np.append(data_to_generate_gmm,velocity_for_model,axis=1)

#%% Get groupings of data from model

if filter_pcs:
    filter_length = 10 
    tau = 3
    
    exp_filter = np.append(np.exp(np.arange(-filter_length,1)/tau),np.zeros(filter_length))
    exp_filter = np.flip(exp_filter / sum(exp_filter),axis=0)
    gauss_filter = np.exp(-np.arange(-filter_length,filter_length+1)**2/tau**2)
    gauss_filter = gauss_filter / sum(gauss_filter)
    pc_filter = gauss_filter
    
    data_to_fit_gmm = np.zeros(data_to_generate_gmm.shape)
    for pc in range(data_to_generate_gmm.shape[1]):
        data_to_fit_gmm[:,pc] = np.convolve(data_to_generate_gmm[:,pc],pc_filter,mode='same')
else:
    data_to_fit_gmm = data_to_generate_gmm
    
gmm.fit(data_to_fit_gmm) 
    
probabilities = gmm.predict_proba(data_to_fit_gmm)
chosen_components = gmm.predict(data_to_fit_gmm)
chosen_probabilities = np.max(probabilities,axis=1)
np.save(file_loc+'chosen_components.npy',chosen_components)

unchosen_probabilities = probabilities
for i in range(probabilities.shape[0]):
    unchosen_probabilities[i,chosen_components[i]]=0 
    
unchosen_components = np.argmax(unchosen_probabilities,axis=1)
unchosen_probabilities = np.max(unchosen_probabilities,axis=1)

colors = ['red','deepskyblue','green','blueviolet','saddlebrown','yellow','white','lightpink']

components_over_time = np.zeros((data_to_fit_gmm.shape[0],num_components))
unchosen_components_over_time = np.zeros((data_to_fit_gmm.shape[0],num_components))
for n in range(num_components):
    components_over_time[:,n] = (chosen_components == n)
    unchosen_components_over_time[:,n] = (unchosen_components == n)

normalized_pca_coeffs = data_to_fit_gmm / np.max(data_to_fit_gmm)

#%% plot the clusters and certainties over time
plt.style.use('classic')
plt.figure(figsize=(30,10))
plt.plot(normalized_pca_coeffs[:,0:3])
for n in range(num_components):
    component_frames = find(components_over_time[:,n])
    plt.scatter(component_frames,np.ones(len(component_frames))*1,color=colors[n],alpha=.7,marker='|',s=700)
    
#    confident_frames = find((chosen_probabilities<.8)*(chosen_components==n))
#    plt.scatter(confident_frames,np.ones(len(confident_frames))*.9,color='k',alpha=.2,marker='|',s=500)
    
    unchosen_component_frames = find(unchosen_components_over_time[:,n] * (unchosen_probabilities>.15))
    plt.scatter(unchosen_component_frames,np.ones(len(unchosen_component_frames))*.9,color=colors[n],alpha=1,marker='|',s=700)

if add_velocity:
    plt.plot(normalized_pca_coeffs[:,-2] * 2, color = 'k',linewidth=2)
    plt.plot(normalized_pca_coeffs[:,-1] * 2, color = 'gray', linestyle = '--',linewidth=2)

legend = plt.legend(('PC1','PC2','PC3','direct velocity','ortho velocity'))
legend.draggable()
plt.title('Principal Components over Time')
plt.xlabel('frame no.')
plt.ylabel('PC amplitude')
plt.xlim([2000,5000])
plt.ylim([-1,1.05])

#plt.close('all')

#%% Get subcluster distribution 
dual_components = (chosen_components+1)*10+(unchosen_components+1)*(unchosen_probabilities>.1)
#plt.figure(figsize=(40,7))
#plt.hist(chosen_components,bins = np.arange(0,6),density = True)
labels = np.array([10,12,13,14,15,20,21,23,24,25,30,31,32,34,35,40,41,42,43,45,50,51,52,53,54]).astype(str)

plt.figure(figsize=(40,7))
for n in range(num_components):
    plt.hist(dual_components[(dual_components>=(n+1)*10) * (dual_components<(n+2)*10)],label=labels,bins = np.arange(10,61),density = True, color = colors[n])
plt.xlim([10,59])
plt.ylim([0,.8])
plt.xticks([10,12,13,14,15,20,21,23,24,25,30,31,32,34,35,40,41,42,43,45,50,51,52,53,54])
plt.title('Distribution of clusters and sub-clusters')

#sum(dual_components==)

#%% Get transition probabilities, for clusters and dual clusters
dual_transitions = False
rc = chosen_components
if dual_transitions:
    ri = dual_components
    lookup = np.empty((65+1,), dtype=int)
    lookup[[10,12,13,14,15,20,21,23,24,25,30,31,32,34,35,40,41,42,43,45,50,51,52,53,54]] = np.arange(25)
    # translate c, r, s to 0, 1, 2
    rc = lookup[ri]

cnts = np.zeros((25,25), dtype=int)
np.add.at(cnts, (rc[:-1], rc[1:]), 1)
# or as probs
origin_prob = cnts / np.sum(cnts,axis=0)
np.nan_to_num(origin_prob,copy=False)
# or as condional probs (if today is sun how probable is rain tomorrow etc.)
transition_prob = (cnts.T / np.sum(cnts,axis=1)).T 
np.nan_to_num(transition_prob,copy=False)

print(cnts)
print((1000*origin_prob).astype(int)/10)
print((1000*transition_prob).astype(int)/10)

#%% Do autoregression in 1-D (cluster) or 9-D (PC) space 

#%% Add context: Iteratively update mixture model / transition probabilities for points in an uncertain? or NOT



#%% Identify changepoints over different temporal scales, using different temporal filters



#%% Cluster sequences in 1-D or 9-D space at varying temporal scales, and get their transition probabilities, as done above with poses