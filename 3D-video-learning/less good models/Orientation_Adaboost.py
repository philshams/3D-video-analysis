'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

-----------                                   Classify Mouse Orientation                             --------------------------------


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import numpy as np; import cv2; import os; import matplotlib.pyplot as plt
from learning_funcs import add_velocity_as_feature
#%% -------------------------------------------------------------------------------------------------------------------------------------
# ------------------------              Select data file and analysis parameters                 --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------
# Select data file name and folder location
# ------------------------------------------
# Data for model generation
file_loc = 'C:\Drive\Video Analysis\data\\'
date = 'baseline_analysis\\'
mouse_session = 'together_for_model\\'
save_vid_name = 'all'
adaboost_name = 'adaboost'

model_file_loc = file_loc + date + mouse_session + save_vid_name
adaboost_file_loc = file_loc + date + mouse_session + adaboost_name

# ---------------------------
# Select analysis parameters
# ---------------------------
start_PC = 3 #up to four
end_PC = 8

add_velocity = True


#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------              Prepare data                                    --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

# -------------------
# Load relevant data
# -------------------

# Load Velocity from normal videos
velocity = np.load(model_file_loc + '_velocity.npy')
disruptions = np.load(model_file_loc + '_disruption.npy')

# Load PCs
pca_coeffs = np.load(model_file_loc + '_pca_coeffs.npy')
pca_coeffs_upside_down = np.load(adaboost_file_loc + '_upside_down_pca_coeffs.npy')

pca_coeffs = np.load(adaboost_file_loc + '_pca_coeffs.npy')
pca_coeffs_upside_down = np.load(adaboost_file_loc + '_upside_down_pca_coeffs.npy')

# Use this data for the classifier
data_for_classifier = pca_coeffs[:,start_PC:end_PC]
data_for_classifier_upside_down = pca_coeffs_upside_down[:,start_PC:end_PC]

# normalise data 0 - 1
max_feature_value = np.max(abs(data_for_classifier),axis=0)
data_for_classifier = data_for_classifier / max_feature_value
data_for_classifier_upside_down = data_for_classifier_upside_down / max_feature_value

# Include velocity as a pseudo-PC
if add_velocity:
    speed_only = False
    add_turn = False
    #find disruptions and 6 stdev, throw out those beyond that as spurious
    velocity[disruptions==1,0] = 0
    mean_vel = np.mean(velocity[:,0],axis=0)
    std_vel = np.std(velocity[:,0],axis=0)
    velocity[abs(velocity[:,0]-mean_vel) > 6*std_vel,0] = 0
    mean_vel = np.mean(velocity[:,0],axis=0)
    
    velocity_for_model = np.zeros((data_for_classifier.shape[0],1))
    velocity_for_model[:,0] = (velocity[:,0]) / 6*std_vel #scale 0 to 1 ... 
    data_for_classifier = np.append(data_for_classifier,velocity_for_model,axis=1)
    data_for_classifier_upside_down = np.append(data_for_classifier_upside_down,-1*velocity_for_model,axis=1)



# append data and create labels
labels = np.concatenate((np.ones(data_for_classifier.shape[0]),-1 * np.ones(data_for_classifier_upside_down.shape[0])), axis = 0 )
data_for_classifier = np.concatenate((data_for_classifier, data_for_classifier_upside_down), axis = 0)



#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------                     Classifier Functions                       --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------


#coordinate is the dimension, polarity is -1 or 1 (above or below), theta is the threshold value
def evaluate_stump(features, coordinate_wl, polarity_wl, theta_wl): 
    """Evaluate the stump's response for each point.""" 
    feature_slice = features[:, coordinate_wl]
    weak_learner_output = polarity_wl * (2*(feature_slice >= theta_wl) - 1) #decision stump is +1 for true and -1 for false

    return weak_learner_output


def find_best_weak_learner(weights, features, labels, feature_range, last_coordinate):
    """Find the best decision stump for the given weight distribution."""
    coordinate_wl = 0
    polarity_wl = 1
    theta_wl = 0.
    err_wl = np.inf
    err_best = np.inf

    #Find the best weak learner ht at each round.
    #calculate error for each d, relevant thetas, and s
    i = 0
    for coordinate in range(0,features.shape[1]): #over all dimensions
        if coordinate == last_coordinate:
            continue
        for polarity in [-1,1]:
            for theta in feature_range:
                i+= 1
                labels_wl = evaluate_stump(features, coordinate, polarity, theta)
                err = sum(weights * np.not_equal(labels_wl, labels))
                
                if err < err_best:
                    coordinate_wl = coordinate  #Dimension 'd' along which the threshold is applied.
                    polarity_wl = polarity #Polarity 's' of the decision stump.
                    theta_wl = theta #Threshold 'theta' for the decision.
                    err_wl = err #Weighted error for the decision stump.
                    err_best = err
                    
    return coordinate_wl, polarity_wl, theta_wl, err_wl

def evaluate_stump_on_grid(feature_range, coordinate_wl, polarity_wl, theta_wl):
    """Evaluate the stump's response for each point on a PC-dimensional grid."""
    
    feature_slice = np.meshgrid(feature_range,feature_range,feature_range,feature_range,feature_range,feature_range)[coordinate_wl] #5D
    #feature_slice = np.meshgrid(feature_range, feature_range)[coordinate_wl] #2D
    
    weak_learner_on_grid = polarity_wl * (2*(feature_slice >= theta_wl) - 1)

    return weak_learner_on_grid


#%% -------------------------------------------------------------------------------------------------------------------------------------
#------------------------                           Run Adaboost                       --------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------

npoints = data_for_classifier.shape[0]
num_rounds_boosting = 1000

output = np.zeros(data_for_classifier.shape[0])
loss = np.zeros(num_rounds_boosting)

# Initialize arrays.
weights = np.ones(npoints) / npoints  # Weight distribution on samples
last_coordinate =  10

f_on_grid = 0  # Used to plot function
feature_range = np.linspace(-1, 1, 30)

plt.close('all')
fig2 = plt.figure('Strong Learner')
ax = fig2.add_subplot(1, 1, 1)
ax.set_title("Strong learner")
ax.set_ylabel("PC 1")
ax.set_xlabel("Vel")

for i in range(num_rounds_boosting):

    print('Round ' + str(i))
    
    # Find best weak learner at current round of boosting.
    coordinate_wl, polarity_wl, theta_wl, err_wl = find_best_weak_learner(weights, data_for_classifier, labels, feature_range, last_coordinate)
    print(coordinate_wl, polarity_wl, theta_wl, err_wl)
    if err_wl > .5:
        print('no more guarantees, buddy!')
        break

    # Estimate alpha.
    alpha = .5 * np.log((1 - err_wl) / (err_wl))

    # Reweight samples.
    labels_wl = evaluate_stump(data_for_classifier, coordinate_wl, polarity_wl, theta_wl)
    weights = weights * np.exp(alpha * -(labels_wl * labels)) 
    weights = weights / sum(weights)
    
    # Compute overall response at current round.
    output += labels_wl * alpha

    # Compute loss at current round.
    loss[i] = sum(np.exp(-output * labels))
    #print(loss[i])
    print(str(np.sum(np.equal(sign(output), labels)) / len(labels) * 100) + '% correct')
    print('')
    
    # Evaluate f on a grid to produce the images.
    weak_learner_on_grid = evaluate_stump_on_grid(feature_range, coordinate_wl, polarity_wl, theta_wl)
    f_on_grid += alpha*weak_learner_on_grid
    
    #Plot posterior distribution over the PC1 and velocity        
    approx_posterior = 1 / (1 + np.exp(-2 * f_on_grid))
    ax.imshow(np.squeeze(approx_posterior[:,15,15,15,15,:]), extent=[-1, 1, -1, 1], origin='lower')
    #ax.imshow(approx_posterior, extent=[-1, 1, -1, 1], origin='lower') #PC1 and velocity
    plt.pause(1)

  
#Plot loss
fig1 = plt.figure()
ax = fig1.add_subplot(1, 1, 1)
ax.plot(loss)
ax.set_title("Decrease of the loss function over time")
ax.set_xlabel("Iteration Number")
ax.set_ylabel("Loss Function")






