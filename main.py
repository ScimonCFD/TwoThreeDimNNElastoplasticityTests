# License
#  This program is free software: you can redistribute it and/or modify 
#  it under the terms of the GNU General Public License as published 
#  by the Free Software Foundation, either version 3 of the License, 
#  or (at your option) any later version.
#  This program is distributed in the hope that it will be useful, 
#  but WITHOUT ANY WARRANTY; without even the implied warranty of 
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
#  See the GNU General Public License for more details. You should have 
#  received a copy of the GNU General Public License along with this 
#  program. If not, see <https://www.gnu.org/licenses/>. 

# Description
#  This code evaluates the performance of a recurrent neural network as a surrogate 
#  model for the isotropic hardening constitutive law. The code accomplishes 
#  this through the following main steps:
#  1. Generation of several strain sequences:
#  Multiple strain sequences are generated, with a maximum accumulated strain 
#  restricted to 5%.
#  2. Calculation of Isotropic Hardening Stresses:
#  The code calculates the isotropic hardening stresses corresponding to the 
#  strain sequences using Simo's algorithm.
#  3. Dataset Splitting:
#  The dataset is divided into training (70%), validation (20%), and test (10%) 
#  sets.
#  4. Testing the neural network designs:
#  Several unseen strain sequences are passed to the neural network, which is 
#  used to calculate the corresponding stress sequences. 
#  5. Plotting Expected vs. Neural Network-Based Results:
#  The code generates plots comparing expected results to those produced by the 
#  neural network for the sequences in the test set.


# Authors
#  Simon A. Rodriguez, UCD. All rights reserved
#  Philip Cardiff, UCD. All rights reserved

import numpy as np
from numpy import linalg as LA
from distutils.dir_util import mkpath
from tqdm import tqdm
from functions import *
from input_file import *
from sklearn.preprocessing import MinMaxScaler
import random
from joblib import dump
import time

# Seed everything
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Create loss functions
mae = tf.keras.losses.MeanAbsoluteError()
mse = tf.keras.losses.MeanSquaredError()

N_DIM_LIST = [2, 3]

TYPE_STRAINS_LIST = ["control_points", "random"]
mae_nn_random = []
mse_nn_random = []
mae_nn_control_points = []
mse_nn_control_points = []
mae_nn_random3D = []
mse_nn_random3D = []
mae_nn_control_points3D = []
mse_nn_control_points3D = []

for type_strains in TYPE_STRAINS_LIST:
    for n_dim in N_DIM_LIST:
        sequence_lenght = NUMBER_CONTROL_POINTS * NUMBER_INTERPOLATION_POINTS 
        # Build elastic moduli
        ce = build_elastic_moduli(n_dim, LAME_1, LAME_2)
        STRAINS_PATH_N = type_strains + "/" + str(n_dim) +"D/" + STRAINS_PATH
        STRESSES_PATH_N = type_strains + "/" + str(n_dim) +"D/" + STRESSES_PATH 
        PLOTS_PATH_N = type_strains + "/" + str(n_dim) +"D/" + PLOTS_PATH
        # Create the folders for strains and stresses
        mkpath(STRAINS_PATH_N)
        mkpath(STRESSES_PATH_N)
        
        # Write the strains/stresses sequences
        if (type_strains == "control_points"):
            write_strains(STRAINS_PATH_N, n_dim, NUMBER_STRAIN_SEQUENCES, 
                          sequence_lenght, MAX_ABS_DEFORMATION, 
                          NUMBER_CONTROL_POINTS, NUMBER_INTERPOLATION_POINTS)
        
        elif (type_strains == "random"):
            write_random_strains(STRAINS_PATH_N, n_dim, NUMBER_STRAIN_SEQUENCES, 
                         sequence_lenght, MAX_ABS_DEFORMATION, 
                         NUMBER_CONTROL_POINTS, NUMBER_INTERPOLATION_POINTS)            
        sequence_lenght = NUMBER_CONTROL_POINTS * \
                                                NUMBER_INTERPOLATION_POINTS + 1
        
        # Plot acc. strain
        plot_acc_strain(STRAINS_PATH_N, STRAINS_PATH_N + "Acc_strains/", 10, n_dim)
        
        calculate_stresses(LAME_2, SIGMA_Y, BETA, STRAINS_PATH_N, 
                           STRESSES_PATH_N, sequence_lenght, 
                           NUMBER_STRAIN_SEQUENCES, n_dim, H, ce)
        
        # Create the scalers
        x_scaler =  MinMaxScaler()
        y_scaler  =  MinMaxScaler()
        
        # Create the dataset
        [x_train, y_train, x_validation, y_validation, x_test, y_test] =  \
            create_datasets(n_dim, sequence_lenght, NUMBER_STRAIN_SEQUENCES, 
                            STRAINS_PATH_N, STRESSES_PATH_N, SPLITTER)
        
        # Remove the first row which is full of zeros 
        x_train = x_train[:, 1:, :]
        y_train = y_train[:, 1:, :]
        x_validation = x_validation[:, 1:, :]
        y_validation = y_validation[:, 1:, :]
        x_test = x_test[:, 1:, :]
        y_test = y_test[:, 1:, :]
        
        # Normalise the dataset 
        [x_train_scaled, 
         y_train_scaled, 
         x_validation_scaled, 
         y_validation_scaled, 
         x_test_scaled, 
         y_test_scaled] =  create_scaled_dataset(x_scaler, y_scaler, x_train, 
                                                 y_train, x_validation, 
                                                 y_validation, x_test, y_test)
        
        # Create the model 
        RNN = create_simple_rnn(10, n_dim)
        
        # Compile the RNN
        compile_nn(RNN, "Adam", "MSE")
        
        # Train the neural network
        model_history = train_nn(RNN, x_train_scaled, y_train_scaled, 
                                 x_validation_scaled,  y_validation_scaled, 
                                 x_test_scaled, y_test_scaled, 
                                 NUMBER_OF_EPOCHS)
        
        # Calculate y_prediction
        y_prediction_scaled = RNN.predict(x_test_scaled)
        y_prediction = y_scaler.inverse_transform(
            y_prediction_scaled.reshape([y_prediction_scaled.shape[0]*
                                         y_prediction_scaled.shape[1],
                                         y_prediction_scaled.shape[2]]))
        y_prediction = y_prediction.reshape(y_prediction_scaled.shape)
        
        # Plot the results
        generate_all_plots(x_test, y_test/1e9, y_prediction/1e9, PLOTS_PATH_N, 
                            model_history, 10, n_dim)
        
        # Serialise the dataset, scalers and machine-learning model
        serilaise_dataset(x_train, y_train, x_validation, y_validation, x_test,  
                          y_test, y_prediction, type_strains + "/" + 
                          str(n_dim) + "D/")
        RNN.save(type_strains + "/" + str(n_dim) + "D/ML_model.h5")
        dump(x_scaler, type_strains + "/" + str(n_dim) + 'D/x_scaler.joblib')
        dump(y_scaler, type_strains + "/" +str(n_dim) + 'D/y_scaler.joblib')
        
        if (type_strains == "control_points"):
            if (n_dim == 2):
                mae_nn_control_points = mae(y_test, y_prediction).numpy()
                mse_nn_control_points = mse(y_test, y_prediction).numpy()
            elif (n_dim == 3):
                mae_nn_control_points3D = mae(y_test, y_prediction).numpy()
                mse_nn_control_points3D = mse(y_test, y_prediction).numpy()
                
        
        elif (type_strains == "random"):
            if (n_dim == 2):
                mae_nn_random = mae(y_test, y_prediction).numpy()
                mse_nn_random = mse(y_test, y_prediction).numpy()
                
            elif (n_dim == 3):
                mae_nn_random3D = mae(y_test, y_prediction).numpy()
                mse_nn_random3D = mse(y_test, y_prediction).numpy()
                
with open('control_points/2D/report.txt', 'a') as f:
    f.write("Results are the following: \n \n")
    f.write("MAE(expected vs. predicted) using control points " + " is " + 
            str(mae_nn_control_points) + "\n")
    f.write("MSE(expected vs. predicted) using control points " + " is " + 
            str(mse_nn_control_points) + "\n")  
    f.close()

with open('control_points/3D/report.txt', 'a') as f:              
    f.write("Results are the following: \n \n")
    f.write("MAE(expected vs. predicted) using control points " + " is " + 
            str(mae_nn_control_points3D) + "\n")
    f.write("MSE(expected vs. predicted) using control points " + " is " + 
            str(mse_nn_control_points3D) + "\n")  
    f.close()                 

with open('random/2D/report.txt', 'a') as f:
    f.write("Results are the following: \n \n")
    f.write("MAE(expected vs. predicted) using control points " + " is " + 
            str(mae_nn_random) + "\n")
    f.write("MSE(expected vs. predicted) using control points " + " is " + 
            str(mse_nn_random) + "\n")  
    f.close()

with open('random/3D/report.txt', 'a') as f:              
    f.write("Results are the following: \n \n")
    f.write("MAE(expected vs. predicted) using control points " + " is " + 
            str(mae_nn_random3D) + "\n")
    f.write("MSE(expected vs. predicted) using control points " + " is " + 
            str(mse_nn_random3D) + "\n")  
    f.close()    
    
print("\n The end")