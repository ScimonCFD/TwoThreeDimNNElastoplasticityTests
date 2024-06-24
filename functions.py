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
# This file contains the functions that are used in the main code

# Authors
#  Simon A. Rodriguez, UCD. All rights reserved
#  Philip Cardiff, UCD. All rights reserved


import numpy as np
from numpy import linalg as LA
from distutils.dir_util import mkpath
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle

def plot_acc_strain(strains_path, plots_path, number_sequences, n_dim):
    def calculate_eq_strain(strains):
        trace = (np.sum(strains[:, 0:3], axis = 1)/3).reshape(
                                                       strains.shape[0], 1)         
        dev_strains = np.copy(strains) #Initialisation of dev_strains array
        dev_strains[:, 0:3] = dev_strains[:, 0:3] - trace
        # eq_strains = (((2/3)*np.sum((dev_strains*dev_strains), 
        #                             axis = 1))**0.5)[:, np.newaxis] # Equation in https://www.continuummechanics.org/vonmisesstress.html
        eq_strains = (
                       (
                            (2/3)*np.sum((dev_strains**2),axis = 1) + 
                            (2/3)*np.sum((dev_strains[:, 3:]**2), axis = 1)
                         )**0.5)[:, np.newaxis] # Equation in https://www.continuummechanics.org/vonmisesstress.html
        return eq_strains
    
    mkpath(plots_path)
    labels = ["sigma_xx", "sigma_yy", "sigma_zz", "sigma_xy", "sigma_xz",  
              "sigma_yz"]
    labels_colors = ["dimgray", "steelblue", "forestgreen", 
                          "indianred", "saddlebrown", "purple",  "gold"]
    for i in range(number_sequences):
        strains = (np.loadtxt(strains_path + '/%i.txt' %(i))[:, :])
        plt.figure(figsize=(15, 10))
        for j in range(strains.shape[1]):
            if (n_dim ==2):
                if (j in [0, 1, 3]):
                    plt.plot(range(strains.shape[0]), np.cumsum(strains[:, j]), 
                          linewidth=3, label = labels[j], color = labels_colors[j])
            else:
                plt.plot(range(strains.shape[0]), np.cumsum(strains[:, j]), 
                          linewidth=3, label = labels[j], color = labels_colors[j])   
        plot_name  = plots_path + "sequence_" + str(i) + '.png'
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("time step", fontsize = 20)
        plt.ylabel("Accumulated strain (mm/mm)", fontsize = 20)
        plt.grid(True)
        plt.savefig(plot_name, bbox_inches='tight')
        plt.close()
        
        #Plot accumulated equivalent strain
        plt.figure(figsize=(15, 10))
        plot_name  = plots_path + "Acc_strain_sequence_" + str(i) + '.png'
        total_acc_strain = np.cumsum(np.copy(strains), axis = 0)
        eq_strains = calculate_eq_strain(np.copy(total_acc_strain))
        max_eq_strain = np.max(abs(eq_strains), axis = 0)
        # print(max_eq_strain)
        plt.plot(range(strains.shape[0]), eq_strains, linewidth=3)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("time step", fontsize = 20)
        plt.ylabel("Accumulated equivalent strain (mm/mm)", fontsize = 20)
        plt.grid(True)
        plt.savefig(plot_name, bbox_inches='tight')
        plt.close()


def generate_all_plots(x, y, y_pred, plots_path, history, number_of_plots, 
                       n_dim):
    places=np.arange(number_of_plots)
    mkpath(plots_path)
    if (n_dim == 2):
        labels  = ["sigma_xx", "sigma_yy", "sigma_zz", "sigma_xy"]
        labels_pred = ["sigma_xx_pred", "sigma_yy_pred", "sigma_zz_pred", 
                       "sigma_xy_pred"]
        labels_colors = ["dimgray", "steelblue", "forestgreen", 
                         "indianred", "saddlebrown", "purple",  "gold"]
        labels_colors_predictions = ["lightgray", "deepskyblue", "lime", 
                                     "lightcoral",  "peru", "darkviolet", 
                                     "yellow"]
    else:
        labels = ["sigma_xx", "sigma_yy", "sigma_zz", "sigma_xy", 
                  "sigma_xz",  "sigma_yz"]
        labels_pred = ["sigma_xx_pred", "sigma_yy_pred", "sigma_zz_pred", 
                     "sigma_xy_pred", "sigma_xz_pred", "sigma_yz_pred"]
        labels_colors = ["dimgray", "steelblue", "forestgreen", 
                         "indianred", "saddlebrown", "purple",  "gold"]
        labels_colors_predictions = ["lightgray", "deepskyblue", "lime", 
                                     "lightcoral", "peru", "darkviolet", 
                                     "yellow"]
    coord_x = np.array(range(0, x.shape[1]), dtype = float)
    for i in places:            
        coord_y  = np.zeros([y.shape[1]])
        coord_y_pred = np.zeros([y.shape[1]])
        plt.figure(figsize=(15, 10))
        ax = plt.subplot(111)
        plot_name  = plots_path + '/test_sample_number_' + str(i) + '.png'
        for j in range(n_dim * 2):
            coord_y = y[i, :, j]
            coord_y_pred = y_pred[i, :, j]   
            plt.plot(coord_x, coord_y, label = labels[j],
                     color = labels_colors[j])
            plt.plot(coord_x, coord_y_pred, linestyle='dashed', 
                     label = labels_pred[j], color = labels_colors[j])                                       
        # Shrink current axis by 20%
        box = ax.get_position() #
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])#
        # Put a legend to the right of the current axis
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("time steps", fontsize=16)
        plt.ylabel("Stress (GPa)", fontsize=16)
        plt.grid(True)
        plt.savefig(plot_name, bbox_inches='tight')
        plt.close()
    pd.DataFrame(history.history).plot(figsize=(15, 10))
    plot_name = plots_path + '/Convergence history.png'
    plt.grid(True)
    plt.legend()
    plt.yscale("log")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Epoch", fontsize = 16)
    plt.ylabel("Loss", fontsize = 16)
    plt.savefig(plot_name, bbox_inches='tight')
    
def create_simple_rnn(nodes, n_dim):
    model = Sequential()
    if (n_dim == 3):
        model.add(GRU(nodes, return_sequences = True, input_shape = [None, 6])),
        model.add(Dense(units = nodes, kernel_initializer =  'he_normal', 
                        activation = 'relu'))
        model.add(Dense(units = 6, kernel_initializer = 'he_normal',
                        activation = 'linear'))                
    else:
        model.add(GRU(nodes, return_sequences = True, input_shape = [None, 3])),
        model.add(Dense(units = nodes,  kernel_initializer =  'he_normal', 
                        activation = 'relu'))
        model.add(Dense(units = 4, kernel_initializer = 'he_normal', 
                        activation = 'linear'))    
    return model
    
def compile_nn(model, opt, loss_function):
    model.compile(optimizer=opt, loss= loss_function)
    
def train_nn(model, x_train, y_train, x_validation, y_validation, x_test, 
             y_test, number_of_epochs):
    history = model.fit(x_train, y_train,  epochs = number_of_epochs,
                        validation_data = (x_validation,  y_validation))
    return history


def create_datasets(n_dim, sequence_lenght, number_strain_sequences, 
                    strains_path, simo_results_path, splitter):
    total_indices = range(0, number_strain_sequences)
    training_indices = random.sample(total_indices,  round(splitter[0]* \
                                           number_strain_sequences))
    possible_validation_indices = list(set(total_indices) - \
                                       set(training_indices))
    validation_indices = random.sample(possible_validation_indices, 
                                       round(splitter[1]\
                                       * number_strain_sequences))
    test_indices = list(set(total_indices) - set(training_indices) - \
                        set(validation_indices))
         
    x_train = np.zeros([len(training_indices), sequence_lenght, 6])
    y_train = np.zeros([len(training_indices), sequence_lenght, 6])
    x_validation = np.zeros([len(validation_indices), sequence_lenght, 6])
    y_validation = np.zeros([len(validation_indices), sequence_lenght, 6])
    x_test = np.zeros([len(test_indices), sequence_lenght, 6])
    y_test = np.zeros([len(test_indices), sequence_lenght, 6])

    print("Creating training set")
    for i in tqdm(range(0, len(training_indices))):
        x_train[i,:,:] = (np.loadtxt(strains_path + 
                                     '/%i.txt' %(training_indices[i]))[:, :])
        y_train[i,:,:] = (np.loadtxt(simo_results_path + 
                                     '/%i.txt' %(training_indices[i]))[:, :])

    print("Creating validation set")
    for i in tqdm(range(0, len(validation_indices))):
        x_validation[i,:,:] = (np.loadtxt(strains_path + 
                                     '/%i.txt' %(validation_indices[i]))[:, :])
        y_validation[i,:,:] = (np.loadtxt(simo_results_path + 
                                     '/%i.txt' %(validation_indices[i]))[:, :])

    print("Creating test set")
    for i in tqdm(range(0, len(test_indices))):
        x_test[i,:,:] = (np.loadtxt(strains_path + 
                                    '/%i.txt' %(test_indices[i]))[:, :])
        y_test[i,:,:] = (np.loadtxt(simo_results_path + 
                                    '/%i.txt' %(test_indices[i]))[:, :])
        
    if (n_dim == 2):
        x_train = np.delete(x_train[:, :, 0:4], 2, 2) 
        x_validation = np.delete(x_validation[:, :, 0:4], 2, 2) 
        x_test = np.delete(x_test[:, :, 0:4], 2, 2)
        y_train = y_train[:,:, 0:4]
        y_validation = y_validation[:,:, 0:4]
        y_test = y_test[:,:, 0:4]
        
    return x_train, y_train, x_validation, y_validation, x_test, y_test

def write_strains(strains_path, n_dim, number_strain_sequences, 
             sequence_lenght, max_abs_deformation, number_control_points,
             number_interpolation_points):
    
        def calculate_eq_strain(strains):
            trace = (np.sum(strains[:, 0:3], axis = 1)/3).reshape(
                                                           strains.shape[0], 1)         
            dev_strains = np.copy(strains) #Initialisation of dev_strains array
            dev_strains[:, 0:3] = dev_strains[:, 0:3] - trace
            # eq_strains = (((2/3)*np.sum((dev_strains*dev_strains), 
            #                             axis = 1))**0.5)[:, np.newaxis] # Equation in https://www.continuummechanics.org/vonmisesstress.html
           
            eq_strains = (
                  (
                     ((2/3)*np.sum((dev_strains**2), axis = 1)) +
                     ((2/3)*np.sum((dev_strains[:, 3:]**2), axis = 1))
                  )
                   **0.5)[:, np.newaxis] # Equation in https://www.continuummechanics.org/vonmisesstress.html
            return eq_strains

        print("Calculating strains")
        mu, sigma = 0, 1 #mean and variance
        for i in tqdm(range(0, number_strain_sequences)):
            control_points = np.zeros([number_control_points, 6])
            if (n_dim == 2):
                control_points[:, 0:3] = np.random.normal(mu, sigma, \
                                             size = (number_control_points,3))
                control_points[:, 3] = control_points[:, 2]
                control_points[:, 2] = 0
            else:
                control_points[:, :] = np.random.normal(mu, sigma, 
                                             size = (number_control_points, 6)) 
            total_accumulated_strain = np.cumsum(control_points, axis = 0)
            eq_strains = calculate_eq_strain(np.copy(total_accumulated_strain)) 
            max_eq_strain = np.max(abs(eq_strains), axis = 0)

            if (max_eq_strain > max_abs_deformation): 
                control_points = control_points * (max_abs_deformation / 
                                                   max_eq_strain)  
                # Check that it was limited properly
                total_accumulated_strain = np.cumsum(control_points, axis = 0)
                eq_strains = calculate_eq_strain(
                                             np.copy(total_accumulated_strain)) 
                max_eq_strain = np.max(abs(eq_strains), axis = 0)
                # print("max_eq_strain is ", max_eq_strain)
            strains = np.zeros([sequence_lenght + 1, 6])
            strains[1:, :] = interpolate_linearly(np.cumsum(control_points, 
                                                            axis = 0), 
                                                  number_interpolation_points)
            np.savetxt('./' + strains_path + '/' + '%i.txt' %(i), strains, 
                                    delimiter = ' ')
        print("Strains calculation finished")
        # time.sleep(0.3)
        
def write_random_strains(strains_path, n_dim, number_strain_sequences, 
             sequence_lenght, max_abs_deformation, number_control_points,
             number_interpolation_points):
        def calculate_eq_strain(strains):
            trace = (np.sum(strains[:, 0:3], axis = 1)/3).reshape(
                                                           strains.shape[0], 1)         
            dev_strains = np.copy(strains) #Initialisation of dev_strains array
            dev_strains[:, 0:3] = dev_strains[:, 0:3] - trace
            # eq_strains = (((2/3)*np.sum((dev_strains*dev_strains), 
            #                             axis = 1))**0.5)[:, np.newaxis] # Equation in https://www.continuummechanics.org/vonmisesstress.html
            eq_strains = (
                           (
                                (2/3)*np.sum((dev_strains**2),axis = 1) + 
                                (2/3)*np.sum((dev_strains[:, 3:]**2), axis = 1)
                             )**0.5)[:, np.newaxis] # Equation in https://www.continuummechanics.org/vonmisesstress.html
            return eq_strains

        print("Calculating strains \n")
        time.sleep(0.3)
        mu, sigma = 0, 1 #mean and variance
        for i in tqdm(range(0, number_strain_sequences)):
            control_points = np.zeros([number_control_points * 
                                       number_interpolation_points + 1, 6])
            if (n_dim == 2):
                control_points[1:, 0:3] = np.random.normal(mu, sigma, \
                                            size = (number_control_points *
                                                   number_interpolation_points, 
                                                      3))
                control_points[:, 3] = control_points[:, 2]
                control_points[:, 2] = 0
            else:
                control_points[1:, :] = np.random.normal(mu, sigma, size = 
                                                       (number_control_points *
                                               number_interpolation_points, 6))
            total_accumulated_strain = np.cumsum(control_points, axis = 0)
            eq_strains = calculate_eq_strain(np.copy(total_accumulated_strain)) 
            max_eq_strain = np.max(abs(eq_strains), axis = 0)

            if (max_eq_strain > max_abs_deformation):           
                control_points = control_points * (max_abs_deformation / 
                                                   max_eq_strain)  
                # Check that it was limited properly
                total_accumulated_strain = np.cumsum(control_points, axis = 0)
                eq_strains = calculate_eq_strain(np.copy(
                                                     total_accumulated_strain)) 
                max_eq_strain = np.max(abs(eq_strains), axis = 0)
                # print("max_eq_strain is ", max_eq_strain)
            strains = np.copy(control_points) 
            np.savetxt('./' + strains_path + '/' + '%i.txt' %(i), strains, 
                       delimiter = ' ')
        print("Strains calculation finished")
        # time.sleep(0.3)

def interpolate_linearly(array, number_interpolation_points):
    # Insert 0 before the first line
    dum = np.zeros([array.shape[0]+1, array.shape[1]])
    dum[1:, :] = array
    #########################################    
    diff = (np.diff(dum, axis=0))/number_interpolation_points
    interpolated = np.zeros([(len(array)) * number_interpolation_points, 6])    
    for i in range(array.shape[0]):
        dummy = diff[i]
        interpolated[i * number_interpolation_points:(i + 1) * 
                     (number_interpolation_points)] = dummy
    return interpolated


def calculate_stresses(lame_2, sigma_y, beta, strains_path, stresses_path, 
                       sequence_lenght, number_strain_sequences, n_dim, H, ce):
    mkpath(stresses_path + "/sigma_eq_plots")  
    for i in tqdm(range(0, number_strain_sequences)):
        ##############Initial state###############
        sigma_n = np.zeros([6, 1])
        alpha_n = np.zeros([6, 1])           
        # if (i == 0):
        eq_plastic_strain = 0
        K_N = ((2/3)**0.5) * sigma_y
        # K_N = ((2/3)**0.5) * (sigma_y + ((3/2) * H) * eq_plastic_strain) #(3/2 * H ) = H_PRIME 
        delta_lambda = 0
        ##########################################
        j = 0
        ############################################################
        dum_index = 0
        #These lines are here to plot the von mises stress 
        coord_x_sigma_eq = np.array(range(0, sequence_lenght + 1))
        coord_y_sigma_eq = np.zeros(sequence_lenght + 1) 
        ############################################################
        with open(strains_path+'/%i.txt' %(i), 'r') as f:
            strains = np.loadtxt(strains_path +'/%i.txt' %(i))[:,:]
            stresses = np.zeros([sequence_lenght, 6])
            for delta_strains in strains:
                #Step 1. Compute sigma trial
                sigma_trial = sigma_n + \
                np.dot(ce, delta_strains).reshape(6, 1)
                p = trace(sigma_trial)
                dummy = np.zeros([6,1])
                dummy[0:3] = p
                p = dummy #Trace of the sigma_trial
                s = sigma_trial - p #Deviatoric stress of sigma_trial
                E = s - alpha_n # Effective stress
                #Step 2. Check yield condition
                # a = LA.norm(E)
                E_extended = np.zeros([9, 1])
                E_extended[:6, :] = np.copy(E)
                E_extended[6:, :] = np.copy(E[3:, :])
                a = LA.norm(E_extended)
                # Insert J2 calculation ####
                # J2 = 0.5 * ((s * s).sum()) #(Eq 1.156 in Heidelberg and https://www.continuummechanics.org/tensornotationbasic.html)
                # J2 = 0.5 * ((s**2).sum() + (s[:, 3:]**2).sum()) #(Eq 1.156 in Heidelberg and https://www.continuummechanics.org/tensornotationbasic.html)
                J2 = 0.5 * ((s[:3, :]**2).sum()) + (s[3:, :]**2).sum() #(Eq 1.156 in Heidelberg and https://www.continuummechanics.org/tensornotationbasic.html)
                vmises_stress = (3 * J2) ** 0.5
                ############################
                if (a > K_N):
                    n = E/a
                    delta_lambda = (a-K_N)/(2 * lame_2 + H)
                    sigma_n = sigma_trial - 2 * lame_2 * \
                    delta_lambda * n
                    K_N = K_N + beta * H * delta_lambda
                    # For isotropic hardening                   
                    alpha_n = alpha_n + (1-beta) * H * delta_lambda * n
                    # eq_plastic_strain = eq_plastic_strain + ((2/3) ** 0.5) *  delta_lambda
                    # K_N = ((2/3)**0.5) * (sigma_y + ((3/2) * H) * eq_plastic_strain) #(3/2 * H ) = H_PRIME 
                    # print(eq_plastic_strain, "\n")
                else:
                    sigma_n = sigma_trial
                ### Check J2. This is not needed in the main calculation.
                mean_trace = trace(sigma_n)
                p_new = np.zeros(p.shape)
                p_new[0:3, 0] = mean_trace
                deviatoric_new = sigma_n - p_new
                # J2_new = 0.5 * ((deviatoric_new * deviatoric_new).sum())
                # J2_new = 0.5 * ((deviatoric_new ** 2).sum() + (deviatoric_new[:, 3:] ** 2).sum())
                J2_new = 0.5 * ((deviatoric_new[:3, :]**2).sum()) + (deviatoric_new[3:, :]**2).sum() #(Eq 1.156 in Heidelberg and https://www.continuummechanics.org/tensornotationbasic.html)
                vmises_stress_new = (3 * J2_new) ** 0.5
                # dev_norm = LA.norm(deviatoric_new)
                ########################################################
                #For plotting sigma_eq at every iteration, turn this line on
                coord_y_sigma_eq[j+1] =  vmises_stress_new
                ##########################################################
                stresses[j, :] = sigma_n.reshape(6)
                j = j + 1
        ##################################################################
       
        # print(eq_plastic_strain, "\n")
       
        #For plotting sigma_eq at every iteration, turn these lines on
        if ( i % 200 == 0):#( i % 200 == 0):
            plot_name  = '/test_sample_number_' + str(i) + '.png'
            plt.figure(figsize=(15, 10))
            ax = plt.subplot(111) #
            plt.plot(coord_x_sigma_eq, coord_y_sigma_eq/1e9, color ="black")
            # Shrink current axis by 20%
            box = ax.get_position() #
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            grid_x_ticks = np.arange(0, len(coord_x_sigma_eq))
            ax.set_xticks(grid_x_ticks, minor=True)
            ax.grid(which='both', linestyle='--')
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlabel("time steps", fontsize = 16)
            plt.ylabel("von Mises stress (GPa)", fontsize = 16)
            plt.grid(True)
            plt.savefig(stresses_path + "/sigma_eq_plots" + plot_name, 
                        bbox_inches='tight')
            plt.close()
        ##################################################################
        np.savetxt('./' + stresses_path + '/' + '%i.txt' %(i), stresses,  
                   delimiter = ' ')
        
def trace(array):
    # diag = np.diag(array)
    # if (array.shape[0] == 3):
    #     return (np.sum(array[0:2])) / 3
    # else:
    return (np.sum(array[0:3])) / 3
    

def create_scaled_dataset(x_scaler, y_scaler, x_train, y_train, x_validation, 
                          y_validation, x_test, y_test):
    
    x_scaler.fit(x_train.reshape([x_train.shape[0] * x_train.shape[1], 
                                     x_train.shape[2]]))

    y_scaler.fit(y_train.reshape([y_train.shape[0] * y_train.shape[1], 
                                     y_train.shape[2]]))

    
    x_train_scaled = x_scaler.transform(x_train.reshape([x_train.shape[0] *
                                                         x_train.shape[1], 
                                                         x_train.shape[2]]))
    x_train_scaled = x_train_scaled.reshape(x_train.shape)

    y_train_scaled = y_scaler.transform(y_train.reshape([y_train.shape[0] *
                                                         y_train.shape[1], 
                                                         y_train.shape[2]]))

    y_train_scaled = y_train_scaled.reshape(y_train.shape)

    x_validation_scaled = x_scaler.transform(
                                 x_validation.reshape([x_validation.shape[0] *
                                                       x_validation.shape[1], 
                                                       x_validation.shape[2]]))
    
    x_validation_scaled = x_validation_scaled.reshape(x_validation.shape)

    y_validation_scaled = y_scaler.transform(
                                  y_validation.reshape([y_validation.shape[0] *
                                                         y_validation.shape[1], 
                                                       y_validation.shape[2]]))

    y_validation_scaled = y_validation_scaled.reshape(y_validation.shape)


    x_test_scaled = x_scaler.transform(x_test.reshape([x_test.shape[0] *
                                                       x_test.shape[1], 
                                                       x_test.shape[2]]))
    x_test_scaled = x_test_scaled.reshape(x_test.shape)

    y_test_scaled = y_scaler.transform(y_test.reshape([y_test.shape[0] *
                                                       y_test.shape[1], 
                                                       y_test.shape[2]]))
    y_test_scaled = y_test_scaled.reshape(y_test.shape)


    return x_train_scaled, y_train_scaled, x_validation_scaled, \
y_validation_scaled, x_test_scaled, y_test_scaled


def serilaise_dataset(x_train, y_train, x_validation, y_validation, x_test, 
                      y_test, y_prediction, route):
    with open(route + 'x_training.pkl', 'wb') as f:
        pickle.dump(x_train, f)
    f.close()
    with open(route + 'y_training.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    f.close()
    with open(route + 'x_validation.pkl', 'wb') as f:
        pickle.dump(x_validation, f)
    f.close()
    with open(route + 'y_validation.pkl', 'wb') as f:
        pickle.dump(y_validation, f)
    f.close()
    with open(route + 'x_test.pkl', 'wb') as f:
        pickle.dump(x_test, f)
    f.close()
    with open(route + 'y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    f.close()
    with open(route + 'y_prediction.pkl', 'wb') as f:
        pickle.dump(y_prediction, f)
    f.close()


def build_elastic_moduli(n_dim, lame_1, lame_2):
    #ce is given by Eq 4.2.8 in Sadd, Martin
    ce = np.eye(6)
    if (n_dim == 3):
        ce = 2 * lame_2 * ce
        for i in range(0, n_dim):
            ce[i, 0:n_dim] = lame_1
            ce[i, i] = ce[i, i] + 2 * lame_2
    else:
        #ce is given by equation 7.1.3 in Sadd, Martin
        ce[0:2,:] = 2 * lame_2 * ce[0:2,:]
        ce[3,:] = 2 * lame_2 * ce[3,:]
        ce[0:2,0:2] = ce[0:2,0:2] + lame_1
        ce[2, 0:2] = ce[2, 0:2] + lame_1
        ce[2, 2] = 0.0
        ce[3, 3] = 2 * lame_2
        ce[4:6, 4:6] = 0.0
    return ce