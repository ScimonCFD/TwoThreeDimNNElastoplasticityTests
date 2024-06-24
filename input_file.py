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
# Material properties and other required constants

# Authors
#  Simon A. Rodriguez, UCD. All rights reserved
#  Philip Cardiff, UCD. All rights reserved


# Material constants
E = 200e9
v = 0.3
LAME_1 = E * v / ((1 + v) * (1 - 2 * v))
LAME_2 = E / (2 * (1 + v))
SIGMA_Y = 1e9 
H_PRIME = 300e6 
H = (2/3) * H_PRIME 
K_N = ((2/3) ** 0.5) * SIGMA_Y                     
BETA = 1#If BETA = 0, no #isotropic hardening. If BETA = 1, no kinematic hardening. 
STRAINS_PATH = 'strains'
STRESSES_PATH = 'stresses'
MAX_ABS_DEFORMATION = 0.025#0.025 #0.02
NUMBER_CONTROL_POINTS = 10 #15#10
NUMBER_INTERPOLATION_POINTS = 10#5
NUMBER_STRAIN_SEQUENCES = 10000 #1000
# N_DIM = 2 #3
SPLITTER = [0.7, 0.2, 0.1] 
PLOTS_PATH = 'Plots'
NUMBER_OF_EPOCHS = 1000 #500
SEED = 2