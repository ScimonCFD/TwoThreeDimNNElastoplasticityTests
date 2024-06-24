## Assessing Neural Networks in Predicting Stress-Strain Behavior with Isotropic Hardening

### Description

This Python code is designed for engineers and researchers in materials science to explore the potential of neural networks in predicting stress-strain behavior in materials exhibiting 2D/3D isotropic hardening. The code consists of three main components:

1. **Generation of 2D/3D Strain Sequences:**
   - The code generates a set of 1D strain sequences, which represent the deformation of a material under load. These sequences can be customized for specific test scenarios or data requirements.

2. **Calculation of Corresponding Stresses with Isotropic Hardening:**
   - Utilizing an isotropic hardening model, the code calculates the corresponding stresses for the generated strain sequences. Isotropic hardening is a common material behavior where a material's yield strength increases with plastic deformation.

3. **Evaluation of Neural Networks:**
   - The code implements a recurrent neural network (RNNs). This network is trained and tested using the generated stress-strain data.
   - The goal is to assess the performance of the RNN in approximating the stress-strain behavior, and ultimately, to evaluate if it can serve as a replacement for the theoretical material model. Metrics such as mean squared error and mean absolute error are computed to gauge the accuracy of the neural network predictions.
   - Researchers can fine-tune the neural network architecture, hyperparameters, and training datasets to optimize its performance and explore the potential for more efficient material modeling.

This code provides a valuable tool for investigating the feasibility of using neural networks to predict stress-strain behavior in materials exhibiting isotropic hardening, potentially offering a faster or more versatile alternative to traditional material models.

### How do I get set up? ###

The following libraries are required:

* Python 3.8.12

* NumPy 1.18.5

* TensorFlow 2.4.0

* pip 21.3.1

* Matplotlib 3.3.1

* scikit-learn 1.0.1

* pandas 1.1.1

* tqdm 4.50.2

These libraries can be installed from the supplied environment.yml file using the conda software (https://conda.io). Once conda is installed, the Python environment is installed with:

    conda env create -f environment.yml

The conda environment can be activated with:

    conda activate pythonPal-no-gpu

Then, clone this folder:

    git clone git@github.com:ScimonCFD/2D_3D_NN_Code_For_Thesis.git

Finally, the code can be run with:

    python main.py

## Some results ##

<figure>
  <img src="https://github.com/ScimonCFD/2D_3D_NN_Code_For_Thesis/blob/master/img/2D_control_points.png" alt="">
  <figcaption>Expected sigma xx (continuous gray), sigma yy (continuous blue), sigma zz (continuous green) and sigma xy (continuous red) vs predicted sigma xx (dashed gray), sigma yy (dashed blue), sigma zz (dashed green) and sigma xy (dashed red) behaviour on unseen data using the RNN </figcaption>
</figure>

<figure>
  <img src="https://github.com/ScimonCFD/2D_3D_NN_Code_For_Thesis/blob/master/img/3D_control_points.png" alt="">
  <figcaption>Expected sigma xx (continuous gray), sigma yy (continuous blue), sigma zz (continuous green), sigma xy (continuous red), sigma xz (continuous brown) and sigma yz (continuous purple) vs predicted sigma xx (dashed gray), sigma yy (dashed blue), sigma zz (dashed green) and sigma xy (dashed red), sigma xz (dashed brown) and sigma yz (dashed purple) behaviour on unseen data using the RNN </figcaption>
</figure>



### Who do I talk to? ###

    Simon Rodriguez
    simon.rodriguezluzardo@ucdconnect.ie
    https://www.linkedin.com/in/simonrodriguezl/
    
    Philip Cardiff
    philip.cardiff@ucd.ie
    https://www.linkedin.com/in/philipcardiff/
