# Predicting protein-ligand binding affinities using Convolutional Neural Networks (CNN)
This is an implementation of the CNN network architecture described in the following paper 

**KDEEP: Protein–Ligand Absolute Binding Affinity Prediction via 3D-Convolutional Neural Networks** <br>
José Jiménez , Miha Škalič , Gerard Martínez-Rosell , and Gianni De Fabritiis <br>
DOI: 10.1021/acs.jcim.7b00650 <br>

### Requirements
  * Tensorflow : `pip install tensorflow-gpu` or `pip install tensorflow`
  * Keras: `pip install keras`
  * Scikit-learn: `pip install -U scikit-learn`
  * oddt: `conda install -c oddt oddt`
  * tqdm: `pip install tqdm`
  * htmd: 
  ```
  conda config --add channels acellera 
  conda config --add channels psi4 
  conda install htmd 
  ```

### Training the model (by generating augmented data on the fly)
 * First extract the voxel features using the script `feature_extraction_htmd.py` insdie the directory `training_with_htmd`. The script will create the file called `data.h5` inside the `dataset` dir
 * Run the script `train.py` to train the model. Modify the code in the method `main()` to enable/disable data augmentation and tweak other training parameters.

### Training the model (using saved augmented data)
 * Make sure `npy_data` directory is inside the `dataset` directory.
 * Run `train2.py` to train the model.
