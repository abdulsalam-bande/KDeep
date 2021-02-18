# Predicting protein-ligand binding affinities using Convolutional Neural Networks (CNN)
This is an implementation of the CNN network architecture described in the following paper

**KDEEP: Protein–Ligand Absolute Binding Affinity Prediction via 3D-Convolutional Neural Networks** <br>
José Jiménez , Miha Škalič , Gerard Martínez-Rosell , and Gianni De Fabritiis <br>
DOI: 10.1021/acs.jcim.7b00650 <br>

### Requirements
  * Tensorflow : `pip install tensorflow==1.14`
  * Keras: `pip install keras==2.2.4`
  * Scikit-learn: `pip install -U scikit-learn`
  * oddt: `conda install -c oddt oddt`
  * molekulekit: `pip install moleculekit`
  * copy the dataset/refined-set-2016 folder into the dataset folder(this repository) from https://github.com/hassanmohsin/DLSCORE-CNN




### Training the model (by generating augmented data on the fly)
 * First extract the voxel features using the script `Featurizer.ipynb` insdie the directory `training_with_htmd`. The script will create the file called `data.h5` inside the `dataset` dir
 * Run the script `trainer.ipynb` to train the model.
