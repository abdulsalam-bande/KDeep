# Script for extracting features from pdb-bind complexes.import
# Mahmudulla Hassan
# Last modified: 09/10/2018

import htmd.ui as ht
import htmd.molecule.voxeldescriptors as vd
import htmd.smallmol.smallmol as sm
import csv
from tqdm import *
import os
import pickle
import numpy as np
import multiprocessing as mp
from sklearn.model_selection import train_test_split
import h5py
from oddt import toolkit
from oddt import datasets

# Directory paths
data_dir = "../dataset"
pdbbind_dir = os.path.join(data_dir, "refined-set-2016/")
pdbbind_dataset = datasets.pdbbind(home=pdbbind_dir, default_set='refined', version=2016)


def get_pdb_complex_feature(protein_file, ligand_file):
    """ Returns voxel features for a pdb complex """

    def get_prop(mol, left_most_point):
        """ Returns atom occupancies """
        n = [24, 24, 24] # Voxel size
        
        # Get the channels
        channels = vd._getAtomtypePropertiesPDBQT(mol)
        sigmas = vd._getRadii(mol)
        channels = sigmas[:, np.newaxis] * channels.astype(float)
        
        # Choose the grid centers
        centers = vd._getGridCenters(llc=left_most_point, N=n, resolution=1)
        centers = centers.reshape(np.prod(n), 3)
        
        # Extract the features and return
        features = vd._getOccupancyC(mol.coords[:, :, mol.frame], centers, channels)
        return features.reshape(*n, -1)
    
    # Generate the HTMD Molecule objects
    protein_mol = ht.Molecule(protein_file)
    ligand_mol = ht.Molecule(ligand_file)
    
    # Find the left most point. Half of the voxel's length is subtracted from the center of the ligand
    left_most_point = list(np.mean(ligand_mol.coords.reshape(-1, 3), axis=0) - 12.0)    
    
    # Get the features for both the protein and the ligand. Return those after concatenation.
    protein_featuers = get_prop(protein_mol, left_most_point)
    ligand_features = get_prop(ligand_mol, left_most_point)
    
    return np.concatenate((protein_featuers, ligand_features), axis=3)


def get_pdb_features(ids, sets="refined"):
    """ Returns features for given pdb ids"""
    pdb_ids = []
    pdb_features = []

    for pdbid in tqdm(ids):
        protein_file = os.path.join(pdbbind_dir, pdbid, pdbid + "_protein.pdbqt")
        ligand_file = os.path.join(pdbbind_dir, pdbid, pdbid + "_ligand.pdbqt")
        if not os.path.isfile(protein_file) or not os.path.isfile(ligand_file): continue

        try:
            features = get_pdb_complex_feature(protein_file, ligand_file)
        except Exception as e:
            #print("ERROR in ", pdbid , " ", str(e))
            continue

        pdb_ids.append(pdbid)
        pdb_features.append(features)
    
    # Convert the list of features as numpy array and return
    data_x = np.array(pdb_features, dtype=np.float32)
    data_y = np.array([pdbbind_dataset.sets[sets][_id] for _id in pdb_ids], dtype=np.float32)

    return data_x, data_y


def get_features():
    """ Returns features for all the complexes in the dataset. """ 
    # List ids in the core set
    core_ids = list(pdbbind_dataset.sets['core'].keys())
    # List ids in the refined set
    refined_ids = list(pdbbind_dataset.sets['refined'].keys()) 
    # remove core ids from the refined set.
    refined_ids = [i for i in refined_ids if i not in core_ids]
    
    # Get the features 
    print("Extracting features for the core set")
    core_x, core_y = get_pdb_features(core_ids, sets="core")
    print("Extracting features for the refined set")
    refined_x, refined_y = get_pdb_features(refined_ids)    
    
    return core_x, core_y, refined_x, refined_y


def main():
    # Get the features
    test_x, test_y, train_x, train_y = get_features()
    # Split it
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=1)
    print("Shapes in the training, test and the validation set: ", train_x.shape, test_x.shape, valid_x.shape)

    # Save it
    print("Saving the data in data.h5")
    h5f = h5py.File(os.path.join(data_dir, "data.h5"), 'w')
    h5f.create_dataset('train_x', data=train_x)
    h5f.create_dataset('train_y', data=train_y)
    h5f.create_dataset('valid_x', data=valid_x)
    h5f.create_dataset('valid_y', data=valid_y)
    h5f.create_dataset('test_x', data=test_x)
    h5f.create_dataset('test_y', data=test_y)
    h5f.close()


if __name__=="__main__": main()