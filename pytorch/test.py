import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchviz import make_dot
from model import cnnscore
from dataset import CustomDataset
import sys
import numpy as np
from tqdm import *

if len(sys.argv) == 1:
    print("Please provide a checkpoint..")
    exit(1)
    
train_data_dir = "../dataset/train"
test_data_dir = "../dataset/test"


device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")

print("Loading the model...")
model = cnnscore()
model.cuda()
# DistributedDataParallel will divide and allocate batch_size to all
# available GPUs if device_ids are not set
model = torch.nn.DataParallel(model)
print("Loading the checkpoints...")

    
checkpoint = torch.load(sys.argv[1])
model.load_state_dict(checkpoint['state_dict'])
model.eval()

print("Loading the dataset ...")
dataset = CustomDataset(test_data_dir)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=24, shuffle=False,
    num_workers=8, pin_memory=True)

print("Predicting on the test set ...")
actual = np.zeros(290)
pred = np.zeros(290)
with torch.no_grad():
    for i, (input, target) in enumerate(tqdm(data_loader)):
        input = input.cuda()
        actual[i] = target.mean().numpy()
        _pred = model(input)
        pred[i] = _pred.mean().cpu().numpy()
        
from scipy.stats import pearsonr
print("PEARSON R: {:.3f}".format(pearsonr(actual, pred)[0]))

