import torch
from model import vggnet
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vggnet().to(device)
summary(model, input_size=(16, 24, 24, 24))