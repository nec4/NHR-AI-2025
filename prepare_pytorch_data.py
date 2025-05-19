import torchvision
import os
WORK = os.environ["WORK"]
data = torchvision.datasets.mnist.FashionMNIST(f"{WORK}/pytorch_datasets", download=True)
