import torchvision
import subprocess

data = torchvision.datasets.mnist.FashionMNIST(f"${WORK}/pytorch_datasets", download=True)
