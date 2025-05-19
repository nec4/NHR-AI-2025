import torchvision
import os

data = torchvision.datasets.mnist.FashionMNIST(f"${os.environ["WORK"]}/pytorch_datasets", download=True)
