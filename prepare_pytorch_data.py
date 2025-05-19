import torchvision
import subprocess

me = subprocess.run("whoami", capture_output=True)
me = me.stdout.decode('utf8').strip()
data = torchvision.datasets.mnist.FashionMNIST(f"${WORK}/pytorch_datasets", download=True)
