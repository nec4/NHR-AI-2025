import torchvision
import subprocess

me = subprocess.run("whoami", capture_output=True)
me = me.stdout.decode('utf8').strip()
sub_dir="nhr_grad_school_2025"
data = torchvision.datasets.mnist.FashionMNIST(f"/home/woody/ihpc/{me}/{sub_dir}/pytorch_datasets", download=True)
