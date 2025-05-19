import tensorflow_datasets as tfds
import subprocess
import os

me = subprocess.run("whoami", capture_output=True)
me = me.stdout.decode('utf8').strip()
sub_dir="nhr_grad_school_2025"
os.environ["TFDS_DATA_DIR"] = f"/home/woody/ihpc/{me}/{sub_dir}/jax_datasets" 

data = tfds.load("fashion_mnist", batch_size=512, split=["train[0%:80%]", "train[80%:100%]"])
