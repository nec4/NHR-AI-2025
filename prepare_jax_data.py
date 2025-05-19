import tensorflow_datasets as tfds
import subprocess
import os

os.environ["TFDS_DATA_DIR"] = f"{WORK}/jax_datasets" 
data = tfds.load("fashion_mnist", batch_size=512, split=["train[0%:80%]", "train[80%:100%]"])
