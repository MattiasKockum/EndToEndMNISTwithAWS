import os
from dotenv import load_dotenv

import sagemaker
from sagemaker.pytorch import PyTorch

import boto3

load_dotenv()

sess = sagemaker.Session()

role = os.getenv("role")

output_path = "s3://" + sess.default_bucket() + "/DEMO-mnist"

estimator = PyTorch(
    entry_point="train.py",
    source_dir="code",
    role=role,
    framework_version="1.5.0",
    py_version="py3",
    instance_type="ml.c5.xlarge",
    instance_count=1,
    volume_size=250,
    output_path=output_path,
    hyperparameters={
        "batch-size": 128,
        "epochs": 1,
        "learning-rate": 1e-3,
        "log-interval": 10},
    environment={"WANDB_API_KEY": os.getenv("wandb_api_key")}
)


def download_mnist_from_public_s3(data_dir="./data", train=True):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if train:
        images_file = "train-images-idx3-ubyte.gz"
        labels_file = "train-labels-idx1-ubyte.gz"
    else:
        images_file = "t10k-images-idx3-ubyte.gz"
        labels_file = "t10k-labels-idx1-ubyte.gz"

    # download objects
    s3 = boto3.client("s3")
    bucket = f"sagemaker-sample-files"
    for obj in [images_file, labels_file]:
        key = os.path.join("datasets/image/MNIST", obj)
        dest = os.path.join(data_dir, obj)
        if not os.path.exists(dest):
            s3.download_file(bucket, key, dest)


download_mnist_from_public_s3("./data", True)
download_mnist_from_public_s3("./data", False)


# Upload to the default bucket

prefix = "DEMO-mnist"
bucket = sess.default_bucket()
loc = sess.upload_data(path="./data", bucket=bucket, key_prefix=prefix)

channels = {"training": loc, "testing": loc, "validation": loc}

estimator.fit(inputs=channels)

pt_mnist_model_data = estimator.model_data
print("Model artifact saved at:\n", pt_mnist_model_data)


# Download the trained model

if not os.path.exists("models"):
    os.makedirs("models")

s3 = boto3.client("s3")
l = pt_mnist_model_data.split('/')
key = '/'.join(l[3:])
dest = "models/" + l[-1]
s3.download_file(bucket, key, dest)
