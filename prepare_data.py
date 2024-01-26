import os
import boto3

from dotenv import load_dotenv

load_dotenv()

import sagemaker

sess = sagemaker.Session()

bucket = sess.default_bucket()

prefix = os.getenv("prefix")

keys = [
    ["train-images-idx3-ubyte.gz", "training/images.gz"],
    ["train-labels-idx1-ubyte.gz", "training/labels.gz"],
    ["t10k-images-idx3-ubyte.gz", "testing/images.gz"],
    ["t10k-labels-idx1-ubyte.gz", "testing/labels.gz"],
]
s3 = boto3.resource("s3")
source_bucket = "sagemaker-sample-files"
for source_key, dest_key in keys:
    copy_source = {
      'Bucket': source_bucket,
      'Key': f"datasets/image/MNIST/{source_key}"
    }
    s3.meta.client.copy(copy_source, bucket, f"{prefix}/data/{dest_key}")


if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/testing"):
    os.makedirs("data/testing")
s3 = boto3.client("s3")
s3.download_file(bucket, "DEMO-mnist/data/testing/images.gz", "data/testing/images.gz")
s3.download_file(bucket, "DEMO-mnist/data/testing/labels.gz", "data/testing/labels.gz")

