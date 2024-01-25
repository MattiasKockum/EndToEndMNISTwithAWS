import os
import gzip
import numpy as np

from sagemaker.pytorch import PyTorchModel
from sagemaker import Session
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()


sess = Session()

role = os.getenv("role")

pt_mnist_model_data = os.getenv("pt_mnist_model_data")

model = PyTorchModel(
    entry_point="inference.py",
    source_dir="code",
    role=role,
    model_data=pt_mnist_model_data,
    framework_version="1.5.0",
    py_version="py3",
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.c5.xlarge",
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
)


def mnist_to_numpy(data_dir="data", images_file="t10k-images-idx3-ubyte.gz", labels_file="t10k-labels-idx1-ubyte.gz"):

    with gzip.open(os.path.join(data_dir, images_file), "rb") as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

    with gzip.open(os.path.join(data_dir, labels_file), "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return (images, labels)

mnist = mnist_to_numpy()

indexes = [i for i in range(len(mnist[0]))]
np.random.shuffle(indexes)
indexes = indexes[:16]

data = {"inputs": np.expand_dims(mnist[0][indexes], axis=1).tolist()}

res = predictor.predict(data)

predictor.delete_endpoint()


images = mnist[0][indexes]
targets = mnist[1][indexes]

for i in range(16):
    image = images[i]
    target = targets[i]
    prediction = np.array(res[i])
    prediction -= min(prediction)
    prediction /= sum(prediction)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image)
    axes[1].bar([i for i in range(10)], prediction)
    axes[1].set_xticks([i for i in range(10)])
    fig.savefig(f"{i}.png")
    fig.show()
    print(f"Label : {target}, Prediction : {prediction.argmax()}")

