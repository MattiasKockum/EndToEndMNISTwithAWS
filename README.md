# A test End-To-End MNIST project

## How tu use it

git clone https://github.com/MattiasKockum/EndToEndMNISTwithAWS.git
cd EndToEndMNISTwithAWS
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Fill a .env file with your own data
``` python
prefix = "DEMO-mnist"
role = "..." # Get it from AWS
pt_mnist_model_data = "..." You can get it by running launch_training.py
wandb_api_key = "..." Get it from Weights And Biases
```

python prepare_data.py
python launch_training.py
python deploy.py

Look into outputs directory

TODO :
    Separate MNIST related code
