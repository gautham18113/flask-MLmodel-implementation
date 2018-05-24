# flask-MLmodel-implementation

STRUCTURE
=========
data\
  --test.csv
  --train.csv
model\
  --model_v1.pk
loan_pred.py
preprocess.py
server.py
train.py

FLASK PARAMETERS
================
SERVER = Localhost
ENVIRONMENT = Development
DEBUG = True

MODULES
=======
loan_pred.py - Main module responsible for training the model and subsequently implement unit tests.
train.py - Module to train the model and store it as a pickle file.
preprocess.py - Module to perform data cleaning and custom transformations to be fed into the pipeline.
server.py - Contains flask application get and post methods, where the data to be predicted is sent a JSON payload with request.post.
          - The subsequent GET method feeds this data through the model and sends the predictions as response.
