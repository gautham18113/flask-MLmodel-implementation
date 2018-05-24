# flask-MLmodel-implementation

STRUCTURE
=========
data\\<br />
  --test.csv\\<br />
  --train.csv\\<br />
model\\<br />
  --model_v1.pk\\<br />
loan_pred.py<br /><br />
preprocess.py<br />
server.py<br />
train.py<br />

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
