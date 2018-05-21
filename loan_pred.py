import os
import pandas as pd
from sklearn.externals import joblib
from flask import Flask, jsonify, request
from train import train
import json
import dill as pickle
import requests
from json2html import *

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World!"


if __name__ == "__main__":
    app.run(debug=True)
    #train()



