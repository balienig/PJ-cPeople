# -*- coding: utf-8 -*-
from flask import Flask
import json

app = Flask(__name__)

# def who(name):
#     return name

# def Accuracy(acuu):
#     return name

# def num(num):
#     return name
  
@app.route('/People/<sentence>', methods = ['GET'])        
def Action(sentence):
    return json.dumps({""})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)