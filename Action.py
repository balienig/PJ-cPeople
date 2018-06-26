# -*- coding: utf-8 -*-
from flask import Flask
import json

app = Flask(__name__)

  
@app.route('/Action/<sentence>', methods = ['GET'])        
def Action(sentence):
    A = 'walk'
    return json.dumps({"Behavior":A,"AccuracyBehavior":'13.00'})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)