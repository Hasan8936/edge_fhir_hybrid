from flask import Flask,request,jsonify
import os,json
from datetime import datetime
import numpy as np
from .edge_model import HybridDeployedModel
from .detector import EdgeDetector
from .fhir_features import extract_features
from .config import LOG_FILE
app=Flask(__name__)
model=HybridDeployedModel()
det=EdgeDetector(model)
os.makedirs(os.path.dirname(LOG_FILE),exist_ok=True)
def log_alert(e): open(LOG_FILE,"a").write(json.dumps(e)+"\n")
@app.route("/health") 
def h(): return {"status":"ok"},200
@app.route("/fhir/notify",methods=["POST"])
def n():
    data=request.get_json()
    feats,meta=extract_features(data)
    X=feats.reshape(1,-1)
    res=det.analyze(X,meta)
    if res["anom"]:
        log_alert({"ts":datetime.utcnow().isoformat()+"Z",**res})
    return jsonify(res)
app.run(host="0.0.0.0",port=5001)
