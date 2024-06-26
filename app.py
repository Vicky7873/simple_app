from flask import Flask,render_template,request, jsonify
import os
import yaml
import joblib
import numpy as np
from prediction_service import prediction



webapp_store = "webapp"

static_dir = os.path.join(webapp_store,"static")
template_dir = os.path.join(webapp_store,"templates")

app = Flask(__name__,static_folder=static_dir,template_folder=template_dir)

@app.route("/",methods=["GET","POST"])
def index():
    if request.method == "POST":
        try:
            if request.form:
                data = dict(request.form)
                data = [list(map(float,data))]
                response = prediction.form_response(data)
                return render_template("index.html",response=response)
            
            elif request.json:
                response = prediction.api_response(request.json)
                return jsonify(response)
        except Exception as e:
            print(e)
            error = {"error":"Something went wrong!! Try again later"}
            return render_template("404.html",error=error)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080,debug=True)