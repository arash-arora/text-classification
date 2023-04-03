from flask import Flask, request, jsonify
import pickle
from flask_cors import cross_origin
import numpy as np

app = Flask(__name__)
lrmodel = pickle.load(open('models/logisticregression.pkl', 'rb'))
svmmodel = pickle.load(open('models/svm.pkl', 'rb'))
cv = pickle.load(open('models/cv.pkl', 'rb'))

labels = ['ham', 'spam']

def transform(modelName, text):
    str = cv.transform([text])
    if(modelName=="svm"):
        res = {"Accuracy":svmmodel.predict_proba(str)[0], "label": svmmodel.predict(str), "text":str}
    elif(modelName=="lr"):
        res = {"Accuracy":lrmodel.predict_proba(str)[0], "label": lrmodel.predict(str), "text": str}
        
        print(res["Accuracy"])
    else: 
        res = ["Model selection error"]
    return res

@app.route("/api", methods=["GET"])
@cross_origin()
def predict():
    model = request.args.get('model')
    text = request.args.get('text')
    print(text)
    result = transform(model, text)
    response = jsonify({
        "accuracy":max(result["Accuracy"].tolist()), 
        "label": result["label"][0]
        })
    return response

if __name__ == '__main__':
    app.run(port=5000, debug=True)