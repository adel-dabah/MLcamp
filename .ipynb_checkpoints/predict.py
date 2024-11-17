# %%
import pickle

from flask import Flask


# %%
input_file='model_C=10.bin'
f_in= open(input_file,'rb')
dv,model=pickle.load(f_in)
f_in.close()

# %% [markdown]
# test it 

# %%
user={'customerid': '5575_gnvde',
 'gender': 'male',
 'seniorcitizen': 0,
 'partner': 'no',
 'dependents': 'no',
 'tenure': 10,
 'phoneservice': 'yes',
 'multiplelines': 'no',
 'internetservice': 'dsl',
 'onlinesecurity': 'yes',
 'onlinebackup': 'no',
 'deviceprotection': 'yes',
 'techsupport': 'no',
 'streamingtv': 'no',
 'streamingmovies': 'no',
 'contract': 'monthly',
 'paperlessbilling': 'no',
 'paymentmethod': 'mailed_check',
 'monthlycharges': 56.95,
 'totalcharges': 19.5
 #'churn': 0
 }

# %%


from flask import request
from flask import jsonify
app=Flask('predict')
@app.route('/predict',methods=['POST'])
def predict():
    customer=request.get_json()
    x_user= dv.transform([customer])
    x_pred=model.predict_proba(x_user)[0,1]
    churn=x_pred>=0.5
    result={
        'churn_proba':x_pred,
        'churn': bool(churn)
    }
    #print (x_pred)

    return jsonify(result)
if __name__=="__main__":
    app.run(debug=True,host='0.0.0.0',port=9696)