from flask import Flask,request,render_template,url_for,redirect
import pickle
import numpy as np
import logging
app=Flask(__name__)
logging.basicConfig(filename='sudu.txt',filemode='a',format='%(asctime)s %(levelname)s-%(message)s',
                    datefmt='%Y-%m-%d %H-%M-%S')
try:
    model=pickle.load(open('model.pkl','rb'))
except:
    logging.critical("no model pickel file")
try:
    ct=pickle.load(open('scaler.pkl','rb'))
except:
    logging.critical('no scaler pickel file')

@app.route('/')
def index():
    return render_template('indata.html')
@app.route('/predicted',methods=['GET','POST'])
def i2():
    pred1=[x for x in request.form.values()]
    pred1[0]=float(pred1[0])
    pred1[2]=float(pred1[2])
    pred1[4]=float(pred1[4])
    pred1[5]=float(pred1[5])    
    pred1[6]=float(pred1[6])
    x=np.reshape(pred1,(1,-1))
    pred1=ct.transform(x).toarray()
    pred1=model.predict(pred1)
    if pred1[0]==0:
        msg='expected income is below 50 k'
    else:
        msg='expected income is above 50 k'
    return render_template('predicted.html',pred=msg) 

if __name__=='__main__':
    app.run(debug=True,port=8000)