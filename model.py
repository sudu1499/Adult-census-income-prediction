import pandas as pd 
import numpy as np
import pickle
import logging
logging.basicConfig(filename='sudu.txt',filemode='a',format='%(asctime)s %(levelname)s-%(message)s',
                    datefmt='%Y-%m-%d %H-%M-%S')
try:
    data=pd.read_csv('adult.csv')
except:
    logging.critical('no file found')
data=data[data!=' ?']

x=data.iloc[:,[0,1,4,6,10,11,12]].values
y=data.iloc[:,-1].values

from sklearn.impute import SimpleImputer
si=SimpleImputer(missing_values=np.nan,strategy="most_frequent")
si.fit(x)
x=si.transform(x)


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1,3])],remainder='passthrough')
x=ct.fit_transform(x).toarray()


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.2,random_state=1)



from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=12)
knn.fit(xtrain,ytrain)

pickle.dump(knn,open('model.pkl','wb'))
pickle.dump(ct,open('scaler.pkl','wb'))


