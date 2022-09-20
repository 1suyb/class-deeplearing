from operator import truediv
import numpy as np
import seaborn as sns
import matplotlib as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import BinaryAccuracy

if __name__ == "__main__" :
    train_x = pd.read_csv(r".\Dataset\titanic_train.csv")
    test_x = pd.read_csv(r".\Dataset\titanic_test.csv")

    train_x.drop(['age','deck'],inplace=True,axis=1)
    _,train_x['sex']=np.unique(train_x['sex'],return_inverse=True)
    _,train_x['embarked'] = np.unique(train_x['embarked'],return_inverse=True)
    _,train_x['embark_town'] = np.unique(train_x['embark_town'],return_inverse=True)
    _,train_x['class'] = np.unique(train_x['class'],return_inverse=True)
    _,train_x['who'] = np.unique(train_x['who'],return_inverse=True)
    _,train_x['adult_male'] = np.unique(train_x['adult_male'],return_inverse=True)
    _,train_x['alive'] = np.unique(train_x['alive'],return_inverse=True)
    _,train_x['alone'] = np.unique(train_x['alone'],return_inverse=True)

    test_x.dropna(inplace=True)
    test_x.drop(['age','deck'],inplace=True,axis=1)
    _,test_x['sex']=np.unique(test_x['sex'],return_inverse=True)
    _,test_x['embarked'] = np.unique(test_x['embarked'],return_inverse=True)
    _,test_x['embark_town'] = np.unique(test_x['embark_town'],return_inverse=True)
    _,test_x['class'] = np.unique(test_x['class'],return_inverse=True)
    _,test_x['who'] = np.unique(test_x['who'],return_inverse=True)
    _,test_x['adult_male'] = np.unique(test_x['adult_male'],return_inverse=True)
    _,test_x['alive'] = np.unique(test_x['alive'],return_inverse=True)
    _,test_x['alone'] = np.unique(test_x['alone'],return_inverse=True)

    train_y = np.ravel(train_x.survived)
    test_y = np.ravel(test_x.survived)
    train_x.drop(['survived'],inplace=True,axis=1)
    test_x.drop(['survived'],inplace=True,axis=1)

    model = Sequential()
    model.add(Dense(1,activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    model.fit(train_x,train_y,epochs=50,batch_size=1,verbose=1)

    Predict = model.predict(test_x).flatten()
    test = BinaryAccuracy()
    test.update_state(test_y, Predict)
    print(test.result().numpy())


    