from Dataset.mnist import load_mnist
import numpy as np
import MnistLayers as ly
from PIL import Image
import matplotlib.pyplot as plt

np.random.seed(777)


class Model : 
    def __init__(self,inputsize, outputsize, hiddenlayer) :
        self.linear1 = ly.Linear(inputsize,hiddenlayer)
        self.relu = ly.ReLU()
        self.linear2 = ly.Linear(hiddenlayer, outputsize)
        self.parameters = {}
        self.parameters['linear1'] = self.linear1.parameters
        self.parameters['linear2'] = self.linear2.parameters
    
    def forward(self,X) :
        X1 = self.linear1.forward(X)
        X2 = self.relu.forward(X1)
        X3 = self.linear2.forward(X2)
        return X3

    def backward(self,dY) :
        dX2, dW2, db2 = self.linear2.backward(dY)
        dZ = self.relu.backward(dX2)
        dX1, dW1, db1 = self.linear1.backward(dZ)
        return dW1,db1,dW2,db2
    
    def step(self, stepf, *grad) :
        self.parameters['linear1']['W'] = stepf(self.parameters['linear1']['W'],grad[0])
        self.parameters['linear1']['b'] = stepf(self.parameters['linear1']['b'],grad[1])
        self.parameters['linear2']['W'] = stepf(self.parameters['linear2']['W'],grad[2])
        self.parameters['linear2']['b'] = stepf(self.parameters['linear2']['b'],grad[3])

class Model_2 :
    def __init__(self,inputsize,outputsize) :
        self.linear1 = ly.Linear(inputsize=inputsize,outputsize=outputsize)
        self.parameters = []
        self.parameters.append(self.linear1.parameters)
    def forward(self,X) :
        X1 = self.linear1.forward(X)
        return X1
    def backward(self,dY) :
        dX,dW,db = self.linear1.backward(dY)
        return dW,db
    def step(self,stepf,*grad) :
        self.parameters[0]['W'] = stepf(self.parameters[0]['W'],grad[0])
        self.parameters[0]['b'] = stepf(self.parameters[0]['b'],grad[1])

class SGD : 
    def __init__(self, model, lr) :
        self.model = model
        self.lr = lr
    def step(self,*grad):
        stepf = lambda x,y : x - self.lr*y
        self.model.step(stepf,*grad)

class Momentom : 
    def __init__(self, model, lr, mu) :
        self.model = model
        self.lr = lr
        self.mu = mu
        self.v = [0,0]
    def step(self,*grad):
        self.v[0] = self.mu*self.v[0] + self.lr*grad[0]
        self.v[1] = self.mu*self.v[1]+ self.lr*grad[1]
        stepf = lambda x,y : x - y
        self.model.step(stepf,self.v)

def Accuracy(predict,target) :
    predict = np.argmax(predict,axis=1)
    target = np.argmax(target,axis=1)
    true = 0
    for i in range(len(predict)) :
        if predict[i] == target[i] :
            true += 1
    print("true : ", true)
    return true/len(predict)*100

def img_show(img) :
    img = img.reshape(28,28)
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def drowgraph(X,Y) :
    plt.plot(X,color='green')
    plt.plot(Y)
    plt.show()


(train_feature, train_target),(test_feature,test_target) = load_mnist(flatten=True, normalize = True, one_hot_label=True)
trainrate = int(train_feature.shape[0]*0.9)
x_train, x_validation = train_feature[:trainrate,:], train_feature[trainrate:,:]
y_train, y_validation = train_target[:trainrate,:], train_target[trainrate:,:]

#img_show(x_train[0])
batchsize = 500
lr = 1e-3
model = Model(inputsize = x_train.shape[1], hiddenlayer=50, outputsize=y_train.shape[1])
model = Model_2(inputsize=x_train.shape[1],outputsize=y_train.shape[1])

LossFunction = ly.SoftMaxWithCrossEntropy()

optim = SGD(model,lr=lr)

epochs = 100
loss = 0
lossval = []
losstrain = []
for epoch in range(epochs) :
    for i in range(0,x_train.shape[0],batchsize) :
        x_batch = x_train[i:i+batchsize,:]
        y_batch = y_train[i:i+batchsize,:]
        
        predict = model.forward(x_batch)
        loss += LossFunction.forward(predict,y_batch)
        
        dY = LossFunction.backward()
        dW1,db1 = model.backward(dY)
        optim.step(dW1,db1)
    losstrain.append(loss/int(x_train.shape[0]/batchsize))
    loss = 0
    x = x_validation[:batchsize,:]
    y = y_validation[:batchsize,:]

    predict = model.forward(x)
    lossval.append(LossFunction.forward(predict,y))

    if(epoch%10 == 0) : 
        print(f"{epoch}======================================")
        print("Accuracy : ",Accuracy(predict,y))
        print("loss : ", lossval[len(lossval)-1] )
drowgraph(losstrain,lossval)

test_predict = model.forward(test_feature)
print(Accuracy(test_predict,test_target))






