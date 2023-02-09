import warnings
import numpy as np

# Utils
def init_Xavier_weights(n,m) :
    W = np.sqrt(1./(n+m))*np.random.randn(n,m)
    b = np.zeros(m)
    return W,b

# Functions
def SoftMax(X) :
    expX = np.exp(X)
    Y = expX/np.sum(expX,axis=1,keepdims=True)
    return Y

def CrossEntropy(X,T) :
        eps = 1e-8
        return np.mean((-1.0)*np.sum(T*np.log(eps+X))/X.shape[0])

# Calculate Layers
class Unsqueeze :
    def forward(self, X, axis) :
        self.axis = axis
        return np.expand_dims(X,axis = axis)
    def backward(self,dY) :
        return np.squeeze(dY,axis = self.axis)

class Repeat :
    def forward(self,X,N,axis) :
        self.axis = axis
        self.unsqueeze = False
        self.model = 0
        if len(X.shape) == 1 :
            self.unsqueeze=True 
            self.model = Unsqueeze()
            X = self.model.forward(X,axis)
        return np.repeat(X,N,axis=axis)
        
    def backward(self,dy) :
        if self.unsqueeze :
            return np.sum(dy, axis=self.axis)
        return np.sum(dy, axis=self.axis, keepdims=True)

class Mul : 
    def forward(self,X,W) :
        if len(X.shape)==1 or len(W.shape)==1 :
            warnings.warn("X, W need matrix, not Vector")
            return
        self.X = X
        self.W = W
        return np.dot(X,W)
    def backward(self,dY) :
        X = self.X 
        W = self.W
        dX = np.dot(dY,W.T)
        dW = np.dot(X.T,dY)
        return (dX,dW)

class Add : 
    def forward(self,X,B) :
        self.isrepeat = False
        self.repeat = 0
        if B.shape != X.shape :
            self.isrepeat = True
            if B.shape != (X.shape[1],) or B.shape == (1,X.shape[1]) :
                warnings.warn(f"B need ({X.shape[1]},) or (1,{X.shape[1]})ndarray")
                return
            self.repeat = Repeat()
            B = self.repeat.forward(B,X.shape[0],axis=0)
        return X+B

    def backward(self,dY) :
        dX = dY
        dB= dY
        if self.isrepeat :
            dB = self.repeat.backward(dY)
        return (dX,dB)

class Squeeze :
    def forward(self,X,axis) :
        self.axis = axis
        return np.squeeze(X,axis=axis)
    def backward(self,dY) :
        return np.expand_dims(dY,axis=self.axis)

class Sum : 
    def forward(self,X,axis) :
        self.axis = axis
        self.N = X.shape[axis]
        return np.sum(X,axis=axis,keepdims=True)
    
    def backward(self,dy) :
        return np.repeat(dy,self.N,axis=self.axis)

# Activation Fucntion Layers
class ReLU:
    def forward(self,X) :
        Y = X.copy()
        self.mask = (X<=0)
        Y[self.mask] = 0
        return Y
    def backward(self,dY) :
        dX = np.ones_like(dY)
        dX[self.mask] = 0.
        return dX

class Sigmoid :
    def forward(self,X) :
        Y = 1./(1.+np.exp(-X))
        self.Y = Y
        return Y
    def backward(self,dY) :
        Y = self.Y
        return dY*Y*(1-Y)

# Loss Function Layer
class SoftMaxWithCrossEntropy : 
    def forward(self,X,T) : 
        self.T = T
        self.Y = SoftMax(X)
        return CrossEntropy(self.Y,T)
    def backward(self) :
        return self.Y-self.T

class Linear :
    def __init__(self,inputsize,outputsize) : 
        self.parameters = {}
        self.parameters['W'], self.parameters['b'] = init_Xavier_weights(inputsize,outputsize)
        self.mul = Mul()
        self.add = Add()
    def forward(self,X) : 
        Z = self.mul.forward(X,self.parameters['W'])
        return self.add.forward(Z,self.parameters['b'])
    
    def backward(self,dY) :
        dZ,db = self.add.backward(dY)
        dX,dW = self.mul.backward(dZ)
        return dX, dW, db
