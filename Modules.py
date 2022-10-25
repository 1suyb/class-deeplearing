import numpy as np
eps = 1e-8

class Mul :
    def __init__(self, W) :
        self.W = W
    
    def forward(self, X) :
        self.X = X
        return np.dot(self.X, self.W)

    def backward(self,dY) :
        X = self.X
        W = self.W
        dX = np.dot(dY, W.T)
        dY = np.dot(X.T, dY)
        return (dX,dY)

class Add :
    def __init__ (self,B) :
        self.B = B
    def forward(self,X) :
        return X+self.B
    def backward(self,dY) :
        db = np.sum(dY,axis=0)
        return (dY,db)

class Repeat :
    def forward(self,X,N,axis) :
        self.axis = axis
        return np.repeat(X,N,axis=axis)
    def backward(self,dy) :
        return np.sum(dy, axis=self.axis, keepdims=True)

class Unsqueeze :
    def forward(self, X, axis) :
        self.axis = axis
        return np.expand_dims(X,axis = axis)
    def backward(self,dY) :
        return np.squeeze(dY,axis = self.axis)

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

class ReLU:
    def forward(self,X) :
        Y = X.copy()
        self.mask = (X<0)
        Y[self.mask] = 0
        return Y
    def backward(self,dY) :
        dX = np.ones_like(dY)
        dX[self.mask] = 0.
        return dX

class SigmoidLayer :
    def forward(self,X) :
        Y = 1./(1.+np.exp(-X))
        self.Y = Y
        return Y
    def backward(self,dY) :
        Y = self.Y
        return dY*Y*(1-Y)

class SoftMax : 
    def forward(self,X) :
        c= np.max(X)
        exp_X = np.exp(X-c)
        sum_exp_X = np.sum(exp_X)
        Y = exp_X/sum_exp_X
        self.Y = Y
        return Y
    def backward(self,dY) :
        T = -dY*self.Y
        return self.Y-T
        
    def backward(self,dY,T) :
        return self.Y-T

class BinaryEntropy :
    def forward(self,T,X) :
        self.T = T; self.X = X
        Y = -T*np.log(eps+X) - (1.-T)*np.log(eps+1.-X)
        return Y
    def backward(self,dY) :
        T = self.T
        X = self.X
        return -dY*(T/(X+eps) + (1.-T)/(eps+1.-X))

class CrossEntropy :
    def __init__(self,T) :
        self.T = T  
    def forward(self,X) :
        self.X = X
        Y = - np.sum(self.T*np.log(eps+X),axis=1)
        return Y
    def backward(self,dY) :
        T = self.T; X = self.X
        return - dY*(T/X) 

def init_Xavier_weights(n,m) :
    W = np.sqrt(1./(n+m))*np.random.randn(n,m)
    b = np.zeros(m)
    return W,b

class Linear :
    def __init__(self,W,b) :
        self.Mul = Mul(W)
        self.Add = Add(b)
    def forward(self,X) :
        Z = self.Mul.forward(X)
        Y = self.Add.forward(Z)
        return Y
        
    def backward(self,dY) :
        dZ,B = self.Add.backward(dY)
        db = np.sum(B,axis=0)
        dX,dW = self.Mul.backward(dZ)
        return dX, dW, db

class CrossEntropyWithSoftMax :
    def __init__(self,T) :
        self.SoftMax = SoftMax()
        self.CrossEntropy = CrossEntropy(T)

    def forward(self,X) :
        Z = self.SoftMax.forward(X)
        Y = self.CrossEntropy.forward(Z)
        return Y
    def backward(self,dY) :
        dZ = self.CrossEntropy.backward(dY)
        dX = self.SoftMax.backward(dY,self.CrossEntropy.T)

