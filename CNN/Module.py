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

def im2col(X, filtersize, padding, strride, layertype) :
    outputsize_h = int((X.shape[2] + 2*padding - filtersize)/strride) + 1  # output size
    outputsize_w = int((X.shape[3] + 2*padding - filtersize)/strride) + 1
    X = np.pad(X,((0,0),(0,0),(padding,padding),(padding,padding)),constant_values=(0,))
    X_col = np.zeros([X.shape[0],X.shape[1], filtersize,filtersize,outputsize_h,outputsize_w])
    for i in range(filtersize) :
        i1 = i; i2 = i1 + strride*outputsize_h
        for j in range(filtersize) :
            j1 = j; j2 = j1 + strride*outputsize_w
            X_col[:,:,i,j,:,:] = X[:,:,i1:i2:strride,j1:j2:strride]
    if layertype == 'conv' :
        X_col = np.transpose(X_col,[0,4,5,1,2,3])
        X_col = np.reshape(X_col,[X.shape[0]*outputsize_h*outputsize_w,X.shape[1]*filtersize*filtersize])
    elif layertype == 'pool' :
        X_col = np.transpose(X_col,[0,1,4,5,2,3])
        X_col = np.reshape(X_col,[X.shape[0]*outputsize_h*outputsize_w*X.shape[1],filtersize*filtersize])
    return X_col, outputsize_h, outputsize_w

# def col2im(X_col,N,n_i,n_ic,n_f,n_o,p,s,layer_type) :
# if layer_type == 'conv' :
# X_col = np.reshape(X_col,[N,n_o,n_o,n_ic,n_f,n_f])
# X_col = np.transpose(X_col,[0,3,4,5,1,2])
# elif layer_type == 'pool' :
# X_col = np.reshape(X_col,[N,n_ic,n_o,n_o,n_f,n_f])
# X_col = np.transpose(X_col,[0,1,4,5,2,3])
# X = np.zeros([N,n_ic,n_i+2*p,n_i+2*p])
# for i in range(n_f) :
# i1 = i; i2 = i1 + s*n_o
# for j in range(n_f) :
# j1 = j; j2 = j1 + s*n_o
# X[:,:,i1:i2:s,j1:j2:s] = X_col[:,:,i,j,:,:]
# X = X[:,:,p:n_i+1,p:n_i+1]
# return X


class Conv2d :
    def __init__(self,inchannel, outchannel=1, fildtersize=3, padding = 0, strride = 1,) : 
        self.filtersize=fildtersize
        self.padding = padding
        self.strride = strride
        self.outchannel = outchannel
        self.inchannel = inchannel
        self.W = np.sqrt(1./(fildtersize+fildtersize))*np.random.randn(inchannel,outchannel,fildtersize,fildtersize)
        self.mul = Mul()
    def forward(self, X):
        self.X_col, oh, ow = im2col(X,self.filtersize,self.padding,self.strride,'conv')
        self.W_col = self.W.reshape([self.outchannel, self.inchannel*self.filtersize*self.filtersize])
        self.W_col = self.W_col.T
        Y = self.mul.forward(self.X_col,self.W_col)
        Y = Y.reshape([X.shape[0],-1,self.outchannel])
        Y = np.transpose(Y,[0,2,1])
        Y = Y.reshape(X.shape[0],self.outchannel,oh,ow)
        return Y
    
    def backword(self,dY) :
        dX,dW = self.mul.backward(dY)
        return dX,dW

class maxpooling :
    def __init__(self,kernelsize, padding=0) :
        self.kernelsize = kernelsize
        self.padding = padding

    def forward(self,X) :
        self.batch = X.shape[0]
        self.inchannel = X.shape[1]
        X_col,self.oh, self.ow = im2col(X,self.kernelsize,self.padding,self.kernelsize-1,'pool')
        Y = np.max(X_col,axis=1)
        Y = np.expand_dims(Y,axis=1)
        Y_index = np.argmax(X_col,axis=1)
        Y_index = np.expand_dims(Y_index,axis=1)
        self.Y_index = np.concatenate((Y_index,Y),axis=1)
        Y = Y.reshape([X.shape[0],X.shape[1],self.oh,self.ow])
        return Y

    def backward(self,dy) :
        dx = np.zeros([self.batch*self.oh*self.ow*self.inchannel,self.kernelsize*self.kernelsize])
        print(self.Y_index[:,0])
        print(dx.shape)
        for i,j in zip(dx, self.Y_index) :
            i[int(j[0])] = j[1]
        self.grad = dx
        return dx
