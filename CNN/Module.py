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
        self.dW = np.dot(X.T,dY)
        return (dX,self.dW)
    
    def step(self,stepf) :
        self.dW = stepf(self.W,self.dW)

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
        self.dW = dW
        self.db = db
        return dX

def im2col(X,
            batchsize,
            filtersize,
            outputsize_h,
            outputsize_w,
            padding, 
            stride, 
            layertype) :
    X = np.pad(X,((0,0),(0,0),(padding,padding),(padding,padding)),constant_values=(0,))
    X_col = np.zeros([batchsize,X.shape[1], filtersize,filtersize,outputsize_h,outputsize_w])
    for i in range(filtersize) :
        i1 = i; i2 = i1 + stride*outputsize_h
        for j in range(filtersize) :
            j1 = j; j2 = j1 + stride*outputsize_w
            X_col[:,:,i,j,:,:] = X[:,:,i1:i2:stride,j1:j2:stride]
    if layertype == 'conv' :
        X_col = np.transpose(X_col,[0,4,5,1,2,3])
        X_col = np.reshape(X_col,[batchsize*outputsize_h*outputsize_w,X.shape[1]*filtersize*filtersize])
    elif layertype == 'pool' :
        X_col = np.transpose(X_col,[0,1,4,5,2,3])
        X_col = np.reshape(X_col,[batchsize*outputsize_h*outputsize_w*X.shape[1],filtersize*filtersize])
    return X_col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """(im2col과 반대) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.
    
    Parameters
    ----------
    col : 2차원 배열(입력 데이터)
    input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    img : 변환된 이미지들
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] = col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

def poolcol2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """(im2col과 반대,pool) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.
    
    Parameters
    ----------
    col : 2차원 배열(입력 데이터)
    input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    img : 변환된 이미지들
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, C, out_h, out_w, filter_h, filter_w).transpose(0, 1, 4, 5, 2, 3)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

class Conv2d :
    def __init__(self,inchannel, outchannel=1, fildtersize=3, padding = 0, stride = 1,) : 
        self.filtersize=fildtersize
        self.padding = padding
        self.stride = stride
        self.outchannel = outchannel
        self.inchannel = inchannel
        self.W = np.sqrt(1./(fildtersize+fildtersize+inchannel))*np.random.randn(inchannel,outchannel,fildtersize,fildtersize)
        self.mul = Mul()

    def forward(self, X):
        self.inputshape = X.shape
        batchsize , _, height, width = self.inputshape[0], self.inputshape[1], self.inputshape[2], self.inputshape[3]
        self.outputsize_h = int((X.shape[2] + 2*self.padding - self.filtersize)/self.stride) + 1  # output size
        self.outputsize_w = int((X.shape[3] + 2*self.padding - self.filtersize)/self.stride) + 1

        X_col= im2col(X,batchsize,self.filtersize,self.outputsize_h,self.outputsize_w,self.padding,self.stride,'conv')
        W_col = self.W.reshape([self.outchannel, self.inchannel*self.filtersize*self.filtersize])
        W_col = W_col.T

        Y = self.mul.forward(X_col,W_col)

        Y = Y.reshape([batchsize,-1,self.outchannel])
        Y = np.transpose(Y,[0,2,1])
        Y = Y.reshape(batchsize,self.outchannel,self.outputsize_h,self.outputsize_w)
        return Y
    
    def backword(self,dY) :
        batchsize , _, height, width = self.inputshape[0], self.inputshape[1], self.inputshape[2], self.inputshape[3]
        dY = np.transpose(dY,[0,2,3,1]).reshape(-1,self.outchannel)

        dX,dW = self.mul.backward(dY)
        dW = dW.transpose(1, 0).reshape(self.W.shape)
        dX = col2im(dX,self.inputshape,self.filtersize,self.filtersize,self.stride,self.padding)
        self.dW = dW
        return dX

    def step(self, stepf) :
        self.W = stepf(self.W,self.dW)

class maxpooling :
    def __init__(self,kernelsize, padding=0) :
        self.kernelsize = kernelsize
        self.padding = padding

    def forward(self,X) :
        self.inputshape = X.shape
        self.outputsize_h = int((X.shape[2] + 2*self.padding - self.kernelsize)/(self.kernelsize-1)) + 1  # output size
        self.outputsize_w = int((X.shape[3] + 2*self.padding - self.kernelsize)/(self.kernelsize-1)) + 1
        
        batch,inputchannel,input_H,input_W = self.inputshape[0],self.inputshape[1],self.inputshape[2],self.inputshape[3]

        X_col = im2col(X,batch,self.kernelsize,self.outputsize_h,self.outputsize_w,self.padding,self.kernelsize-1,'pool')

        Y = np.max(X_col,axis=1)
        Y = np.expand_dims(Y,axis=1)
        Y_index = np.argmax(X_col,axis=1)
        Y_index = np.expand_dims(Y_index,axis=1)
        self.Y_index = np.concatenate((Y_index,Y),axis=1)
        Y = Y.reshape([batch,inputchannel,self.outputsize_h,self.outputsize_w])
        return Y

    def backward(self,dY) :
        batch,inputchannel,input_H,input_W = self.inputshape[0],self.inputshape[1],self.inputshape[2],self.inputshape[3]
        dx = np.zeros([batch*self.outputsize_h*self.outputsize_w*inputchannel, self.kernelsize*self.kernelsize])
        dY = dY.reshape(-1,1)
        for i, j, k in zip(dx, self.Y_index, dY) :
            i[int(j[0])] = j[1] * k
        dx = poolcol2im(dx,self.inputshape,self.kernelsize,self.kernelsize,self.kernelsize-1,self.padding)
        self.grad = dx
        return dx

class DenseLayer :
    def __init__(self,outputsize) :
        self.outputsize = outputsize
        self.mul = Mul()
        self.add = Add()
        self.W = None
        self.b = None

    def forward(self,X) : 
        self.inputshape = X.shape
        input = X.reshape(-1,1)
        if self.W is None :
            self.W, self.b = init_Xavier_weights(input.shape[0], self.outputsize)
        Z = self.mul.forward(input.T,self.W)
        Y = self.add.forward(Z,self.b)
        return Y

    def backward(self,dY) :
        dZ, db = self.add.backward(dY)
        dX, dW = self.mul.backward(dZ)
        self.dW = dW
        self.db = db
        dX = dX.reshape(self.inputshape)
        return dX
    
    def step(self,stepf) :
        self.W = stepf(self.W,self.dW)
        self.b = stepf(self.b,self.db)

class ConvNet :
    def __init__(self,inchannel,outsize) :
        self.conv1 = Conv2d(inchannel,5,fildtersize=3)
        self.relu = ReLU()
        self.pooling1 = maxpooling(3)
        self.conv2 = Conv2d(5,10)
        self.relu2 = ReLU()
        self.pooling2 = maxpooling(3)
        self.conv3 = Conv2d(10,20)
        self.relu3 = ReLU()
        self.pooling3 = maxpooling(3)
        self.linear = DenseLayer(outsize)
    
    def forward(self,X) :
        X = self.conv1.forward(X)
        X = self.relu.forward(X)
        X = self.pooling1.forward(X)
        X = self.conv2.forward(X)
        X = self.relu2.forward(X)
        X = self.pooling2.forward(X)
        X = self.conv3.forward(X)
        X = self.relu3.forward(X)
        X = self.pooling3.forward(X)
        X = self.linear.forward(X)
        return X

    def backward(self,dY) :
        dY = self.linear.backward(dY)
        dY = self.pooling3.backward(dY)
        dY = self.relu3.backward(dY)
        dY = self.conv3.backword(dY)
        dY = self.pooling2.backward(dY)
        dY = self.relu2.backward(dY)
        dY = self.conv2.backword(dY)
        dY = self.pooling1.backward(dY)
        dY = self.relu.backward(dY)
        dY = self.conv1.backword(dY)

    def step(self,stepf) :
        dY = self.linear.step(stepf)
        dY = self.conv3.step(stepf)
        dY = self.conv2.step(stepf)
        dY = self.conv1.step(stepf)

class Optimaizer :
    def __init__(self, model, lr) :
        self.model = model
        self.lr = lr
    def step(self) :
        stepf = lambda x,y : x-self.lr * y
        self.model.step(stepf)
