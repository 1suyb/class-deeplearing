import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

datasetFolderPath = r"./Dataset/"
traindsetName = r"penguin_train.csv"
testsetName = r"penguin_test.csv"

def makedata(path,Train=True) :
    x = pd.read_csv(path)
    x.dropna(inplace=True)
    x.drop('sex',inplace=True,axis=1)
    for var in ('species','island') :
        _, x[var] = np.unique(x[var],return_inverse=True)
    
    label = np.ravel(x['species'])
    x.drop(['species'],inplace=True,axis=1)
    x = (x-np.mean(x,axis=0))/np.std(x,axis=0)
    x = torch.from_numpy(x.to_numpy(np.float32))

    label = torch.from_numpy(label)
    y = F.one_hot(label).float()
    print(y.shape)
    
    return x, y

if __name__ == "__main__" :
    train_x,train_y = makedata(datasetFolderPath+traindsetName)
    test_x,test_y = makedata(datasetFolderPath+testsetName,Train=False)

    model = nn.Linear(5,3)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.5)
    lossfn = nn.CrossEntropyLoss()

    epochs =10
    for epoch in range(epochs) :
        pred = model(train_x)
        cost = lossfn(pred,train_y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print(f'epoch : {epoch} | cost : {cost.item()}')
    print("Done")

    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad() :
        predtion = model(test_x)
        test_loss=lossfn(predtion,test_y).item()
        correct += (predtion.argmax(1) == test_y.argmax(1)).type(torch.float).sum().item()
    correct /= len(test_y)
    print(f'testset Loss : {test_loss:>.3f},Accuracy : {correct*100:>.3f}%')
    






