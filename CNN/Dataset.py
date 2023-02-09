import numpy as np
from PIL import Image
import glob

class Dataset :
    def __init__(self,*args,size=(64,64),color=True) :
        self.size = size
        self.color=color
        img_path1 = glob.glob(args[0] + args[1])
        img_path2 = glob.glob(args[0] + args[2])
        img_path3 = glob.glob(args[0] + args[3])
        self.img_path= img_path1+img_path2+img_path3
        np.random.shuffle(self.img_path)
        self.img = {imgpath : 0 for imgpath in img_path1}
        for imgpath in img_path2 :
            self.img[imgpath] = 1
        for imgpath in img_path3 :
            self.img[imgpath] = 2
        
    
    def __getitem__(self,index) :
        if(self.color) :
             img = Image.open(self.img_path[index])
        else :
            img = Image.open(self.img_path[index]).convert("L")
        img = np.array(img.resize(self.size))
        img= img/255
        return img

    def label(self,index) :
        return self.img[self.img_path[index]]

    def __len__(self) :
        return len(self.img_path)
    
    def random(self) :
        np.random.shuffle(self.img_path)