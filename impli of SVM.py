import numpy as np
import matplotlib.pyplot as plt

def geneda():
    m = 400
    xv=[]
    yv=[]
    for i in range (0,m):
        x1=np.random.uniform(-4,4)
        x2=np.random.uniform(-4,4)
        xv.append([x1,x2])
        val = x1**2+x2**2
        yv.append(-1) if val<=4 else yv.append(1)
    return xv,yv

def drawda(pack,w,b,flag):
    xv = pack[0]
    yv = pack[1]
    plt.figure(figsize=(6,6))
    for i,(data,label) in enumerate(zip(xv,yv)):
        color = 'blue' if label == -1 else 'red'
        marker = 'o' if label == -1 else 'x'
        plt.scatter(data[0],data[1],color=color,marker=marker)
    if flag==1:
        x = np.linspace(-4,4,100)
        y = np.linspace(-4,4,100)
        x,y = np.meshgrid(x,y)
        z = x**2+y**2-4
        plt.contour(x,y,z,levels=[0],color='g')
    [w1,w2,w3]=w
    x1 = np.linspace(-4,4,100)
    x2 = np.linspace(-4,4,100)
    x1,x2 = np.meshgrid(x1,x2)
    z1 = w1*x1**2+np.sqrt(2)*w2*x1*x2+w3*x2**2+b
    plt.contour(x1,x2,z1,levels=[0],c='k')
    plt.show()

class SVM:
    def __init__(self,iter_max=1000,l_rate=0.001,pack=[]):
        self.iter_max = iter_max
        self.l_rate = l_rate
        self.data = np.array(pack[0])
        self.label = (np.array(pack[1])+1)/2
        self.x = self.data[:,0]**2
        self.y = np.sqrt(2)*self.data[:,0]*self.data[:,1]
        self.z = self.data[:,1]**2
        self.coordinate = np.column_stack((self.x,self.y,self.z))

    def Kernel(self,x1,x2):
        return np.dot(x1,x2)**2
     
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def get_weight(self):
        w = np.array([0.01,0.01,0.01])
        b = 0.01
        for _ in range(self.iter_max):
            for data,label in zip(self.coordinate,self.label):
                z = b + np.dot(data,w)
                h = self.sigmoid(z)
                w += self.l_rate*(label-h)*data
                b += self.l_rate*(label-h)
        return w,b

sample = geneda()
svm = SVM(iter_max=2000,l_rate=0.002,pack=sample)
w,b = svm.get_weight()
print(w,b)
drawda(sample,w,b,0)