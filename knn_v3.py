from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def create_trainset():
    iris = datasets.load_iris()
    iris_data  = np.array(iris.data) 
    iris_target= np.array(iris.target)
    Data = np.column_stack((iris_data,iris_target))
    #所以的訓練樣本各取30個將樣本分成3種集合#
    set0=np.array(Data[0:30])
    set1=np.array(Data[50:80])
    set2=np.array(Data[100:130])
    set=np.row_stack((set0,set1,set2))
    return set

def knn_classify(input_tf,train_Data, k):
    k_class=k
    dist_finish=qksort( dist(input_tf,train_Data))
    Data_No=Compare(dist_finish,dist(input_tf,train_Data),k_class)
    tag_NO=target(Data_No,k_class)
    return tag_NO    


def Compare(D1,D2,k_class):
    D2_len=D2.size
    res=np.array([])
    for i in range (0,k_class):
        for j in range (0,D2_len):
            if(D1[i]==D2[j]):
                res = np.append(res,j)
    return(res)

def target(Data_No,k_class):
    Final_Res=np.array([])
    for i in range (0,k_class):
        x=int(Data_No[i])
        Final_Res=np.append(Final_Res,train_Data[x,4])   
    print(Final_Res)
    y=Final_Res.size
    t0=t1=t2=0
    for j in range (0,y):
        if(int(Final_Res[j])==0):
            t0=t0+1
        elif (int(Final_Res[j])==1):
            t1=t1+1
        elif (int(Final_Res[j])==2):
            t2=t2+1
    tt=np.array([t0,t1,t2])   
    target=-1 
    for p in range (0,3):
     
        if( target < tt[p]):
            target=p         
    return target

def dist(a,b):
    x0=a[0]
    x1=a[1]
    x2=a[2]
    x3=a[3]
    res=np.array([])
    for i in range(0,60):
        
        y0=b[i][0]
        y1=b[i][1]
        y2=b[i][2]
        y3=b[i][3]    
        res = np.append(res, np.sqrt( np.square(x0-y0) + np.square(x1-y1) + np.square(x2-y2) + np.square(x3-y3)  )  )
    return res
def qksort(array):
    if len(array)<2:
        return array
    return qksort([x for x in array if x < array[0]])  + [array[0]] + qksort([x for x in array if x > array [0]])   
if __name__ == '__main__':
    input_tf =np.array([4.7 ,2.8, 5.3, 1.8]) 
    print("測試資料",input_tf)
    train_Data=create_trainset()
    tag_NO=knn_classify(input_tf,train_Data, 3)
    print('分類為第{0:d}類'.format(tag_NO))
