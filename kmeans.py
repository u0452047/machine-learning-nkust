import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

x =np.random.rand(100,2)
x[0]
print(x)
plt.scatter(x[:,0],x[:,1],s=20)
plt.show()
clf = KMeans(n_clusters=4)
print(clf.fit(x))
print(clf.labels_)
print(x)
plt.scatter(x[:,0],x[:,1],c=clf.labels_)
plt.show()


