import pymongo
import numpy as np
import matplotlib.pyplot as plt


myclient = pymongo.MongoClient(host='192.168.1.12', port=27017)
ovodata = myclient["guidang4"]['ovodata']

X = []
Y = []
print(ovodata.count())
i = 0
for o in ovodata.find():
    X.append(o['ovo'])
    Y.append(o['yituovo'])
    i += 1
    if i % 1000 == 0:
        print(i)

plt.legend()

plt.scatter(X, Y, s=1, c="#ff1212", marker='o')
plt.show()
