import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
counter = 0
newDataTest = np.zeros(25200)

if len(sys.argv) < 2:
    print(
        "Uso: python smooth.py <path para o arquivo csv>  <path para o arquivo de saida> "
    )
dRef = pd.read_csv(sys.argv[1])
dRef = np.squeeze(dRef)
dTesStart = pd.read_csv(sys.argv[2],usecols=[1])
dTesEnd = pd.read_csv(sys.argv[2],usecols=[2])
# print(dTes["end"][0])

# print(dataTest)

for start,end in zip(dTesStart["start"],dTesEnd["end"]):
    newDataTest[start:end] = 1

size = len(dRef)
newDataTest = newDataTest[0:size]
for i in newDataTest:
    if (newDataTest==dRef).all():
        counter = counter + 1

acuracy = counter/size

print(acuracy)
# plt.plot(dRef)
plt.plot(newDataTest)
plt.show()