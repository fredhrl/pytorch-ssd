import pandas as pd 
import sys
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 2:
    print(
        "Uso: python smooth.py <path para o arquivo csv>  <path para o arquivo de saida> "
    )

kernel = np.ones(30)/30



def plot(Data):
    plt.plot(range(len(Y)),Y)
    plt.show()
def count(Data):
    count = 0
    flag = 0
    for i in range(len(Data)) :
        if Data[i] > 0 and flag ==0:
            count += 1
            flag = 1
    
        if Data[i] == 0 and flag ==1:
            flag = 0

    print(count)

data = pd.read_csv(sys.argv[1])
print(np.squeeze(data).shape,kernel.shape)
Y = np.convolve(np.squeeze(data),kernel)
mean = Y.mean()
var = Y.var()
Y[Y<mean] = 0
Y[Y>mean] = 1
plot(Y)
count(Y)

df = pd.DataFrame(Y, columns=["Signal"])

df.to_csv(
            "./" + sys.argv[2].split("/")[-1].split(".")[0] + ".csv",
            header=None,
            index=None,
            sep=" ",
            mode="a",
        )
