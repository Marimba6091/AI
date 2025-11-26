import numpy as np
import normalize_data as nrd
from PIL import Image
import pandas


def sig(x):
    return 1 / (1 + np.exp(-x))

def d_sig(x):
    return x * (1 - x)

class NN:
    def __init__(self):
        self.__w1 = np.random.normal(size=(64, 32))
        self.__w2 = np.random.normal(size=(32, 16))
        self.__w3 = np.random.normal(size=(16, 10))

        self.__b1 = np.zeros((32,))
        self.__b2 = np.zeros((16,))
        self.__b3 = np.zeros((10,))

        self.__loss = None
    
    @property
    def loss(self):
        return self.__loss
    
    def set_the_best_data(self):
        df = pandas.read_json("number/train_data.json")
        self.__w1 = np.array(df["_NN__w1"][0])
        self.__w2 = np.array(df["_NN__w2"][0])
        self.__w3 = np.array(df["_NN__w3"][0])
        self.__b1 = np.array(df["_NN__b1"][0])
        self.__b2 = np.array(df["_NN__b2"][0])
        self.__b3 = np.array(df["_NN__b3"][0])
    
    def write(self, loss):
        if loss < pandas.read_json("number/train_data.json")["loss"][0]:
            dataframe = {}
            for key, val in self.__dict__.items():
                dataframe[key] = [val.tolist()]
            dataframe["loss"] = [loss]
            df = pandas.DataFrame(dataframe)
            df.to_json("number/train_data.json")

    def predict(self, x):
        x_n = x / 15
        n1 = sig(np.dot(x_n, self.__w1) + self.__b1)
        n2 = sig(np.dot(n1, self.__w2) + self.__b2)
        o = sig(np.dot(n2, self.__w3) + self.__b3)
        return o

    def train(self, data):
        epochs = 100000
        lmd = .01
        for epoch in range(epochs):
            for x, y_true in data:
                x_n = x / 15
                n1 = sig(np.dot(x_n, self.__w1) + self.__b1)
                n2 = sig(np.dot(n1, self.__w2) + self.__b2)
                y_pred = sig(np.dot(n2, self.__w3) + self.__b3)

                error = y_pred - y_true

                D_n3 = error * d_sig(y_pred)

                D_n2 = np.dot(D_n3, self.__w3.T) * d_sig(n2)

                D_n1 = np.dot(D_n2, self.__w2.T) * d_sig(n1)

                self.__w3 -= lmd * np.outer(n2, D_n3)
                self.__w2 -= lmd * np.outer(n1, D_n2)
                self.__w1 -= lmd * np.outer(x_n, D_n1)

                self.__b3 -= lmd * D_n3
                self.__b2 -= lmd * D_n2
                self.__b1 -= lmd * D_n1
            if not epoch % 100:
                self.__loss = ((y_pred - y_true).mean() ** 2)
                print(f"{self.__loss:.15f} - {epoch}")
                self.write(self.__loss)

nn = NN()
is_train = bool(input("1>> "))
is_the_best_data = not bool(input("2>> "))
if is_the_best_data:
    nn.set_the_best_data()
if is_train:
    file = open("number/data.txt")
    file_data = []
    for i in file:
        f = i.split(" ")
        try:
            file_data.append((np.array(list(map(int, f[0].split(",")))), (np.array(list(map(int, f[1].split(",")))))))
        except:
            print(f[0], f[1])
    nn.train(file_data)
file = nrd.Img_to_matrix(Image.open("input.png")).get_data()
test_input = np.array(file)
pr = nn.predict(test_input)
for x, i in enumerate(pr):
    print(f"{x} - {i:.7f}")
print(",".join(list(map(str, file))))
if bool(input("3>> ")):
    nn.write(nn.loss)