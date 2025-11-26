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
        self.w1 = np.random.normal(size=(64, 32))
        self.w2 = np.random.normal(size=(32, 16))
        self.w3 = np.random.normal(size=(16, 10))

        self.b1 = np.zeros((32,))
        self.b2 = np.zeros((16,))
        self.b3 = np.zeros((10,))
    
    def set_the_best_data(self):
        df = pandas.read_json("number/train_data.json")
        self.w1 = np.array(df["w1"][0])
        self.w2 = np.array(df["w2"][0])
        self.w3 = np.array(df["w3"][0])
        self.b1 = np.array(df["b1"][0])
        self.b2 = np.array(df["b2"][0])
        self.b3 = np.array(df["b3"][0])
    
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
        n1 = sig(np.dot(x_n, self.w1) + self.b1)
        n2 = sig(np.dot(n1, self.w2) + self.b2)
        o = sig(np.dot(n2, self.w3) + self.b3)
        return o

    def train(self, data):
        epochs = 100000
        lmd = .01
        for epoch in range(epochs):
            for x, y_true in data:
                x_n = x / 15
                n1 = sig(np.dot(x_n, self.w1) + self.b1)
                n2 = sig(np.dot(n1, self.w2) + self.b2)
                y_pred = sig(np.dot(n2, self.w3) + self.b3)

                error = y_pred - y_true

                D_n3 = error * d_sig(y_pred)

                D_n2 = np.dot(D_n3, self.w3.T) * d_sig(n2)

                D_n1 = np.dot(D_n2, self.w2.T) * d_sig(n1)

                self.w3 -= lmd * np.outer(n2, D_n3)
                self.w2 -= lmd * np.outer(n1, D_n2)
                self.w1 -= lmd * np.outer(x_n, D_n1)

                self.b3 -= lmd * D_n3
                self.b2 -= lmd * D_n2
                self.b1 -= lmd * D_n1
            if not epoch % 100:
                loss = ((y_pred - y_true).mean() ** 2)
                print(f"{loss:.15f} - {epoch}")
                self.write(loss)

nn = NN()
is_train = bool(input("1>> "))
is_the_best_data = not bool(input("2>> "))
if is_the_best_data:
    nn.set_the_best_data()
if is_train:
    file = open("number/data.txt")
    file_data = []
    for i in file:
        f = file.readline().split(" ")
        file_data.append((np.array(list(map(int, f[0].split(",")))), (np.array(list(map(int, f[1].split(",")))))))
    nn.train(file_data)
f = nrd.Img_to_matrix(Image.open("input.png")).get_data()
test_input = np.array(f)
pr = nn.predict(test_input)
for x, i in enumerate(pr):
    print(f"{x} - {i:.7f}")
print(f)
