import numpy as np
import pandas


def sig(x):
    return 1 / (1 + np.exp(-x))

def d_sig(x):
    return x * (1 - x)

def tg(x):
    return np.exp(2 * x) - 1 / np.exp(2 * x) + 1

def d_tg(x):
    return 1 - tg(x) ** 2

class NN:
    def __init__(self):
        self.w1 = np.random.normal(size=(4, 5))
        self.w2 = np.random.normal(size=(5, 4))
        self.w3 = np.random.normal(size=(4, 3))
        self.w4 = np.random.normal(size=(3, 2))
        self.w5 = np.random.normal(size=(2, 1))
        self.b1 = np.random.normal(size=(5,))
        self.b2 = np.random.normal(size=(4,))
        self.b3 = np.random.normal(size=(3,))
        self.b4 = np.random.normal(size=(2,))
        self.b5 = np.random.normal(size=(1,))
    
    def set_the_best_data(self):
        df = pandas.read_json("iris/train_data.json")
        self.w1 = np.array(df["w1"][0])
        self.w2 = np.array(df["w2"][0])
        self.w3 = np.array(df["w3"][0])
        self.w4 = np.array(df["w4"][0])
        self.w5 = np.array(df["w5"][0])
        self.b1 = np.array(df["b1"][0])
        self.b2 = np.array(df["b2"][0])
        self.b3 = np.array(df["b3"][0])
        self.b4 = np.array(df["b4"][0])
        self.b5 = np.array(df["b5"][0])
    
    def write(self, loss):
        if loss < pandas.read_json("iris/train_data.json")["loss"][0]:
            dataframe = {}
            for key, val in self.__dict__.items():
                dataframe[key] = [val.tolist()]
            dataframe["loss"] = [loss]
            df = pandas.DataFrame(dataframe)
            df.to_json("iris/train_data.json")
    
    def predict(self, x):
        n1 = sig(np.dot(x, self.w1) + self.b1)
        n2 = sig(np.dot(n1, self.w2) + self.b2)
        n3 = sig(np.dot(n2, self.w3) + self.b3)
        n4 = sig(np.dot(n3, self.w4) + self.b4)
        y_pred = sig(np.dot(n4, self.w5) + self.b5)
        print(len(n4), len(self.L))
        return y_pred[0].item()
    
    def train(self, data, trues):
        epochs = 200
        lmd = .1
        for epoch in range(epochs):
            for x, y_true in zip(data, trues):
                n1 = sig(np.dot(x, self.w1) + self.b1)
                n2 = sig(np.dot(n1, self.w2) + self.b2)
                n3 = sig(np.dot(n2, self.w3) + self.b3)
                n4 = sig(np.dot(n3, self.w4) + self.b4)

                y_pred = sig(np.dot(n4, self.w5) + self.b5)
                
                error = y_pred.item() - y_true

                L_n5 = error * d_sig(y_pred)

                L_n4 = np.dot(L_n5, self.w5.T) * d_sig(n4)

                L_n3 = np.dot(L_n4, self.w4.T) * d_sig(n3)

                L_n2 = np.dot(L_n3, self.w3.T) * d_sig(n2)

                L_n1 = np.dot(L_n2, self.w2.T) * d_sig(n1)

                self.L = L_n5
                self.w5 -= lmd * np.outer(n4, L_n5)
                self.w4 -= lmd * np.outer(n3, L_n4)
                self.w3 -= lmd * np.outer(n2, L_n3)
                self.w2 -= lmd * np.outer(n1, L_n2)
                self.w1 -= lmd * np.outer(x, L_n1)

                self.b5 -= lmd * L_n5
                self.b4 -= lmd * L_n4
                self.b3 -= lmd * L_n3
                self.b2 -= lmd * L_n2
                self.b1 -= lmd * L_n1
            if not epoch % 10:
                loss = ((y_pred - y_true).mean() ** 2)
                print(f"{loss:.15f} - {epoch}")
                self.write(loss)

def get_ask(x):
    return {0:"setosa", 5:"versicolor", 10:"virginica"}.get(int(x), False)

def get_species(x):
    return {"Iris-setosa":0, "Iris-versicolor":.5, "Iris-virginica":1}.get(x)

def round_(x):
    a = x * 10
    if int(a):
        if int((a % int(a)) * 10) >= 7:
            return int(a) + 1
        return int(a)
    return 0

nn = NN()
#nn.set_the_best_data()

dct = pandas.read_csv("iris/iris.csv").drop("Id", axis=1)

species = dct["Species"].tolist()

data = np.array(dct.drop("Species", axis=1))

true = list(map(get_species, species))

nn.train(data, true)

input_ = list(map(float, input(">>> ").replace(" ", "").split(",")))
predict = nn.predict(np.array(input_))
ask = get_ask(round_(predict))
if ask:
    print(f"species - {ask}\npredict - {predict:.5f}")
else:
    print(f"ИИ не смог определить цветок\npredict - {predict}")
print(tg(3))
