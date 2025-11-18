import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class NN:
    def __init__(self):
        self.w1 = np.random.normal(size=(3, 3))
        self.w2 = np.random.normal(size=(3, 2))
        self.w3 = np.random.normal(size=(2, 1))
        self.b1 = np.random.normal(size=(3,))
        self.b2 = np.random.normal(size=(2,))
        self.b3 = np.random.normal(size=(1,))

    def sig(self, x):
        return 1 / (1 + np.exp(-x))
    
    def predict(self, x):
        x = self.scale(x)
        n11 = self.sig(self.w1[0][0] * x[0] + self.w1[1][0] * x[1] + self.b1[0])
        n12 = self.sig(self.w1[0][1] * x[0] + self.w1[1][1] * x[1] + self.b1[1])
        n13 = self.sig(self.w1[0][2] * x[0] + self.w1[1][2] * x[1] + self.b1[2])

        n21 = self.sig(self.w2[0][0] * n11 + self.w2[1][0] * n12 + self.w2[2][0] * n13 + self.b2[0])
        n22 = self.sig(self.w2[0][1] * n11 + self.w2[1][1] * n12 + self.w2[2][1] * n13 + self.b2[1])

        out = self.sig(self.w3[0] * n21 + self.w3[1] * n22 + self.b3[0])
        
        return out
    
    def scale(self, x):
        return x

    
    def train(self, test, trues, lmd = .1, epochs = 5000):
        test = self.scale(test)
        for epoch in range(epochs):
            for x, y_true in zip(test, trues):
                n11_s = (self.w1[0][0] * x[0] + self.w1[1][0] * x[1] + self.w1[2][0] * x[2] + self.b1[0])
                n11_r = self.sig(n11_s)

                n12_s = (self.w1[0][1] * x[0] + self.w1[1][1] * x[1] + self.w1[2][1] * x[2] + self.b1[1])
                n12_r = self.sig(n12_s)

                n13_s = (self.w1[0][2] * x[0] + self.w1[1][2] * x[1] + self.w1[2][2] * x[2] + self.b1[2])
                n13_r = self.sig(n13_s)

                n21_s = (self.w2[0][0] * n11_r + self.w2[1][0] * n12_r + self.w2[2][0] * n13_r + self.b2[0])
                n21_r = self.sig(n21_s)

                n22_s = (self.w2[0][1] * n11_r + self.w2[1][1] * n12_r + self.w2[2][1] * n13_r + self.b2[1])
                n22_r = self.sig(n22_s)

                n3 = (self.w3[0] * n21_r + self.w3[1] * n22_r + self.b3[0])
                y_pred = self.sig(n3)

                error = (y_pred.item() - y_true.item())

                L_n3 =  error * y_pred.item() * (1 - y_pred.item())

                L_n21 = L_n3 * self.w3[0].item() * n21_r * (1 - n21_r)
                L_n22 = L_n3 * self.w3[1].item() * n22_r * (1 - n22_r)

                L_n11 = (L_n21 * self.w2[0][0] + L_n22 * self.w2[0][1]) * n11_r * (1 - n11_r)
                L_n12 = (L_n21 * self.w2[1][0] + L_n22 * self.w2[1][1]) * n12_r * (1 - n12_r)
                L_n13 = (L_n21 * self.w2[2][0] + L_n22 * self.w2[2][1]) * n13_r * (1 - n13_r)

                self.w3[0] -= lmd * (L_n3) * n21_r
                self.w3[1] -= lmd * (L_n3) * n22_r

                self.w2[0][0] -= lmd * L_n21 * n11_r
                self.w2[1][0] -= lmd * L_n21 * n12_r
                self.w2[2][0] -= lmd * L_n21 * n13_r

                self.w2[0][1] -= lmd * L_n22 * n11_r
                self.w2[1][1] -= lmd * L_n22 * n12_r
                self.w2[2][1] -= lmd * L_n22 * n13_r

                self.w1[0][0] -= lmd * L_n11 * x[0]
                self.w1[0][1] -= lmd * L_n12 * x[0]
                self.w1[0][2] -= lmd * L_n13 * x[0]

                self.w1[1][0] -= lmd * L_n11 * x[1]
                self.w1[1][1] -= lmd * L_n12 * x[1]
                self.w1[1][2] -= lmd * L_n13 * x[1]

                self.w1[2][0] -= lmd * L_n11 * x[2]
                self.w1[2][1] -= lmd * L_n12 * x[2]
                self.w1[2][2] -= lmd * L_n13 * x[2]

                self.b1[0] -= lmd * L_n11
                self.b1[1] -= lmd * L_n12
                self.b1[2] -= lmd * L_n13

                self.b2[0] -= lmd * L_n21
                self.b2[1] -= lmd * L_n22
                
                self.b3[0] -= lmd * L_n3
            if not epoch % 100:
                loss = (y_pred - y_true).mean() ** 2
                print(f"loss - {(loss):.8f}")

data = np.array([
    [2000, 1, 1],
    [2000, 1, 2],
    [2000, 1, 3],
    [2000, 1, 4],

    [23, 500, 1],
    [25, 560, 2],
    [21, 700, 3],
    [27, 400, 4]
])

all_y_trues = np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1
])

nn = NN()
nn.train(data, all_y_trues, epochs=7500)

print(f"{nn.predict(np.array([0, 250, 4]))[0]:.4f}")
print(f"{nn.predict(np.array([2000, 1, 1]))[0]:.4f}")
