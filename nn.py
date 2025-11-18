import numpy as np
import json


def sigmoid(x):
    # Функция активации sigmoid:: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # Производная от sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    # y_true и y_pred являются массивами numpy с одинаковой длиной
    return ((y_true - y_pred) ** 2).mean()


class OurNeuralNetwork:
    """
    Нейронная сеть, у которой:
        - 2 входа
        - скрытый слой с двумя нейронами (h1, h2)
        - слой вывода с одним нейроном (o1)
    """

    def __init__(self):
        # Вес
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Смещения
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues, learn_rate, epochs):

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):

                # --- Выполняем обратную связь (нам понадобятся эти значения в дальнейшем)
                neuro_1_sum = self.w1 * x[0] + self.w2 * x[1] + self.b1
                neuro_1_result = sigmoid(neuro_1_sum)

                neuro_2_sum = self.w3 * x[0] + self.w4 * x[1] + self.b2
                neuro_2_result = sigmoid(neuro_2_sum)

                neuro_3_sum = self.w5 * neuro_1_result + self.w6 * neuro_2_result + self.b3
                self.sum = neuro_2_sum
                neuro_3_result = sigmoid(neuro_3_sum)
                self.res = neuro_3_result

                y_pred = neuro_3_result

                d_L_d_ypred = -(y_true - y_pred)

                # Нейрон o1
                d_ypred_d_w5 = neuro_1_result * deriv_sigmoid(neuro_3_sum) # изменение для веса 1 нейрона о1
                d_ypred_d_w6 = neuro_2_result * deriv_sigmoid(neuro_3_sum) # изменение для веса 2 нейрона о1
                d_ypred_d_b3 = deriv_sigmoid(neuro_3_sum)      # изменение для смещения нейрона о1

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(neuro_3_sum)  # опускаемся в глубь к нейрону 1
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(neuro_3_sum) # опускаемся в глубь к нейрону 2

                # Нейрон h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(neuro_1_sum) # изменение для веса 2 нейрона h1
                d_h1_d_w2 = x[1] * deriv_sigmoid(neuro_1_sum) # изменение для веса 2 нейрона h1
                d_h1_d_b1 = deriv_sigmoid(neuro_1_sum) # изменение для смещения нейрона h1

                # Нейрон h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(neuro_2_sum)
                d_h2_d_w4 = x[1] * deriv_sigmoid(neuro_2_sum)
                d_h2_d_b2 = deriv_sigmoid(neuro_2_sum)

                # --- Обновляем вес и смещения
                # Нейрон h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Нейрон h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Нейрон o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3
            if not epoch % 100:
                loss = (y_pred - y_true).mean() ** 2
                print(loss)


# Определение набора данных
# Смещение данных вес -60 рост -170
data = np.array([
    [-10, -10],  # Маша
    [0, 19],  # Петя
    [130, 5],  # Вася
    [-1, -1],  # Оля
])

all_y_trues = np.array([
    0,
    1,
    1,
    0,
])

# Тренируем нашу нейронную сеть!
network = OurNeuralNetwork()
network.train(data, all_y_trues, 0.1, 2000)

# Делаем предсказания
emily = np.array([1, 7])  #

print(f"Emily:  {network.feedforward(emily):.3f}")
# print(deriv_sigmoid(network.sum))
# print(network.res)