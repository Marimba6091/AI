import numpy as np


class MSE:
    def __init__(self, y_pred, y_true):
        self.__y_pred = y_pred
        self.__y_true = y_true
        self.__mse = self.__score()
    
    def __score(self):
        mse = (self.__y_pred - self.__y_true).mean() ** 2
        return mse
    
    def __float__(self):
        return float(self.__mse.item())
    
    def __str__(self):
        return f"{self.__mse:.4f}"
    
    def __format__(self, format_spec):
        return format(self.__mse, format_spec)