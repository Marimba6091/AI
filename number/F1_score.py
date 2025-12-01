import numpy as np

class F1_score:
    def __init__(self, y_pred, y_true):
        self.__score = self.__f1_score(y_pred, y_true).item()
    
    def __float__(self):
        return self.__score
    
    def __str__(self):
        return str(self.__score)
    
    def __format__(self, format_spec):
        return format(self.__score, format_spec)
        
    def __precision(self, y_pred, y_true):
        y_pred_n = (y_pred >= .5).astype(int)
        tps = np.sum((y_pred_n == 1) & (y_true == 1))
        fps = np.sum((y_pred_n == 1) & (y_true == 0))
        result = tps / (tps + fps)
        return result

    def __recall(self, y_pred, y_true):
        y_pred_n = (y_pred >= .5).astype(int)
        tps = np.sum((y_pred_n == 1) & (y_true == 1))
        fns = np.sum((y_pred_n == 0) & (y_true == 1))
        result = tps / (tps + fns)
        return result

    def __f1_score(self, y_pred, y_true):
        rec = self.__recall(y_pred, y_true)
        prec = self.__precision(y_pred, y_true)
        return 2 * (prec * rec) / (prec + rec)
    
