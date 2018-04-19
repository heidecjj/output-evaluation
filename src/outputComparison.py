from numpy import power as _power
from numpy import square as _square
from numpy import sqrt as _sqrt
from pandas import DataFrame as _DataFrame

def Euclid(p1, p2):
    return Minkowski(p1, p2, 2)

def Minkowski(p1, p2, q):
    return _power(_power(abs(p1 - p2),q).sum(), 1/q)

def Supremum(p1, p2):
    return abs(p1 - p2).max()

def MSE(p1, p2):
    return _square(p1 - p2).mean()

def RMSE(p1, p2):
    return _sqrt(MSE(p1, p2))

def NRMSE(p1, p2):
    return RMSE(p1, p2) / (max(p1) - min(p1))

def EvaluateSets(p1, p2, cols=None):
    if cols is None:
        cols = p1.columns
    
    df = _DataFrame(index=cols,
                    columns=['Euclid', 'Supremum', 'MSE', 'RMSE', 'NRMSE'])

    for col in cols:
        df['Euclid'][col] = Euclid(p1[col], p2[col])
        df['Supremum'][col] = Supremum(p1[col], p2[col])
        df['MSE'][col] = MSE(p1[col], p2[col])
        df['RMSE'][col] = RMSE(p1[col], p2[col])
        df['NRMSE'][col] = NRMSE(p1[col], p2[col])

    df
    return df