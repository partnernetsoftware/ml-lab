import numpy as np
from scipy.optimize import minimize
from numba import njit
##############################

nn = 9
T = np.random.rand(3*nn+1)
c_sample = 128
x = np.linspace(start=-10,stop=10,num=c_sample).reshape(-1,1) # vectorized
y_real = np.sin(x)/x
y_sample = np.sin(x)/x + (np.random.rand(c_sample,1)*0.1-0.05)

X = x
Y = y_sample
H = np.array([nn])

##############################

PROD = np.dot
EXP = np.exp
MEAN = np.mean

@njit
def A(x): return 1/(1+EXP(-x))
MSE = lambda Y,P: MEAN((Y-P)**2)

@njit
def UPK(T,H):
    NN = H[0]
    W1= T[:NN].reshape(1, NN)
    W2= T[NN:NN*2].reshape(NN, 1)
    B1= T[NN*2:-1].reshape(-1, NN)
    B2= T[-1]
    return W1,W2,B1,B2

@njit
def NNOP(T,X,H):
    W1,W2,B1,B2 = UPK(T,H)
    return PROD(A(PROD(X,W1)-B1),W2)-B2

LOSS_NN = lambda T,X,H,Y: MSE(NNOP(T,X,H),Y)

SOLVE = lambda T,X,Y,H:minimize(LOSS_NN,T,(X,H,Y)).x

RS = SOLVE(T,X,Y,H)

##############################

print('MODEL TO SAVE:\n',UPK(RS,H))

import matplotlib.pyplot as plt
x_test = np.sort(np.random.rand(100,1)*10-5,axis=0)
y_test_real = np.sin(x_test)/x_test
y_test_pred = NNOP(RS, x_test, H)
plt.plot(x_test, y_test_real, 'oy')
plt.plot(x_test, y_test_pred, '-b')
plt.legend(['y_test_real', 'y_test_pred'])
#plt.title('mse={0}'.format(MSE(y_test_real,y_test_pred)))
plt.show()

