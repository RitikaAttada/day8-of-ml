import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras

def prob(num1, num2):
    mp='five.keras'
    x = np.linspace(0,50,200)
    z = np.linspace(0,50,200)
    X,Z = np.meshgrid(x,z)
    X = X.flatten()
    Z = Z.flatten()
    Y = np.maximum(X%5, Z%3)
    xn = X.min()
    xx = X.max()
    zn = Z.min()
    zx = Z.max()
    yn = Y.min()
    yx = Y.max()
    X = (X-xn)/(xx-xn)
    Z = (Z-zn)/(zx-zn)
    Y = (Y-yn)/(yx-yn)
    inp = np.column_stack((X,Z))
    if (os.path.exists(mp)):
        m = keras.models.load_model(mp)
    else:
        m = keras.Sequential([keras.layers.Dense(128, input_shape=(2,)),
                              keras.layers.LeakyReLU(alpha = 0.1),
                              keras.layers.Dense(64),
                              keras.layers.LeakyReLU(alpha=0.1),
                              keras.layers.Dense(32),
                              keras.layers.LeakyReLU(alpha=0.1),
                              keras.layers.Dense(units=1)])
        m.compile(optimizer = 'adam', loss='mean_squared_error')
        h = m.fit(inp, Y, epochs=200)
        lv = h.history['loss']
        pred = m.predict(inp)
        m.save('five.keras')
        plt.figure()
        plt.plot(lv)
        plt.title('loss values')
        plt.show()
        plt.figure()
        plt.scatter(Y, pred, label='actual vs preicted', color='pink')
        plt.grid(True)
        plt.xlabel('actual')
        plt.ylabel('predicted')
        plt.legend()
        plt.show() 
    return m.predict(np.array([[((num1-xn)/(xx-xn)), ((num2-zn)/(zx-zn))]]))[0][0]*(yx-yn)+yn

print(prob(7,4))
print(prob(16,30))
print(prob(5,3))
