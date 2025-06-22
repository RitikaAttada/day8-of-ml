import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow import keras

def prob(num1, num2):
    mp = 'three.keras'
    x = np.linspace(0,100, 100)
    z = np.linspace(0,100,100)
    X,Z = np.meshgrid(x,z)
    X = X.flatten()
    Z = Z.flatten()
    Y = np.minimum(X,Z)
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
        m = keras.Sequential([keras.layers.Dense(32, input_shape=(2,), activation='relu'),
                              keras.layers.Dense(16,activation='relu'),
                              keras.layers.Dense(units=1)])
        m.compile(optimizer = 'adam', loss='mean_squared_error')
        h = m.fit(inp, Y, epochs=100)
        p = m.predict(inp)
        lv = h.history['loss']
        m.save('three.keras')
        plt.figure()
        plt.plot(lv)
        plt.title('loss values')
        plt.show()
        plt.figure()
        plt.scatter(Y, p, label='actual vs preicted', color='pink')
        plt.grid(True)
        plt.xlabel('actual')
        plt.ylabel('predicted')
        plt.legend()
        plt.show()  
    return m.predict(np.array([[((num1-xn)/(xx-xn)), ((num2-zn)/(zx-zn))]]))[0][0]*(yx-yn)+yn

print(prob(10,1))
print(prob(56,78))
print(prob(200,250))
