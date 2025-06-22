import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

def prob(num1, num2):
    mp = 'four.keras'
    x= np.linspace(-5,5,100)
    z = np.linspace(-5,5,100)
    X,Z = np.meshgrid(x,z)
    X = X.flatten()
    Z = Z.flatten()
    Y = np.tanh(X+Z)
    X = X/10
    Z = Z/10
    inp = np.column_stack((X,Z))
    if (os.path.exists(mp)):
        m = keras.models.load_model(mp)
    else:
        m = keras.Sequential([keras.layers.Dense(32, input_shape=(2,), activation='tanh'),
                              keras.layers.Dense(16, activation='tanh'),
                              keras.layers.Dense(units=1, activation='tanh')])
        m.compile(optimizer='adam', loss='mean_squared_error')
        h = m.fit(inp, Y, epochs=100)
        lv = h.history['loss']
        p = m.predict(inp)
        m.save('four.keras')
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
    return m.predict(np.array([[num1/10, num2/10]]))[0][0]

print(prob(3,-2))
print(prob(-5,4))
print(prob(-6, 2))

