import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

def prob(num1, num2):
    mp = 'two.keras'
    x = np.linspace(-10,10, 100)
    z = np.linspace(-10,10,100)
    X,Z = np.meshgrid(x,z)
    X = X.flatten()
    Z = Z.flatten()
    Y = (X*X)/9 + (Z*Z)/4
    # 
    inp = np.column_stack((X,Z))
    if(os.path.exists(mp)):
        m = keras.models.load_model(mp)
    else:
        m = keras.Sequential([keras.layers.Dense(units=32, input_shape=(2,), activation='tanh'),
                              keras.layers.Dense(16, activation='tanh'),
                              keras.layers.Dense(units=1)])
        m.compile(optimizer = 'adam', loss='mean_squared_error')
        h = m.fit(inp, Y, epochs=100)
        pred = m.predict(inp)
        lv = h.history['loss']
        m.save('two.keras')
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
    # return m.predict(np.array([[((num1-xn)/(xx-xn)), ((num2-zn)/(zx-zn))]]))[0][0]
    return m.predict(np.array([[num1, num2]]))[0][0]

print(prob(3,2))
print(prob(-10,10))
print(prob(13,-9))