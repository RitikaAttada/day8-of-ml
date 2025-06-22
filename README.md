# 🌱 Day 8 of My Machine Learning Journey

Welcome to Day 8 of my Machine Learning journey! 🚀  
Today was all about challenging my models with complex nonlinear and modular functions to deepen my understanding of multivariable regression.

---

## 🧠 What I Did

✅ Implemented and tested regression models for:

* `Y = 2 / (1/X + 1/Z)` — Harmonic Mean  
* `Y = X² / 9 + Z² / 4` — Elliptical Paraboloid  
* `Y = min(X, Z)` — Ridge at Diagonal  
* `Y = tanh(X + Z)` — Smooth Saturating Function  
* `Y = max(X % 5, Z % 3)` — Modular Plateau Surface

✅ Visualized:

* Loss values over epochs
* Actual vs Predicted using scatter plots

---

## 🧪 Tools Used

* TensorFlow / Keras  
* NumPy  
* Matplotlib  

---

## 🔍 Key Learnings

* LeakyReLU prevents neuron death by allowing gradients to flow on negative inputs.
* Intermediate dense layer outputs can be negative even with all-positive inputs.
* Modulus-based functions are difficult for standard regression due to discontinuities.
* Regression models tend to smooth out sharp edges, causing loss of accuracy on step functions.
* Not all numerical functions are best solved with regression — some need classification.
