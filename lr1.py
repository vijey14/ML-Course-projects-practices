import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0, 3.0])
y_train = np.array([300.0, 500.0, 700.0])

def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

w = 200
b = 100
tmp = compute_model_output(x_train, w, b)


plt.plot(x_train, tmp, c='b', label='Predicted Line')
plt.plot(x_train, y_train, marker = "x", color = "red", label='Actual Data')

plt.title('Housing Prices')
plt.xlabel('Size (1000 sqft)')
plt.ylabel('Price (1000s of dollars)')
plt.legend()
plt.show()

x_i = 1.2
cost_1200sqft = w * x_i + b
print(f'Predicted price for a 1200 sqft house: {cost_1200sqft:.2f} thousand dollars')