import numpy as np
import matplotlib.pyplot as plt
from lab_utils_common import compute_cost
plt.style.use('./deeplearning.mplstyle')

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

def compute_gradient(x, y, w, b):
    """Calculate gradients for w and b"""
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    return dj_dw, dj_db

def gradient_descent(x, y, w_init, b_init, learning_rate=0.01, num_iters=1000):
    """Automatically find optimal w and b"""
    w = w_init
    b = b_init
    
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        
        # Update w and b
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db
        
        if i % 100 == 0:
            cost = compute_cost(x, y, w, b)
            print(f"Iteration {i}: Cost = {cost:.4f}, w = {w:.4f}, b = {b:.4f}")
    
    return w, b

# Start with initial values
w_init = 0
b_init = 0

# Run gradient descent to find optimal w and b automatically
print("Running Gradient Descent...\n")
w_optimal, b_optimal = gradient_descent(x_train, y_train, w_init, b_init, learning_rate=0.01, num_iters=1000)

print(f"\nOptimal values found:")
print(f"w = {w_optimal:.4f}, b = {b_optimal:.4f}")
print(f"Final cost = {compute_cost(x_train, y_train, w_optimal, b_optimal):.6f}\n")

# Visualization with optimal values
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Data and regression line with optimal parameters
ax = axes[0]
ax.scatter(x_train, y_train, marker='x', c='red', s=100, label='Actual data')
x_line = np.array([0, 3])
y_line = w_optimal * x_line + b_optimal
ax.plot(x_line, y_line, 'b-', linewidth=2, label=f'Best fit (w={w_optimal:.2f}, b={b_optimal:.2f})')
ax.set_xlabel('House size (1000 sqft)')
ax.set_ylabel('House price ($1000)')
ax.set_title('Linear Regression - Optimized')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Cost function
ax2 = axes[1]
w_range = np.linspace(-50, 400, 100)
costs = [compute_cost(x_train, y_train, w_test, b_optimal) for w_test in w_range]
ax2.plot(w_range, costs, 'g-', linewidth=2)
ax2.scatter([w_optimal], [compute_cost(x_train, y_train, w_optimal, b_optimal)], 
           c='red', s=100, label=f'Optimal w={w_optimal:.2f}')
ax2.set_xlabel('Weight (w)')
ax2.set_ylabel('Cost')
ax2.set_title('Cost Function')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()