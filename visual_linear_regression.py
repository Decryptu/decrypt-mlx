import mlx.core as mx
import mlx.core.random as random
import matplotlib.pyplot as plt
import numpy as np

# Problem metadata - with a twist
num_features = 2  # Reduced for visualization
num_examples = 100
num_iters = 1000  # Reduced iterations for quicker demonstration
lr = 0.1  # Increased learning rate

# True parameters - now with a bias term
w_star = random.normal((num_features,))
b_star = random.normal(())

# Input examples (design matrix) - 2D for easy visualization
X = random.normal((num_examples, num_features))

# Noisy labels - adding bias term
eps = 0.1 * random.normal((num_examples,))
y = X @ w_star + b_star + eps

# Loss function
def loss_fn(w, b):
    return 0.5 * mx.mean(mx.square(X @ w + b - y))

# Gradient function
grad_fn = mx.grad(loss_fn, argnums=[0, 1])

# Optimization - including bias
w = random.normal((num_features,))
b = random.normal(())

# Visualization setup
fig, ax = plt.subplots()
ax.scatter(X[:, 0], y, color='blue', label='Data Points')  # Corrected line

# Training with visualization
for i in range(num_iters):
    grad_w, grad_b = grad_fn(w, b)
    w -= lr * grad_w
    b -= lr * grad_b

    # Update plot every 100 iterations
    if i % 100 == 0:
        line_x = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        line_y = line_x * w[0] + b
        ax.plot(line_x, line_y, label=f'Iteration {i}', alpha=0.5)

# Final model plot
line_x = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
line_y = line_x * w[0] + b
ax.plot(line_x, line_y, color='red', label='Final Model')
ax.legend()
plt.xlabel('Feature 1')
plt.ylabel('y')
plt.title('Linear Regression with MLX')
plt.show()
