import mlx.core as mx
import mlx.core.random as random
import matplotlib.pyplot as plt
import numpy as np

# Problem metadata - with a twist
num_features = 2  # Reduced for visualization
num_examples = 100
num_iters = 500  # Reduced iterations for quicker demonstration
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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Scatter plot of data points
ax1.scatter(X[:, 0], y, color='blue', label='Data Points')

# Training with visualization and loss tracking
losses = []
for i in range(num_iters):
    grad_w, grad_b = grad_fn(w, b)
    w -= lr * grad_w
    b -= lr * grad_b

    # Collect loss for plotting
    current_loss = loss_fn(w, b).item()
    losses.append(current_loss)

    # Update model plot every 50 iterations
    if i % 50 == 0:
        line_x = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        line_y = line_x * w[0] + b
        ax1.plot(line_x, line_y, label=f'Iteration {i}', alpha=0.5)

# Final model plot
line_x = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
line_y = line_x * w[0] + b
ax1.plot(line_x, line_y, color='red', label='Final Model')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('y')
ax1.set_title('Linear Regression with MLX')
ax1.legend()

# Loss progression plot
ax2.plot(losses)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss')
ax2.set_title('Loss Progression')

plt.tight_layout()
plt.show()
