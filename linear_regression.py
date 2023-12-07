import mlx.core as mx

# Problem metadata
num_features = 100
num_examples = 1_000
num_iters = 10_000  # iterations of SGD
lr = 0.01  # learning rate for SGD

# True parameters
w_star = mx.random.normal((num_features,))

# Input examples (design matrix)
X = mx.random.normal((num_examples, num_features))

# Noisy labels
eps = 1e-2 * mx.random.normal((num_examples,))
y = X @ w_star + eps

# Loss function
def loss_fn(w):
    return 0.5 * mx.mean(mx.square(X @ w - y))

# Gradient function
grad_fn = mx.grad(loss_fn)

# Optimization
w = 1e-2 * mx.random.normal((num_features,))

for i in range(num_iters):
    grad = grad_fn(w)
    w -= lr * grad
    mx.eval(w)

    # Print log every 1000 iterations
    if i % 1000 == 0:
        current_loss = loss_fn(w).item()
        current_error_norm = mx.sum(mx.square(w - w_star)).item() ** 0.5
        print(f"Iteration {i}: Loss {current_loss:.5f}, |w-w*| = {current_error_norm:.5f}")

# Compute final loss and error norm
final_loss = loss_fn(w)
final_error_norm = mx.sum(mx.square(w - w_star)).item() ** 0.5
print(f"Final Loss {final_loss.item():.5f}, |w-w*| = {final_error_norm:.5f}")
