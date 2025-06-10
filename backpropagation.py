import math
import numpy as np
def relu(x):
    return max(0, x)
def relu_derivative(x):
    return 1 if x > 0 else 0
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
def loss(x):
    return 0.5*(Y-x)**2
def loss_derivative(x):
    return (x-Y)

vectorized_relu = np.vectorize(relu)
vectorized_relu_derivative = np.vectorize(relu_derivative)
vectorized_sigmoid = np.vectorize(sigmoid)
vectorized_sigmoid_derivative = np.vectorize(sigmoid_derivative)

W = [np.array([[0.1, 0.3],
      [0.2, 0.4]]),
      np.array([[0.5, 0.6]])]
b = [np.array([0.1, 0.1]), np.array([0.2])]
activations = [vectorized_relu, vectorized_sigmoid]
activation_derivatives = [vectorized_relu_derivative, vectorized_sigmoid_derivative]

X = np.array([1.0, 0.5])

Y = 0.9

Nu = 0.1

runs = 1000

# Forward pass (fill in)
def forward_pass (X, W, activations):
    pre_activations = []
    activations_out = [X]
    for layer in np.arange(0, len(W)):
        z = np.array(W[layer].dot(activations_out[-1]) + b[layer])
        pre_activations.append(z)
        a = activations[layer](z)
        activations_out.append(a)
    return pre_activations, activations_out
# Backward pass (fill in)
def backward_pass (pre_activations, activations_out):
    grads_W = []
    grads_b = []
    i = len(W) - 1
    delta = np.atleast_1d(loss_derivative(activations_out[-1])*activation_derivatives[-1](pre_activations[-1]))
    while i >= 0:
        grads_W.append(np.outer(activations_out[i], delta).T)
        grads_b.append(delta)
        if i != 0:
            delta = np.dot(W[i].T, delta)*activation_derivatives[i-1](pre_activations[i-1])
        i -= 1
    grads_W.reverse()
    grads_b.reverse()
    return grads_W, grads_b

# Update parameters (fill in)
def update_parameters(pre_activations, activations_out):
    grads_W, grads_b = backward_pass(pre_activations, activations_out)
    for i in np.arange(0, len(W)):
        print(f'Before update W[{i}]:\n{W[i]}')
        print(f'Before update b[{i}]:\n{b[i]}')
        W[i] -=  Nu*grads_W[i]
        b[i] -=  Nu*grads_b[i]
        print(f'After update W[{i}]:\n{W[i]}')
        print(f'After update b[{i}]:\n{b[i]}')
    

def main():
    for i in range(0, runs) :
        pre_activations, activations_out = forward_pass(X, W, activations)
        print(f'Loss: {loss(activations_out[-1][0])}')
        update_parameters(pre_activations, activations_out)

if __name__ == "__main__":
    main()
