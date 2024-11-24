import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr
        self.activation_fn = activation
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        self.activations = {}
        self.gradients = {}

    def forward(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        if self.activation_fn == 'tanh':
            A1 = np.tanh(Z1)
        elif self.activation_fn == 'relu':
            A1 = np.maximum(0, Z1)
        elif self.activation_fn == 'sigmoid':
            A1 = 1 / (1 + np.exp(-Z1))
        else:
            raise ValueError("Unsupported activation function")
        
        self.activations['Z1'] = Z1
        self.activations['A1'] = A1

        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = Z2  # Linear activation for the output
        self.activations['Z2'] = Z2
        self.activations['A2'] = A2
        return A2

    def backward(self, X, y):
        A2 = self.activations['A2']
        A1 = self.activations['A1']
        Z1 = self.activations['Z1']
        m = y.shape[0]

        # Compute gradients for output layer
        delta2 = A2 - y
        dW2 = np.dot(A1.T, delta2) / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m

        # Compute gradients for hidden layer
        if self.activation_fn == 'tanh':
            dZ1 = 1 - np.tanh(Z1) ** 2
        elif self.activation_fn == 'relu':
            dZ1 = (Z1 > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-Z1))
            dZ1 = sigmoid * (1 - sigmoid)
        else:
            raise ValueError("Unsupported activation function")

        delta1 = np.dot(delta2, self.W2.T) * dZ1
        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0, keepdims=True) / m

        # Update weights
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        # Store gradients for visualization
        self.gradients = {
            'dW2': dW2,
            'db2': db2,
            'dW1': dW1,
            'db1': db1,
        }

def generate_data(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1
    y = y.reshape(-1, 1)
    return X, y

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    # Hidden layer features
    hidden_features = mlp.activations['A1']
    ax_hidden.scatter(
        hidden_features[:, 0],
        hidden_features[:, 1],
        c=y.ravel(),
        cmap='bwr',
        alpha=0.7
    )
    ax_hidden.set_title("Hidden Layer Features")
    ax_hidden.view_init(elev=30, azim=45)

    # Add decision hyperplane in the hidden space
    W2 = mlp.W2
    b2 = mlp.b2
    if W2.shape[0] == 3:  # Ensure we have a 3D hidden space
        x_range = np.linspace(-1, 1, 10)
        y_range = np.linspace(-1, 1, 10)
        xx, yy = np.meshgrid(x_range, y_range)
        zz = -(W2[0, 0] * xx + W2[1, 0] * yy + b2[0, 0]) / W2[2, 0]
        ax_hidden.plot_surface(xx, yy, zz, alpha=0.3, color='gray')

    # Input layer decision boundary
    x_min, x_max = -3, 3
    y_min, y_max = -3, 3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = mlp.forward(grid)
    zz = predictions.reshape(xx.shape)

    ax_input.contourf(xx, yy, zz, levels=20, cmap='bwr', alpha=0.6)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title("Input Space Decision Boundary")

    # Gradient visualization
    ax_gradient.set_title("Gradient Visualization")
    ax_gradient.axis('off')

    input_layer_size = mlp.W1.shape[0]
    hidden_layer_size = mlp.W1.shape[1]
    output_layer_size = mlp.W2.shape[1]

    # Node positions
    nodes_input = np.array([[x, 0.8] for x in np.linspace(0.2, 0.8, input_layer_size)])
    nodes_hidden = np.array([[x, 0.5] for x in np.linspace(0.2, 0.8, hidden_layer_size)])
    nodes_output = np.array([[0.5, 0.2]])

    # Draw input nodes
    for idx, node in enumerate(nodes_input):
        ax_gradient.add_patch(Circle(node, radius=0.03, color='blue'))
        ax_gradient.text(
            node[0], node[1] + 0.05,
            f"Input {idx+1}",
            ha='center',
            va='bottom',
            fontsize=10
        )

    # Draw hidden nodes
    for idx, node in enumerate(nodes_hidden):
        ax_gradient.add_patch(Circle(node, radius=0.03, color='green'))
        ax_gradient.text(
            node[0], node[1] + 0.05,
            f"Hidden {idx+1}",
            ha='center',
            va='bottom',
            fontsize=10
        )

    # Draw output nodes
    for idx, node in enumerate(nodes_output):
        ax_gradient.add_patch(Circle(node, radius=0.03, color='red'))
        ax_gradient.text(
            node[0], node[1] - 0.05,
            f"Output {idx+1}",
            ha='center',
            va='top',
            fontsize=10
        )

    # Draw connections from input to hidden
    for i in range(input_layer_size):
        for j in range(hidden_layer_size):
            gradient_magnitude = np.abs(mlp.gradients['dW1'][i, j])
            ax_gradient.plot(
                [nodes_input[i, 0], nodes_hidden[j, 0]],
                [nodes_input[i, 1], nodes_hidden[j, 1]],
                'k-',
                linewidth=gradient_magnitude * 10
            )

    # Draw connections from hidden to output
    for i in range(hidden_layer_size):
        for j in range(output_layer_size):
            gradient_magnitude = np.abs(mlp.gradients['dW2'][i, j])
            ax_gradient.plot(
                [nodes_hidden[i, 0], nodes_output[j, 0]],
                [nodes_hidden[i, 1], nodes_output[j, 1]],
                'k-',
                linewidth=gradient_magnitude * 10
            )


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num // 10, repeat=False)
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
