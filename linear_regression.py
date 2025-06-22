import numpy as np
import matplotlib.pyplot as plt



def generate_synthetic_data(n_samples=100, true_theta0=4, true_theta1=3, noise_std=1):
    """
    Generates simple synthetic data for linear regression.
    y = theta0 + theta1*X + noise

    Args:
        n_samples (int): Number of data points.
        true_theta0 (float): True intercept (bias).
        true_theta1 (float): True slope (weight for the single feature).
        noise_std (float): Standard deviation of Gaussian noise to add.

    Returns:
        tuple: (X_b, y, X_feature)
            X_b (np.ndarray): Design matrix with an intercept column (x0=1).
                              Shape: (n_samples, 2) for a single feature.
            y (np.ndarray): Target variable vector. Shape: (n_samples, 1).
            X_feature (np.ndarray): Original feature X (without intercept).
                                    Shape: (n_samples, 1). Useful for analysis or plotting elsewhere.
    """
    # Generate feature values
    X_feature = 2 * np.random.rand(n_samples, 1)

    # Generate target values with noise
    # y = true_theta0 (bias) + true_theta1 * X_feature (weight * feature) + noise
    noise = noise_std * np.random.randn(n_samples, 1)
    y = true_theta0 + true_theta1 * X_feature + noise

    # Add intercept term (x0=1) to the feature matrix to create the design matrix X_b
    #theta[0] represents the bias term
    X_b = np.c_[np.ones((n_samples, 1)), X_feature]

    return X_b, y, X_feature


class Linear_Regression:
    """
    Linear Regression model implemented from scratch.
    Supports
    -Normal Equation
    -Batch Gradient Descent
    -Mini Batch Gradient Descent
    -Stochastic Gradient Descent
    """
    
    def __init__(self):
        """
        Initializing the Linear Regression Model
        -theta -> vector of parameters
        -cost_hist -> storing cost history for each iteration, used for various gradient descent algorithms
        """
        self.theta=None 
        self.cost_hist=[]
        
    
    def fit_normal_equation(self, X_b, y):
        """
        Fits the model using the Normal Equation.
        theta = (X_b.T @ X_b)^(-1) @ X_b.T @ y
        Stores the result in self.theta and the final cost in self.cost_history.

        Args:
            X_b (np.ndarray): Design matrix.
            y (np.ndarray): Target values.
        """
        try:
            self.theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
            self.cost_history = [self.calculate_mse_cost(X_b, y, self.theta)]
        except np.linalg.LinAlgError:
            self.theta = None
            self.cost_history = []
            print("Error: Normal Equation failed. The matrix (X_b.T @ X_b) is singular and not invertible.")
    
    def calculate_mse_cost(self,X_b,y,theta):
        """
        Calculates the Mean Squared Error (MSE) cost.
         Args:
            X_b (np.ndarray): Design matrix (features + intercept column, x0=1).
            y (np.ndarray): Target values.
            theta (np.ndarray): Model parameters [bias, weight1, ...].
         Returns:
            float: The mean squared error.
        """
        m=len(y)
        #m -> number of training examples 
        if m == 0: return 0
        predictions= X_b@theta
        cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
        return cost
    
    
    def _gradient(self, X_b, y, theta):
        """
        Calculates the gradient of the MSE cost function with respect to theta.
        dJ/d(theta_j) = (1/m) * sum((h_theta(x_i) - y_i) * x_ij)

        Args:
            X_b (np.ndarray): Design matrix (features + intercept column, x0=1).
            y (np.ndarray): Target values.
            theta (np.ndarray): Model parameters [bias, weight1, ...].

        Returns:
            np.ndarray: The gradient vector, which will be same shape as theta.
        """
        m = len(y)
        
        if m == 0:
            return np.zeros_like(theta) # Return zero gradient if no samples
        predictions = X_b @ theta
        errors = predictions - y
        gradient = (1 / m) * (X_b.T @ errors)
        return gradient
    
    
    def batch_gradient_descent(self, X_b, y, theta_initial, learning_rate, n_iterations):
        """
        Performs Batch Gradient Descent (BGD).
        Updates theta using the gradient computed on the entire training set.

        Args:
            X_b (np.ndarray): Design matrix.
            y (np.ndarray): Target values.
            theta_initial (np.ndarray): Initial parameters [bias, weight1, ...].
            learning_rate (float): Learning rate (alpha).
            n_iterations (int): Number of iterations.

        Returns:
            np.ndarray: Optimized parameters (theta).
        """
        theta = np.copy(theta_initial)
        self.cost_history = [] 

        for iteration in range(n_iterations):
            gradient = self._gradient(X_b, y, theta) #Computing Gradient over entire training set
            theta = theta - learning_rate * gradient
            cost = self.calculate_mse_cost(X_b, y, theta)
            self.cost_history.append(cost)
        return theta
    
    
    def stochastic_gradient_descent(self, X_b, y, theta_initial, learning_rate, n_epochs, m_samples):
        """
        Performs Stochastic Gradient Descent (SGD).
        Updates theta using the gradient computed on a single, randomly chosen training example at each step.

        Args:
            X_b (np.ndarray): Design matrix.
            y (np.ndarray): Target values.
            theta_initial (np.ndarray): Initial parameters.
            learning_rate (float): Learning rate.
            n_epochs (int): Number of passes over the entire dataset.
            m_samples (int): Total number of training samples.

        Returns:
            np.ndarray: Optimized parameters (theta).
        """
        theta = np.copy(theta_initial)
        self.cost_history = [] 

        for epoch in range(n_epochs):
            # Shuffle data at the beginning of each epoch/iteration
            indices = np.random.permutation(m_samples)
            X_b_shuffled = X_b[indices]
            y_shuffled = y[indices]

            for i in range(m_samples):
                xi = X_b_shuffled[i:i+1] #As 2D array
                yi = y_shuffled[i:i+1]   #As 2D array
                gradient = self._gradient(xi, yi, theta) # Gradient for one sample
                theta = theta - learning_rate * gradient

            # Calculate and store cost at the end of each epoch/iteration (on full training set)
            cost_epoch = self.calculate_mse_cost(X_b, y, theta)
            self.cost_history.append(cost_epoch)
        return theta
    
    
    def mini_batch_gradient_descent(self, X_b, y, theta_initial, learning_rate, n_iterations, batch_size, m_samples):
        """
        Performs Mini-Batch Gradient Descent.
        Updates theta using the gradient computed on a small batch of training examples.

        Args:
            X_b (np.ndarray): Design matrix.
            y (np.ndarray): Target values.
            theta_initial (np.ndarray): Initial parameters.
            learning_rate (float): Learning rate.
            n_iterations (int): Number of batch updates (iterations).
            batch_size (int): Size of each mini-batch.
            m_samples (int): Total number of training samples.

        Returns:
            np.ndarray: Optimized parameters (theta).
        """
        theta = np.copy(theta_initial)
        self.cost_history = [] # Reset cost history
        # n_batches_per_epoch = int(np.ceil(m_samples / batch_size)) # If using epochs

        for iteration in range(n_iterations):
            # Select a random mini-batch
            random_indices = np.random.choice(m_samples, batch_size, replace=False)
            X_batch = X_b[random_indices]
            y_batch = y[random_indices]

            gradient = self._gradient(X_batch, y_batch, theta) # Gradient for the mini-batch
            theta = theta - learning_rate * gradient

            # Calculate and store cost at each iteration (on full dataset for consistent tracking)
            cost_iteration = self.calculate_mse_cost(X_b, y, theta)
            self.cost_history.append(cost_iteration)
        return theta

    def fit(self, X_b, y, solver="batch_gd", learning_rate=0.01, n_iterations=1000,
            n_epochs=50, batch_size=32, theta_initial=None):
        """
        Fits the linear regression model using the specified solver.

        Args:
            X_b (np.ndarray): Design matrix (features + intercept column, x0=1).
            y (np.ndarray): Target values.
            solver (str): The optimization algorithm.
                          Options: "normal_equation", "batch_gd", "sgd", "mini_batch_gd".
            learning_rate (float): Learning rate for Gradient Descent variants.
            n_iterations (int): Number of iterations (for BGD, Mini-Batch GD if not using epochs).
            n_epochs (int): Number of epochs (for SGD, or can be used for Mini-Batch GD).
            batch_size (int): Batch size for Mini-Batch GD.
            theta_initial (np.ndarray, optional): Initial guess for parameters [bias, weight1,...].
                                                  If None, initialized to zeros.

        Raises:
            ValueError: If an unsupported solver is specified.
        """
        if theta_initial is None:
            # Initialize theta with zeros. X_b.shape[1] is num_features + 1 (for intercept).
            self.theta = np.zeros((X_b.shape[1], 1))
        else:
            self.theta = np.copy(theta_initial) # Ensure it's a copy

        m_samples = len(y) # Total number of training examples

        if solver == "normal_equation":
            self.fit_normal_equation(X_b, y) # This method sets self.theta and self.cost_history
        elif solver == "batch_gd":
            self.theta = self.batch_gradient_descent(X_b, y, self.theta, learning_rate, n_iterations)
        elif solver == "sgd":
            self.theta = self.stochastic_gradient_descent(X_b, y, self.theta, learning_rate, n_epochs, m_samples)
        elif solver == "mini_batch_gd":
            
            self.theta = self.mini_batch_gradient_descent(X_b, y, self.theta, learning_rate, n_iterations, batch_size, m_samples)
        else:
            raise ValueError(f"Unknown solver: '{solver}'. "
                             "Supported: 'normal_equation', 'batch_gd', 'sgd', 'mini_batch_gd'.")

    
    
    def predict(self, X_b):
        """
        Makes predictions using the learned model parameters (self.theta).
        y_predicted = X_b @ theta

        Args:
            X_b (np.ndarray): Design matrix for which to make predictions.
                               Shape: (n_samples_to_predict, n_features + 1)

        Returns:
            np.ndarray: Predicted values. Shape: (n_samples_to_predict, 1)

        Raises:
            ValueError: If the model has not been fitted yet (self.theta is None).
        """
        if self.theta is None:
            raise ValueError("Model has not been fitted yet. Call a 'fit' method first.")
        return X_b @ self.theta
    