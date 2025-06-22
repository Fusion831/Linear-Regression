# Linear Regression from Scratch 📈

## Overview

This project provides a Python implementation of Linear Regression from the ground up, utilizing `NumPy` for efficient numerical computations and `Matplotlib` for visualization. It explores both analytical (Normal Equation) and iterative (Gradient Descent) approaches to finding the optimal parameters for a linear model. The implementation also includes support for L2 Regularization (Ridge Regression) and various Gradient Descent optimization algorithms.

The primary goal is to solidify understanding of the fundamental mathematics and mechanics behind one of the most foundational machine learning algorithms.

## Features ✨

* **Core Linear Regression Model (`Linear_Regression` class):**
  * **Normal Equation:** Direct analytical solution for finding model parameters.
  * **Batch Gradient Descent (BGD):** Iterative optimization using the entire dataset for gradient computation.
  * **Stochastic Gradient Descent (SGD):** Iterative optimization using a single random sample per update.
  * **Mini-Batch Gradient Descent (MBGD):** Iterative optimization using a small batch of random samples per update.
* **L2 Regularization (Ridge Regression):**
  * Integrated into the Normal Equation.
  * Integrated into the cost function and gradient calculations for all Gradient Descent variants to prevent overfitting.
* **Mean Squared Error (MSE) Cost Function:**
  * Calculates the cost, optionally including the L2 penalty.
* **Prediction Capability:**
  * Make predictions on new data once the model is trained.
* **Synthetic Data Generation:**
  * A utility function (`generate_synthetic_data`) to create sample data for testing and demonstration.
* **Visualization Support:**
  * The `sample_run.py` script demonstrates how to:
    * Plot original data points.
    * Overlay regression lines learned by different methods.
    * Plot the cost history during Gradient Descent to visualize convergence.
        *(Note: Plots are displayed and can be saved by running `sample_run.py`)*

## Core Concepts & Reasoning 🧠

This implementation is built upon the following core mathematical and machine learning concepts:

1. **Hypothesis Function (Linear Model):**
    The model assumes a linear relationship between the input features (X) and the target variable (y):
    `h_θ(X) = Xθ`
    For a single feature `x`, this expands to: `y_predicted = θ₀ + θ₁x₁`
    where `θ₀` is the bias (intercept) and `θ₁` is the weight for the feature `x₁`. `X` is the design matrix with a leading column of ones for the bias term.

2. **Cost Function (Mean Squared Error - MSE):**
    To find the best parameters `θ`, we need to minimize a cost function. MSE measures the average squared difference between predicted values and actual values:
    `J(θ) = (1 / 2m) * Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²`
    where `m` is the number of training examples. The `1/2m` factor simplifies the derivative.

    * **With L2 Regularization (Ridge):**
        The cost function is modified to penalize large parameter values (except the bias term `θ₀`):
        `J(θ) = (1 / 2m) * Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)² + (λ / 2m) * Σ(θⱼ²)` (for j=1 to n)
        where `λ` is the regularization parameter.

3. **Normal Equation (Analytical Solution):**
    This method directly calculates the optimal `θ` that minimizes `J(θ)` without iteration:
    `θ = (XᵀX)⁻¹ Xᵀy`
    * **With L2 Regularization:**
        `θ = (XᵀX + λI')⁻¹ Xᵀy`
        where `I'` is an identity matrix with its first element (0,0) set to 0, ensuring the bias term is not regularized.

4. **Gradient Descent (Iterative Solution):**
    An iterative optimization algorithm that adjusts `θ` in the direction of the steepest descent of the cost function.
    * **Update Rule:** `θⱼ := θⱼ - α * (∂J(θ) / ∂θⱼ)`
    * **Gradient of MSE:** `∂J(θ) / ∂θⱼ = (1 / m) * Σ((h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾) * xⱼ⁽ⁱ⁾)`
        In vectorized form: `∇J(θ) = (1 / m) * Xᵀ(Xθ - y)`
    * **Gradient with L2 Regularization:**
        For `θ₀`: `(1 / m) * X₀ᵀ(Xθ - y)`
        For `θⱼ` (j > 0): `(1 / m) * Xⱼᵀ(Xθ - y) + (λ / m) * θⱼ`
    * **Variants:**
        * **Batch GD:** Uses all `m` samples to compute the gradient in each iteration.
        * **Stochastic GD:** Uses 1 random sample to compute the gradient. More noisy, but can escape local minima and be faster per iteration on large datasets.
        * **Mini-Batch GD:** Uses a small batch of `b` samples. A good compromise between BGD's stability and SGD's speed.

5. **Vectorization:**
    All computations heavily rely on `NumPy`'s vectorized operations (e.g., matrix multiplication `@`, transpose `.T`, `np.linalg.inv`) for efficiency, avoiding explicit Python loops over data samples or features where possible.

## Tech Stack 🛠️

* **Python 3.x**
* **NumPy:** For numerical operations and array manipulations.
* **Matplotlib:** For plotting data and regression lines.

## Getting Started 🚀

### Prerequisites

* Python 3.7+
* NumPy
* Matplotlib

You can install the dependencies using pip. It's recommended to use a virtual environment:

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install dependencies from requirements.txt
pip install -r requirements.txt

Running the Example

Clone the repository:

Generated bash
git clone http://github.com/Fusion831/Linear-Regression
cd Linear-Regression



Run the sample script:

python sample_run.py



This script will:

Generate synthetic data.

Train Linear Regression models using both the Normal Equation and Batch Gradient Descent.

Print the learned parameters (theta) from both methods.

Display plots showing the data, regression lines, and cost history (these plots can also be saved by the script).

