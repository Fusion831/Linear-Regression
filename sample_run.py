import numpy as np
import matplotlib.pyplot as plt

try:
    from linear_regression import Linear_Regression, generate_synthetic_data
except ImportError:
    print("Error: 'linear_regression_model.py' not found or incomplete.")
    exit()

def run_mvp_showcase():
    N_SAMPLES = 100
    TRUE_THETA0 = 4
    TRUE_THETA1 = 3
    NOISE_LEVEL = 1
    LEARNING_RATE = 0.1
    N_ITERATIONS = 1000

    X_b_train, y_train, X_feature_train = generate_synthetic_data(
        n_samples=N_SAMPLES,
        true_theta0=TRUE_THETA0,
        true_theta1=TRUE_THETA1,
        noise_std=NOISE_LEVEL
    )

    model_ne = Linear_Regression(lambda_val=0.0)
    model_ne.fit(X_b_train, y_train, solver="normal_equation")

    model_bgd = Linear_Regression(lambda_val=0.0)
    initial_theta_bgd = np.zeros((X_b_train.shape[1], 1))
    model_bgd.fit(X_b_train, y_train,
                  solver="batch_gd",
                  learning_rate=LEARNING_RATE,
                  n_iterations=N_ITERATIONS,
                  theta_initial=initial_theta_bgd)


    if model_ne.theta is not None:
        print(f"Normal Equation Theta: {model_ne.theta.ravel()}")
    else:
        print("Normal Equation training failed.")

    if model_bgd.theta is not None:
        print(f"Batch GD Theta       : {model_bgd.theta.ravel()}")
        if model_bgd.cost_history:
            print(f"Batch GD Final Cost  : {model_bgd.cost_history[-1]:.4f}")
    else:
        print("Batch Gradient Descent training failed.")

    if model_ne.theta is not None and model_bgd.theta is not None:
        diff = np.linalg.norm(model_ne.theta - model_bgd.theta)
        print(f"Difference (norm) between thetas: {diff:.6f}")
        if diff < 1e-2 :
             print("Theta values from both methods are very similar.")
        else:
             print("Theta values differ. Check GD parameters if discrepancy is large.")


    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    plt.scatter(X_feature_train, y_train, alpha=0.7, label='Data Points', color='skyblue', edgecolors='k', s=50)

    x_line_values = np.array([X_feature_train.min(), X_feature_train.max()]).reshape(-1, 1)
    x_line_values_b = np.c_[np.ones((2, 1)), x_line_values]

    if model_ne.theta is not None:
        y_predict_ne = model_ne.predict(x_line_values_b)
        plt.plot(x_line_values, y_predict_ne, "r-", linewidth=2.5, label=f'Normal Eq. (θ₀={model_ne.theta[0,0]:.2f}, θ₁={model_ne.theta[1,0]:.2f})')

    if model_bgd.theta is not None:
        y_predict_bgd = model_bgd.predict(x_line_values_b)
        plt.plot(x_line_values, y_predict_bgd, "g--", linewidth=2.5, label=f'Batch GD (θ₀={model_bgd.theta[0,0]:.2f}, θ₁={model_bgd.theta[1,0]:.2f})')

    plt.xlabel("Feature (X)", fontsize=14)
    plt.ylabel("Target (y)", fontsize=14)
    plt.title(f"Linear Regression Comparison (True θ₀={TRUE_THETA0}, θ₁={TRUE_THETA1})", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("linear_regression_comparison.png")
    plt.show()

    if model_bgd.theta is not None and model_bgd.cost_history:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(model_bgd.cost_history)), model_bgd.cost_history, color='purple', linewidth=2)
        plt.xlabel("Iteration", fontsize=14)
        plt.ylabel("Cost (MSE)", fontsize=14)
        plt.title(f"Batch Gradient Descent Cost Convergence (α={LEARNING_RATE})", fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("batch_gd_cost_history.png")
        plt.show()




if __name__ == "__main__":
    run_mvp_showcase()