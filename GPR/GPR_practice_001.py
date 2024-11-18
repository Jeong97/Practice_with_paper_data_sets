import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared
import GPy
import pyro
import pyro.contrib.gp as gp
import torch
import pandas as pd
from torch.optim import Adam

# Step 1: Data Generation and Preprocessing
np.random.seed(42)
X = np.random.rand(150, 100)  # 150 samples, 100 features
y = np.random.rand(150)      # Target values



# 입력 데이터 생성 (150개의 샘플, 100개의 특징)
n_samples = 150
n_features = 100
X = np.random.rand(n_samples, n_features)

# 타깃 값 계산 (입력 특징의 증가/감소에 따라 타깃 감소)
weights = 1 / (np.arange(1, n_features + 1))  # 특징별 반비례 가중치
row_effect = 1 / (np.arange(1, n_samples + 1))  # 행 인덱스에 따라 감소
y = (X @ weights) * row_effect  # 행과 특징 모두의 영향을 반영
y = y * -1  # 감소 특성 반영

# 노이즈 추가
noise = np.random.normal(0, 0.05, size=n_samples)  # 평균 0, 표준편차 0.05
y += noise


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Kernels for comparison
kernels_sklearn = {
    "RBF": RBF(length_scale=1.0),
    "Matern (ν=1.5)": Matern(length_scale=1.0, nu=1.5),
    "Matern (ν=2.5)": Matern(length_scale=1.0, nu=2.5),
    "Rational Quadratic": RationalQuadratic(length_scale=1.0, alpha=1.0),
    # "Periodic": ExpSineSquared(length_scale=1.0, periodicity=3.0),
}

# Results storage
results = {
    "Scikit-learn": {},
    "GPy": {},
    "Pyro": {},
}

# Step 2: Scikit-learn Implementation
for kernel_name, kernel in kernels_sklearn.items():
    try:
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1, random_state=42)
        gpr.fit(X_scaled, y)
        y_pred, y_std = gpr.predict(X_scaled, return_std=True)
        results["Scikit-learn"][kernel_name] = {
            "model": gpr,
            "y_pred": y_pred,
            "y_std": y_std,
            "log_marginal_likelihood": gpr.log_marginal_likelihood_value_
        }
    except Exception as e:
        print(f"Scikit-learn - {kernel_name}: Error - {e}")

# Step 3: GPy Implementation
for kernel_name, kernel in kernels_sklearn.items():
    try:
        if "Matern" in kernel_name:
            nu = 1.5 if "1.5" in kernel_name else 2.5
            kernel_gpy = GPy.kern.Matern32(input_dim=100) if nu == 1.5 else GPy.kern.Matern52(input_dim=100)
        elif kernel_name == "RBF":
            kernel_gpy = GPy.kern.RBF(input_dim=100)
        elif kernel_name == "Rational Quadratic":
            kernel_gpy = GPy.kern.RatQuad(input_dim=100)
        elif kernel_name == "Periodic":
            kernel_gpy = GPy.kern.PeriodicExponential(input_dim=100)
        else:
            continue

        model = GPy.models.GPRegression(X_scaled, y.reshape(-1, 1), kernel_gpy)
        model.optimize(messages=False)
        y_pred, y_var = model.predict(X_scaled)
        results["GPy"][kernel_name] = {
            "model": model,
            "y_pred": y_pred.ravel(),
            "y_std": np.sqrt(y_var.ravel()),
            "log_marginal_likelihood": model.log_likelihood()
        }
    except Exception as e:
        print(f"GPy - {kernel_name}: Error - {e}")

# Step 4: Pyro Implementation
X_torch = torch.tensor(X_scaled, dtype=torch.float32)
y_torch = torch.tensor(y, dtype=torch.float32)

for kernel_name, kernel in kernels_sklearn.items():
    try:
        if kernel_name == "RBF":
            kernel_pyro = gp.kernels.RBF(input_dim=100)
        elif kernel_name.startswith("Matern"):
            kernel_pyro = gp.kernels.Matern32(input_dim=100) if "1.5" in kernel_name else gp.kernels.Matern52(input_dim=100)
        elif kernel_name == "Rational Quadratic":
            kernel_pyro = gp.kernels.RationalQuadratic(input_dim=100)
        elif kernel_name == "Periodic":
            kernel_pyro = gp.kernels.Periodic(input_dim=100)
        else:
            continue

        # Create GPRegression model
        gpr_pyro = gp.models.GPRegression(X_torch, y_torch, kernel_pyro, noise=torch.tensor(0.1))

        # Define the optimizer
        optimizer = Adam(gpr_pyro.parameters(), lr=0.01)

        # Define the loss function
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss

        # Optimization loop
        num_steps = 100
        for step in range(num_steps):
            optimizer.zero_grad()
            loss = loss_fn(gpr_pyro.model, gpr_pyro.guide)
            loss.backward()
            optimizer.step()

        # Make predictions
        mean, var = gpr_pyro(X_torch, full_cov=False, noiseless=False)
        results["Pyro"][kernel_name] = {
            "model": gpr_pyro,
            "y_pred": mean.detach().numpy(),
            "y_std": torch.sqrt(var).detach().numpy(),
            "log_marginal_likelihood": -loss.item()
        }
        print(f"Pyro - {kernel_name}: Training complete, Log Marginal Likelihood = {-loss.item():.3f}")

    except Exception as e:
        print(f"Pyro - {kernel_name}: Error - {e}")

# Step 5: Visualization
plt.figure(figsize=(20, 10))
num_kernels = len(kernels_sklearn)
num_models = len(results)

for i, (model_name, model_results) in enumerate(results.items(), 1):
    for j, (kernel_name, result) in enumerate(model_results.items(), 1):
        plt.subplot(num_models, num_kernels, (i - 1) * num_kernels + j)
        y_pred = result["y_pred"]
        y_std = result["y_std"]
        plt.plot(y, label="True Values", color='black', alpha=0.6)
        plt.plot(y_pred, label=f"{kernel_name} Predictions", alpha=0.8)
        plt.fill_between(
            range(len(y)),
            y_pred - 1.96 * y_std,
            y_pred + 1.96 * y_std,
            color='blue', alpha=0.2, label="95% Confidence Interval"
        )
        plt.title(f"{model_name}: {kernel_name}")
        plt.legend()

plt.tight_layout()
plt.show()


# Step 6: Log Marginal Likelihood Comparison
print("\nLog-Marginal Likelihood Comparison:")
for model_name, model_results in results.items():
    print(f"\n{model_name}:")
    for kernel_name, result in model_results.items():
        print(f"  {kernel_name}: {result['log_marginal_likelihood']:.3f}")