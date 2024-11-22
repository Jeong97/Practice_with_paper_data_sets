import numpy as np
import optuna
import GPy
import gpflow
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Define a function to calculate performance metrics
def calculate_performance_metrics(y_true, y_pred, y_std):
    max_absolute_error = np.max(np.abs(y_true - y_pred) / np.abs(y_true)) * 100
    mean_absolute_error_perc = mean_absolute_error(y_true, y_pred) / np.mean(np.abs(y_true)) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) / np.mean(np.abs(y_true)) * 100
    lower_bound = y_pred - 1.96 * y_std
    upper_bound = y_pred + 1.96 * y_std
    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound)) * 100
    mean_std_dev = np.mean(y_std) / np.mean(np.abs(y_true)) * 100

    return {'Maximum Absolute Error (%)': max_absolute_error,
            'Mean Absolute Error (%)': mean_absolute_error_perc,
            'Root Mean Square Error (%)': rmse,
            'Coverage Probability (%)': coverage,
            'Mean Standard Deviation (%)': mean_std_dev}


# SK-learn method with Extended Optimization
def train_and_predict_SK_Bayesian(train_x, train_y, test_x, test_y):
    # 데이터 스케일링
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)

    # Bayesian Optimization
    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        length_scale = trial.suggest_float("length_scale", 0.1, 10.0)
        constant_value = trial.suggest_float("constant_value", 0.1, 10.0)
        noise_level = trial.suggest_float("noise_level", 1e-6, 1e-2, log=True)
        nu = trial.suggest_categorical("nu", [1.5])

        kernel = (
            C(constant_value=constant_value, constant_value_bounds=(1e-3, 1e3))
            * Matern(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2), nu=nu)
            + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-6, 1e1))
        )
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha, random_state=42)
        gpr.fit(train_x_scaled, train_y)
        y_pred, _ = gpr.predict(test_x_scaled, return_std=True)
        mse = mean_squared_error(test_y, y_pred)
        return mse

    # Optuna Study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    # 최적화된 GPR 모델
    best_params = study.best_params
    kernel = (
        C(constant_value=best_params["constant_value"], constant_value_bounds=(1e-3, 1e3))
        * Matern(length_scale=best_params["length_scale"], length_scale_bounds=(1e-2, 1e2), nu=best_params["nu"])
        + WhiteKernel(noise_level=best_params["noise_level"], noise_level_bounds=(1e-6, 1e1))
    )
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=best_params["alpha"], random_state=42, n_restarts_optimizer=20)
    gpr.fit(train_x_scaled, train_y)
    y_pred, y_std = gpr.predict(test_x_scaled, return_std=True)

    # Confidence intervals
    lower_bound = y_pred - 1.96 * y_std
    upper_bound = y_pred + 1.96 * y_std

    return y_pred, y_std, lower_bound, upper_bound, best_params


# GPy Method with Extended Optimization
def train_and_predict_GPy_Bayesian(train_x, train_y, test_x, test_y):
    # 데이터 스케일링
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)

    # Bayesian Optimization
    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
        length_scale = trial.suggest_float("length_scale", 0.5, 5.0)
        variance = trial.suggest_float("variance", 0.5, 5.0)
        noise_level = trial.suggest_float("noise_level", 1e-5, 1e-3, log=True)
        nu = trial.suggest_categorical("nu", [1.5, 2.5])

        kernel = GPy.kern.Matern32(
            input_dim=train_x.shape[1],
            lengthscale=length_scale,
            variance=variance
        ) + GPy.kern.White(input_dim=train_x.shape[1], variance=noise_level)

        model = GPy.models.GPRegression(train_x_scaled, train_y.reshape(-1, 1), kernel)
        model.Gaussian_noise.variance = alpha
        model.optimize(messages=False)

        y_pred, y_var = model.predict(test_x_scaled)
        mse = mean_squared_error(test_y, y_pred.ravel())
        if np.isnan(mse) or np.isinf(mse):
            return float("inf")
        return mse

    # Optuna Study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    # 최적화된 GPR 모델
    best_params = study.best_params
    kernel = GPy.kern.Matern32(
        input_dim=train_x.shape[1],
        lengthscale=best_params["length_scale"],
        variance=best_params["variance"]
    ) + GPy.kern.White(
        input_dim=train_x.shape[1],
        variance=best_params["noise_level"]
    )

    model = GPy.models.GPRegression(train_x_scaled, train_y.reshape(-1, 1), kernel)
    model.Gaussian_noise.variance = best_params["alpha"]
    model.optimize(messages=False)

    # 예측 결과
    y_pred, y_var = model.predict(test_x_scaled)
    y_std = np.sqrt(y_var.ravel())

    # 신뢰 구간 계산
    lower_bound = y_pred.ravel() - 1.96 * y_std
    upper_bound = y_pred.ravel() + 1.96 * y_std

    return y_pred.ravel(), y_std, lower_bound, upper_bound, best_params


def train_and_predict_GPflow_Bayesian(train_x, train_y, test_x, test_y):
    # 데이터 스케일링
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)

    # Bayesian Optimization
    def objective(trial):
        length_scale = trial.suggest_float("length_scale", 0.1, 10.0)
        variance = trial.suggest_float("variance", 0.1, 10.0)
        noise_level = trial.suggest_float("noise_level", 1e-6, 1e-2, log=True)

        # Matern 3/2 커널 정의
        kernel = gpflow.kernels.Matern32(lengthscales=length_scale, variance=variance)
        model = gpflow.models.GPR(
            data=(train_x_scaled, train_y.reshape(-1, 1)),
            kernel=kernel,
            mean_function=None
        )
        model.likelihood.variance.assign(noise_level)

        # 모델 학습
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)

        # 예측 및 MSE 계산
        mean, _ = model.predict_f(test_x_scaled)
        mse = mean_squared_error(test_y, mean.numpy().ravel())
        if np.isnan(mse) or np.isinf(mse):
            return float("inf")
        return mse

    # Optuna Study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    # 최적화된 GPR 모델
    best_params = study.best_params
    kernel = gpflow.kernels.Matern32(
        lengthscales=best_params["length_scale"],
        variance=best_params["variance"]
    )
    model = gpflow.models.GPR(
        data=(train_x_scaled, train_y.reshape(-1, 1)),
        kernel=kernel,
        mean_function=None
    )
    model.likelihood.variance.assign(best_params["noise_level"])

    # 최적화
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)

    # 예측
    mean, variance = model.predict_f(test_x_scaled)
    y_pred = mean.numpy().ravel()
    y_std = np.sqrt(variance.numpy().ravel())

    # 신뢰 구간 계산
    lower_bound = y_pred - 1.96 * y_std
    upper_bound = y_pred + 1.96 * y_std

    return y_pred, y_std, lower_bound, upper_bound, best_params
