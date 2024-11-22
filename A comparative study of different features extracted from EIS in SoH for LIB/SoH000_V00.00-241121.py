# import library
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# Graph Font setting
def setGraphFont():
    import matplotlib.font_manager as fm

    # import set fonts
    font_list = [font.name for font in fm.fontManager.ttflist]

    # set default font
    plt.rcParams['font.family'] = font_list[np.min([i for i in range(len(font_list)) if 'Times New Roman' in font_list[i]])]  # -12 : Times New Roman, -14 : Nanum gothic
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rc("axes", unicode_minus=False)

    # Configure rcParams axes.prop_cycle to simultaneously cycle cases and colors.
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors)
setGraphFont()



'''Set the DataFrame from raw_data of Capacity'''
# Set Capacity Raw data path
cap_path = "C:/Users/jeongbs1/오토실리콘/1. python_code/Practice_with_paper_data_sets/A comparative study of different features extracted from EIS in SoH for LIB/raw_data(SDI)"

# Load Raw data
cap_df = pd.read_csv(cap_path+"./Cell1-cell4.txt", skiprows=2, sep='\t')

# Set Columns
cap_df_col = ["cycle_1_3", "Cell1_cap", "Cell1_dec", "Cell3_cap", "Cell3_dec", "cycle_2_4", "Cell2_cap", "Cell2_dec", "Cell4_cap", "Cell4_dec"]

# Apply colums to DataFrame
cap_df.columns = cap_df_col

# Data processing for SoH calculation for each cell
cap_df = cap_df.replace("--", np.nan).astype(float)

# SoH calculation per cell
for i in range(1,5):
    cap_column = f"Cell{i}_cap"
    soh_column = f"Cell{i}_SoH"
    cap_df[soh_column] = round((cap_df[cap_column].astype(float) / 2.75) * 100, 2)

# Set SoH Dataframe column
SoH_Colunms = ['cycle_1_3', 'Cell1_SoH', 'Cell3_SoH', 'cycle_2_4', 'Cell2_SoH', 'Cell4_SoH']

# Make SoH Dataframe
SoH_df = cap_df[SoH_Colunms]

# Data settings for merging with EIS data
def process_soh_df(df, cycle_col, soh_col, cell_name):
    # Make SoH DataFrame by Cell
    df = df[[cycle_col, soh_col]].dropna().copy()
    # Make cycle, SoH DataFrame by Cell
    df.columns = ["Cycle", "SoH"]
    # Separation for merging with eis dataframe by cell
    df["Cell_Name"] = cell_name
    return df

# Apply to individual cells
SoH_df_cell1 = process_soh_df(SoH_df, 'cycle_1_3', 'Cell1_SoH', "Cell_1")
SoH_df_cell2 = process_soh_df(SoH_df, 'cycle_2_4', 'Cell2_SoH', "Cell_2")
SoH_df_cell3 = process_soh_df(SoH_df, 'cycle_1_3', 'Cell3_SoH', "Cell_3")
SoH_df_cell4 = process_soh_df(SoH_df, 'cycle_2_4', 'Cell4_SoH', "Cell_4")

# relocation column order before merging
SoH_df_cell1 = SoH_df_cell1[["Cell_Name", "Cycle", "SoH"]]
SoH_df_cell2 = SoH_df_cell2[["Cell_Name", "Cycle", "SoH"]]
SoH_df_cell3 = SoH_df_cell3[["Cell_Name", "Cycle", "SoH"]]
SoH_df_cell4 = SoH_df_cell4[["Cell_Name", "Cycle", "SoH"]]


# Set Capacity Raw data path
eis_forder_cell_path = "C:/Users/jeongbs1/오토실리콘/1. python_code/Practice_with_paper_data_sets/A comparative study of different features extracted from EIS in SoH for LIB/raw_data(SDI)"
# Make folder list
eis_forder_cell_list = [forder for forder in os.listdir(eis_forder_cell_path) if os.path.isdir(os.path.join(eis_forder_cell_path, forder))]
# Make dictionary for each cell
eis_raw_df = {cell: {} for cell in ["Cell1", "Cell2", "Cell3", "Cell4"]}

# EIS raw data upload loop creation for each cell and cycle
for cell_name in eis_forder_cell_list:
    # Specify file path per cell
    cell_path = Path(eis_forder_cell_path) / cell_name
    # Cycle folder list append
    cycle_list = [cycle for cycle in os.listdir(cell_path) if "cycle" in cycle]
    # Load file with SoC 50% and temperature 25degree for each cycle
    for cycle_name in cycle_list:
        # Specify SoC 50% folder path
        cycle_path = cell_path / cycle_name / "50soc"
        # Temperature 25 degrees file search
        eis_file = next((file for file in os.listdir(cycle_path) if "25d" in file), None)
        # If the file exists, append it to the dataframe
        if eis_file:
            # Save cell information in dataframe with cycle_soc_temp
            df_name = str(cycle_name.split("cycle")[1])+"_"+str(re.findall(r'\d+',(eis_file.split('_')[1]))[0])+"_"+str(re.findall(r'\d+',(eis_file.split('_')[2]))[0])
            file_path = cycle_path / eis_file
            eis_raw_df[cell_name][df_name] = pd.read_csv(file_path, sep=r'\s+', header=None, names=["Frequency", "Real_Z", "Imaginary_Z"])

# make cell list
cell_num = list(eis_raw_df.keys())

for cell in cell_num:
    plt.figure()
    for num in eis_raw_df[cell]:
        plt.plot((eis_raw_df[cell][num]["Real_Z"]), -(eis_raw_df[cell][num]["Imaginary_Z"]), "o")
    plt.xlabel("Real Impedance(mOhm)")
    plt.ylabel("Imaginary Impedance(mOhm)")
    plt.title(cell, fontweight='bold', fontsize=16)
    plt.tight_layout()



# make data frame for each cell
stack_eis_row_df = {cell: pd.DataFrame() for cell in cell_num}
# A loop that transposes rows/columns for EIS files for each cycle within cells saved from EIS raw data and stacks them downwards into one file.
for cell in range(len(cell_num)):
    # make cell info list
    cell_info_num = list((eis_raw_df[cell_num[cell]].keys()))
    transformed_data = []
    # Cycle-by-cycle EIS value staking loop
    for cyc in range(len(cell_info_num)):
        df_dummy = eis_raw_df[cell_num[cell]][cell_info_num[cyc]]
        tran_f_col = {}
        # loop to convert values from individual rows into columns
        for i, row in df_dummy.iterrows():
            freq = f"{row['Frequency']:.2f}"  # 소수 둘째 자리까지 표현
            tran_f_col["Cycle"] = float(cell_info_num[cyc].split("_")[0])
            tran_f_col["SoC"] = int(cell_info_num[cyc].split("_")[1])
            tran_f_col["Temp"] = int(cell_info_num[cyc].split("_")[2])
            tran_f_col[f"F_{freq}_Re_Z"] = row['Real_Z']
            tran_f_col[f"F_{freq}_Im_Z"] = row['Imaginary_Z']
        # Save to data frame
        transformed_data.append(pd.DataFrame([tran_f_col]))
    stack_eis_row_df[cell_num[cell]] = pd.concat(transformed_data, axis=0, ignore_index=True)
    # Specify columns in cell info-real-imaginary order
    info_cols = [col for col in stack_eis_row_df[cell_num[cell]].columns if not 'Z' in col]
    real_cols = [col for col in stack_eis_row_df[cell_num[cell]].columns if '_Re_Z' in col]
    imaginary_cols = [col for col in stack_eis_row_df[cell_num[cell]].columns if '_Im_Z' in col]
    ordered_cols = info_cols + real_cols + imaginary_cols
    # Apply specified column order
    stack_eis_row_df[cell_num[cell]] = stack_eis_row_df[cell_num[cell]][ordered_cols]
    # Sort ascending by cycle
    stack_eis_row_df[cell_num[cell]].sort_values(by="Cycle", ascending=True, ignore_index=True, inplace=True)


# Merge SoH dataframe and eis dataframe by cell
merged_dfs = []
for i, soh_df in enumerate([SoH_df_cell1, SoH_df_cell2, SoH_df_cell3, SoH_df_cell4]):
    merged_dfs.append(pd.merge(soh_df, stack_eis_row_df[cell_num[i]], on="Cycle", how="inner"))

# Merge data frames by merged cells into one
merged_df = pd.concat(merged_dfs, ignore_index=True)
merged_df["Cycle"] = merged_df["Cycle"].astype(int)
# merged_df.to_excel("C:/Users/jeongbs1/Downloads/merge_df.xlsx", index=False)

'''Preprocessing Complete'''



# Correlation between Features and Target
numeric_df, corr_df = {}, {}
for num in range(len(merged_dfs)):
    numeric_df[num] = merged_dfs[num].drop(columns=["Cycle", "SoC", "Temp", "Cell_Name"], axis=1)
    corr_df[num] = (numeric_df[0].corr()["SoH"])
    corr_df[num] = corr_df[num][corr_df[num].abs() > 0.8]
    print(len(corr_df[num]))




'''Machine Learning'''
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import gpflow
from sklearn.pipeline import Pipeline
import optuna
import GPy

# Separate by case
def separate_cases(df, cell_name):
    train_df = df[df['Cell_Name'] != cell_name].copy().reset_index(drop=True)
    test_df = df[df['Cell_Name'] == cell_name].copy().reset_index(drop=True)
    return train_df, test_df


# Define a function to calculate performance metrics
def calculate_performance_metrics(y_true, y_pred, y_std):
    max_absolute_error = np.max(np.abs(y_true - y_pred) / np.abs(y_true)) * 100
    mean_absolute_error_perc = mean_absolute_error(y_true, y_pred) / np.mean(np.abs(y_true)) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) / np.mean(np.abs(y_true)) * 100
    lower_bound = y_pred - 1.96 * y_std
    upper_bound = y_pred + 1.96 * y_std
    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound)) * 100
    mean_std_dev = np.mean(y_std) / np.mean(np.abs(y_true)) * 100

    return {'Cell Name': f'Cell {i + 1}',
            'Maximum Absolute Error (%)': max_absolute_error,
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


''' BroadBand '''
# Create a data frame to apply to ML Model
Broadband_df = merged_df.copy()

# Create a data frame for each case by calling a function
Broadband_cases = {"train" : {}, "test" : {}}
for i in np.arange(1,5):
    train_df, test_df = separate_cases(Broadband_df, f'Cell_{i}')
    Broadband_cases["train"][f'case{i}_train_df'] = train_df
    Broadband_cases["test"][f'case{i}_test_df'] = test_df

# Separate train and test dataframes
B_train_list = list(Broadband_cases["train"].keys())
B_test_list = list(Broadband_cases["test"].keys())

# Define the list of input columns (excluding unnecessary columns)
B_input_drop_col = ["Cell_Name", "Cycle", "SoH", "SoC", "Temp"]

# Initialize the list to store performance indicator
performance_metrics = {"SK-learn" : [], "GPy" : [], "GPflow" : []}
predicted_results = {"SK-learn" : [], "GPy" : [], "GPflow" : []}
Im_methode = ["SK-learn", "GPy", "GPflow"]

for method in Im_methode:
    for i in range(len(B_train_list)):
        # 데이터 분리
        train_x_raw = Broadband_cases["train"][B_train_list[i]].drop(columns=B_input_drop_col).values
        train_y_raw = Broadband_cases["train"][B_train_list[i]]["SoH"].values
        test_x_raw = Broadband_cases["test"][B_test_list[i]].drop(columns=B_input_drop_col).values
        test_y_raw = Broadband_cases["test"][B_test_list[i]]["SoH"].values
        # 모델 학습 및 예측
        if method == "SK-learn":
            y_pred, y_std, lower_bound, upper_bound, best_params = train_and_predict_SK_Bayesian(train_x_raw, train_y_raw, test_x_raw, test_y_raw)
        elif method == "GPy":  # GPy 방식
            y_pred, y_std, lower_bound, upper_bound, best_params = train_and_predict_GPy_Bayesian(train_x_raw, train_y_raw, test_x_raw, test_y_raw)
        elif method == "GPflow":
            y_pred, y_std, lower_bound, upper_bound, best_params = train_and_predict_GPflow_Bayesian(train_x_raw, train_y_raw, test_x_raw, test_y_raw)    
        # 성능 지표 계산
        metrics = calculate_performance_metrics(test_y_raw, y_pred, y_std)
        metrics["Method"] = method
        metrics["Case"] = f"Cell {i + 1}"
        
        performance_metrics[method].append(metrics)
        predicted_results[method].append({
                                            "Actual": test_y_raw.tolist(),
                                            "Predicted": y_pred.tolist(),
                                            "Lower Bound": lower_bound.tolist(),
                                            "Upper Bound": upper_bound.tolist(),
                                            "Case": f"Cell {i + 1}",})

# 출력 결과를 데이터프레임으로 저장
B_predicted_results_df = {method: pd.DataFrame(predicted_results[method]) for method in ["SK-learn", "GPy", "GPflow"]}

# 두 결과를 하나의 데이터프레임으로 결합
B_performance_df = pd.concat([pd.DataFrame(performance_metrics["SK-learn"]),
                              pd.DataFrame(performance_metrics["GPy"]),
                              pd.DataFrame(performance_metrics["GPflow"])]).reset_index(drop=True)

# Calculate Cycle, Actual SoH, Predicted SoH, and confidence intervals for each method
colors = sns.color_palette("Spectral", 3)
actual_color = '#E74C3C'
predicted_color = '#2E86C1'
confidence_color = '#808080'

# Methods to loop over
for method in Im_methode:
    plt.figure(figsize=(12, 8))
    for n in range(len(B_train_list)):
        plt.subplot(2, 2, n + 1)
        x = Broadband_cases["test"][B_test_list[n]]["Cycle"]
        actual_soh = B_predicted_results_df[method]["Actual"][n]
        predicted_soh = B_predicted_results_df[method]["Predicted"][n]
        lower_bound = B_predicted_results_df[method]["Lower Bound"][n]
        upper_bound = B_predicted_results_df[method]["Upper Bound"][n]

        # Draw a graph
        plt.plot(x, actual_soh, '*', color=actual_color, markersize=7, label='Actual SoH')  # 실제값 색상
        plt.plot(x, predicted_soh, '-', color=predicted_color, linewidth=3, label='Predicted SoH')  # 예측값 색상
        plt.fill_between(x, lower_bound, upper_bound, color=confidence_color, alpha=0.3, label='95% Confidence Interval')  # 신뢰 구간 색상

        # Add title and settings to each subplot
        plt.xlabel("Cycle", fontsize=12, fontweight='bold')
        plt.ylabel("SoH (%)", fontsize=12, fontweight='bold')
        plt.title(f"{method} SoH Prediction - Cell {n+1}", fontsize=14, fontweight='bold')
        plt.legend(loc="upper right", fontsize="small")

    # Set overall layout and title
    plt.suptitle(f"{method} Broadband SoH Prediction for Each Cell", fontsize=16, fontweight='bold')
    plt.tight_layout()


# List of performance indicators
metrics = ["Maximum Absolute Error (%)", "Mean Absolute Error (%)",
           "Root Mean Square Error (%)", "Coverage Probability (%)",
           "Mean Standard Deviation (%)"]

# Iterate over methods
for method in Im_methode:
    # Filter performance dataframe for the specific method
    method_performance_df = B_performance_df[B_performance_df["Method"] == method]
    # Define colors for each case using the Seaborn palette
    colors = sns.color_palette("Set2", len(method_performance_df["Case"]))
    # Create a 2x3 subplot (including one empty spot)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    # Draw a bar graph for each performance indicator
    for i, metric in enumerate(metrics):
        row, col = divmod(i, 3)
        # Create a bar graph with the same color for each Case
        axes[row, col].bar(method_performance_df["Case"], method_performance_df[metric], color=colors, edgecolor='black')
        axes[row, col].set_title(metric, fontsize=12, fontweight='bold')
        axes[row, col].set_ylabel(metric, fontsize=10, fontweight='bold')
        axes[row, col].grid(True, linestyle='--', alpha=0.7)
    # Hide empty subplots
    axes[1, 2].axis("off")
    # Set X-axis label
    for ax in axes[1, :2]:  # 마지막 행의 첫 두 축에만 x축 레이블 추가
        ax.set_xlabel("Cell Name", fontsize=12, fontweight='bold')
    # Set overall layout and title for the method
    plt.suptitle(f"{method} Broadband Performance Metrics Comparison by Cell", fontsize=16, fontweight='bold')
    plt.tight_layout()


# Define colors for each method using the Seaborn palette
colors = sns.color_palette("Set1", len(Im_methode))
# Create a 2x3 subplot for comparison (including one empty spot)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
# Iterate over performance metrics to compare methods
for i, metric in enumerate(metrics):
    row, col = divmod(i, 3)
    # Create a bar graph comparing methods for the current metric
    comparison_data = B_performance_df.pivot(index="Case", columns="Method", values=metric)
    comparison_data.plot(kind="bar", ax=axes[row, col], color=colors, edgecolor='black')
    # Set titles and labels
    axes[row, col].set_title(metric, fontsize=12, fontweight='bold')
    axes[row, col].set_ylabel(metric, fontsize=10, fontweight='bold')
    axes[row, col].set_xlabel("Cell Name", fontsize=10, fontweight='bold')
    axes[row, col].grid(True, linestyle='--', alpha=0.7)
    axes[row, col].legend(title="Method", fontsize=8, loc="upper center")
# Hide empty subplot
axes[1, 2].axis("off")
# Set overall layout
plt.suptitle("Broadband Performance Metrics Comparison by Metric and Method", fontsize=16, fontweight='bold')
plt.tight_layout()


B_perf_com_df = (B_performance_df.drop(columns=["Cell Name", "Case"])).groupby("Method").mean()
# Define colors for each method using the Seaborn palette
colors = sns.color_palette("Set1", len(Im_methode))
# Create a 2x3 subplot for comparison (including one empty spot)
# 서브플롯 생성
fig, axes = plt.subplots(1, len(metrics), figsize=(14, 6))
# 각 성능 지표에 대해 막대 그래프 생성
for i, metric in enumerate(metrics):
    ax = axes[i]
    B_perf_com_df[metric].plot(kind="bar", ax=ax, color=colors, edgecolor="black")
    ax.set_title(metric, fontsize=12, fontweight="bold")
    ax.set_ylabel("Value", fontsize=10, fontweight="bold")
    ax.set_xlabel("Method", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.tick_params(axis="x", rotation=0)
# 전체 레이아웃 조정
plt.suptitle("Broadband Performance Comparison by Method", fontsize=16, fontweight="bold")
plt.tight_layout()



''' Fixed_Frequency '''
### Study_1 : Fixed Version
fixed_df = merged_df.copy()

cell_info_col = ["Cell_Name", "Cycle", "SoH"]
x1_f_col = [col for col in fixed_df.columns if any(keyword in col for keyword in cell_info_col+[str(1.00)])]
x2_f_col = [col for col in fixed_df.columns if any(keyword in col for keyword in cell_info_col+[str(5.0)])]
x3_f_col = [col for col in fixed_df.columns if any(keyword in col for keyword in cell_info_col+[str(10.00)])]
x7_f_col = [col for col in fixed_df.columns if any(keyword in col for keyword in cell_info_col+[str(1.00), str(5.0), str(10.00)])]

fixed_cases_list = [x1_f_col, x2_f_col, x3_f_col, x7_f_col]
fixed_case_df = {}
for i, case in enumerate(fixed_cases_list):
    fixed_case_df[f"case_{i+1}"] = fixed_df[case]

# Create a data frame for each case by calling a function
fixed_hold_out = {num: {"train": {}, "test": {}} for num in range(len(fixed_case_df))}
for num, case_key in enumerate(fixed_case_df.keys()):
    for i in range(1, 5):
        train_df, test_df = separate_cases(fixed_case_df[case_key], f'Cell_{i}')
        fixed_hold_out[num]["train"][f'case{i}_train_df'] = train_df
        fixed_hold_out[num]["test"][f'case{i}_test_df'] = test_df


# 성능 지표 및 예측 결과 저장 초기화
F_performance_metrics = {"SK-learn" : [], "GPy" : [], "GPflow" : []}
F_predicted_results = {"SK-learn": [], "GPy" : [], "GPflow" : []}

# 각 fixed_case 반복
for num in range(len(fixed_cases_list)):
    # Train/Test 데이터 분리
    F_train_list = list(fixed_hold_out[num]["train"].keys())
    F_test_list = list(fixed_hold_out[num]["test"].keys())

    for method in Im_methode:
        for i in range(len(F_train_list)):
            # 훈련 및 테스트 데이터 준비
            train_x_raw = fixed_hold_out[num]["train"][F_train_list[i]].drop(columns=["Cell_Name", "Cycle", "SoH"]).values
            train_y_raw = fixed_hold_out[num]["train"][F_train_list[i]]["SoH"].values
            test_x_raw = fixed_hold_out[num]["test"][F_test_list[i]].drop(columns=["Cell_Name", "Cycle", "SoH"]).values
            test_y_raw = fixed_hold_out[num]["test"][F_test_list[i]]["SoH"].values
            # 모델 학습 및 예측
            if method == "SK-learn":
                y_pred, y_std, lower_bound, upper_bound, best_params = train_and_predict_SK_Bayesian(train_x_raw, train_y_raw, test_x_raw, test_y_raw)
            elif method == "GPy":  # GPy 방식
                y_pred, y_std, lower_bound, upper_bound, best_params = train_and_predict_GPy_Bayesian(train_x_raw, train_y_raw, test_x_raw, test_y_raw)
            elif method == "GPflow":
                y_pred, y_std, lower_bound, upper_bound, best_params = train_and_predict_GPflow_Bayesian(train_x_raw, train_y_raw, test_x_raw, test_y_raw)
            # 성능 지표 계산
            metrics = calculate_performance_metrics(test_y_raw, y_pred, y_std)
            metrics["Method"] = method
            metrics["Case"] = f"Fixed {num + 1}"
            F_performance_metrics[method].append(metrics)
            # 예측 결과 저장
            F_predicted_results[method].append({
                "Case": f"Fixed {num + 1}",
                "Cell Name" : f"Cell {i + 1}",
                "Actual": test_y_raw.tolist(),
                "Predicted": y_pred.tolist(),
                "Lower Bound": lower_bound.tolist(),
                "Upper Bound": upper_bound.tolist(),})

# 예측 결과 데이터프레임으로 변환
F_predicted_results_df = {method: pd.DataFrame(F_predicted_results[method]) for method in Im_methode}

# 성능 지표 데이터프레임으로 변환
F_performance_df = pd.concat([(pd.DataFrame(F_performance_metrics["SK-learn"])),
                              (pd.DataFrame(F_performance_metrics["GPy"])),
                              (pd.DataFrame(F_performance_metrics["GPflow"]))]).reset_index(drop=True)
F_performance_df.loc[F_performance_df["Case"] == "Fixed 4", "Case"] = "Fixed 7"

# Iterate over Fixed Cases
fixed_plot_cases = ["Fixed 1", "Fixed 2", "Fixed 3", "Fixed 7"]

compare_F_M_per = {}
for f in fixed_plot_cases:
    compare_F_M_per[f] = F_performance_df[F_performance_df["Case"]==f]
    compare_F_M_per[f] = (compare_F_M_per[f].drop(columns=["Cell Name", "Case"])).groupby("Method").mean()



# Seaborn 스타일 설정
sns.set(style="darkgrid")
# 색상 리스트 정의
# colors = sns.color_palette("Spectral", 4)  # Fixed 케이스별 색상
actual_color = '#E74C3C'  # 실제값: 빨간색
# 플롯 생성
fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # 2x2 서브플롯 생성
# 범례 핸들 및 라벨 저장용 리스트
legend_handles = []
legend_labels = []
# 각 Fixed 케이스 반복 (Fixed 1 ~ Fixed 4)
for f_idx, f_case in enumerate(fixed_plot_cases):
    method = Im_methode[1]  # 선택된 메서드 (첫 번째 메서드)
    # Fixed 케이스에 해당하는 데이터 필터링
    fixed_plot_df = F_predicted_results_df[method][F_predicted_results_df[method]["Case"] == f_case].reset_index(drop=True)
    # 각 Cell에 대해 서브플롯 생성
    for n in range(len(fixed_plot_df)):
        ax = axes[n // 2, n % 2]  # 현재 서브플롯 지정
        x = fixed_hold_out[n]["test"][F_test_list[n]]["Cycle"].values  # Cycle 값
        actual_soh = fixed_plot_df["Actual"].reset_index(drop=True)[n]  # 실측값
        predicted_soh = fixed_plot_df["Predicted"].reset_index(drop=True)[n]  # 예측값
        lower_bound = fixed_plot_df["Lower Bound"].reset_index(drop=True)[n]  # 신뢰 구간 하한
        upper_bound = fixed_plot_df["Upper Bound"].reset_index(drop=True)[n]  # 신뢰 구간 상한
        # 실측값 플롯
        actual_plot, = ax.plot(x, actual_soh, '*', color=actual_color, markersize=7, label='Actual SoH')
        # 예측값 플롯
        predicted_plot, = ax.plot(x, predicted_soh, '-', linewidth=3, label=f'{f_case} Predicted SoH')
        # 신뢰 구간 플롯
        confidence_plot = ax.fill_between(x, lower_bound, upper_bound, alpha=0.3, label=f'{f_case} 95% Confidence Interval')
        # 서브플롯 제목 및 축 레이블 설정
        ax.set_xlabel("Cycle", fontsize=12, fontweight='bold')  # X축 이름
        ax.set_ylabel("SoH (%)", fontsize=12, fontweight='bold')  # Y축 이름
        ax.set_title(f"Cell {n + 1}", fontsize=14, fontweight='bold')  # 서브플롯 제목

        # 실측값 플롯 (범례에 사용할 핸들 추가)
        if f_idx == 0 and n == 0:  # 첫 번째 루프에서만 범례 저장
            actual_plot, = ax.plot(x, actual_soh, '*', color=actual_color, markersize=7, label='Actual SoH')
            if not legend_handles:  # 중복 방지
                legend_handles.append(actual_plot)
                legend_labels.append('Actual SoH')
        if n == 0:  # 첫 번째 Cell만 해당 f_case의 범례 저장
            legend_handles.append(predicted_plot)
            legend_labels.append(f'{f_case} - Predicted SoH')
            # fill_between 핸들은 범례용 패치로 따로 추가
            from matplotlib.patches import Patch
            confidence_patch = Patch(color=predicted_plot.get_color(), alpha=0.3,label=f'{f_case} 95% - Confidence Interval')
            legend_handles.append(confidence_patch)
            legend_labels.append(f'{f_case} - 95% Confidence Interval')
# 전체 제목 설정
plt.suptitle(f"{method} SoH Predictions", fontsize=16, fontweight='bold')
plt.tight_layout()  # 여백 조정
# 범례만 표시할 새로운 창 생성
fig_legend = plt.figure()  # 새 창 크기 지정
fig_legend.legend(handles=legend_handles, labels=legend_labels, loc='center', ncol=1, fontsize=10, frameon=True)  # 중앙에 범례 표시
plt.axis('off')  # 축 제거
plt.title("Legend", fontsize=14, fontweight='bold')  # 범례 창 제목
plt.tight_layout()


# 2x3 서브플롯 생성 (마지막 subplot은 비어 있게 유지)
fixed_compare_df = F_performance_df[F_performance_df["Method"]=="SK-learn"].reset_index(drop=True)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
# 각 성능 지표에 대해 반복
for i, metric in enumerate(metrics):
    row, col = divmod(i, 3)
    ax = axes[row, col]
    # 지표에 따라 Fixed 모델 비교 데이터 생성
    metric_data = fixed_compare_df.reset_index()[["Case", metric]]  # Case와 해당 성능 지표만 선택
    # 막대 그래프 그리기
    sns.barplot(
        x="Case",
        y=metric,
        data=metric_data,
        ax=ax,
        palette=sns.color_palette("Set2", len(metric_data["Case"].unique())),
        hue="Case",  # 경고 제거를 위해 hue 추가
        dodge=False,  # hue가 있을 경우 막대 간격 이동 방지
        legend=False)  # 범례 제거
    # 서브플롯 제목 및 레이블 설정
    ax.set_title(metric, fontsize=12, fontweight='bold')  # 서브플롯 제목
    ax.set_ylabel(metric, fontsize=10, fontweight='bold')  # Y축 이름
    ax.set_xlabel("Case", fontsize=10, fontweight='bold')  # X축 이름
    ax.grid(True, linestyle='--', alpha=0.7)  # 격자 추가
    # X축 라벨 각도 조정 (Case 이름이 겹치지 않게)
    ax.tick_params(axis='x')
# 빈 서브플롯 숨기기
if len(metrics) < 6:
    for j in range(len(metrics), 6):
        row, col = divmod(j, 3)
        axes[row, col].axis("off")  # 빈 서브플롯 비활성화
# 전체 제목 설정
plt.suptitle("Comparison of Performance Metrics for Cases", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])  # 여백 조정



# Comparison between Broadband and Fixed 7\
B_perf_com_df.loc["GPflow",:]
compare_F_M_per["Fixed 7"].loc["GPflow",:]

metrics = ["Maximum Absolute Error (%)", "Mean Absolute Error (%)",
           "Root Mean Square Error (%)", "Coverage Probability (%)",
           "Mean Standard Deviation (%)"]

for f in fixed_plot_cases:
    # Define colors for each method using the Seaborn palette
    colors = sns.color_palette("Set1", len(Im_methode))
    # Create a 2x3 subplot for comparison (including one empty spot)
    # 서브플롯 생성
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 6))
    # 각 성능 지표에 대해 막대 그래프 생성
    for i, metric in enumerate(metrics):
        ax = axes[i]
        compare_F_M_per[f][metric].plot(kind="bar", ax=ax, color=colors, edgecolor="black")
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylabel("Value", fontsize=10, fontweight="bold")
        ax.set_xlabel("Method", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.tick_params(axis="x", rotation=0)
    # 전체 레이아웃 조정
    plt.suptitle(f"{f} Performance Comparison by Method", fontsize=16, fontweight="bold")
    plt.tight_layout()


compare_dff = pd.concat([(pd.DataFrame(B_perf_com_df.loc["GPy",:])),(pd.DataFrame(compare_F_M_per["Fixed 7"].loc["GPy",:]))],axis=1)
compare_dff.columns = ["Broadband", "Fixed_7"]
# Define colors for each method using the Seaborn palette
colors = sns.color_palette("Set1", 2)
# Create a 2x3 subplot for comparison (including one empty spot)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
# Iterate over performance metrics to compare methods
for i, metric in enumerate(metrics):
    row, col = divmod(i, 3)
    # Create a bar graph comparing methods for the current metric
    comparison_data = compare_dff.loc[metric]  # 선택된 metric에 대한 행 데이터
    comparison_data.plot(kind="bar", ax=axes[row, col], color=colors, edgecolor='black',legend=False)
    # Set titles and labels
    axes[row, col].set_title(metric, fontsize=12, fontweight='bold')  # 서브플롯 제목
    axes[row, col].set_ylabel(metric, fontsize=10, fontweight='bold')  # Y축 이름
    axes[row, col].grid(True, linestyle='--', alpha=0.7)  # 격자 추가
    axes[row, col].set_xticklabels(comparison_data.index, rotation=0, fontsize=10, fontweight='bold')
# Hide empty subplot (if any)
if len(metrics) < 6:
    for j in range(len(metrics), 6):
        row, col = divmod(j, 3)
        axes[row, col].axis("off")  # 빈 서브플롯 숨기기
# Set overall layout
plt.suptitle("Comparison by Metric", fontsize=16, fontweight='bold')
plt.tight_layout()  # 레이아웃 조정










''''''
train_x_raw = Broadband_cases["train"][B_train_list[-1]].drop(columns=B_input_drop_col).values
train_y_raw = Broadband_cases["train"][B_train_list[-1]]["SoH"].values
test_x_raw = Broadband_cases["test"][B_test_list[-1]].drop(columns=B_input_drop_col).values
test_y_raw = Broadband_cases["test"][B_test_list[-1]]["SoH"].values

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic

# 1. 데이터 준비
train_x = train_x_raw
train_y = train_y_raw
test_x = test_x_raw
test_y = test_y_raw
cycle = Broadband_cases["test"][B_test_list[-1]]["Cycle"].values

# 입력값 스케일링
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)


# 2. Scikit-Learn 모델 학습 및 예측
def train_and_predict_sklearn():
    kernels = {
        "RBF": RBF(length_scale=1.0),
        "Matern(1.5)": Matern(length_scale=1.0, nu=1.5),
        "Matern(2.5)": Matern(length_scale=1.0, nu=2.5),
        "Rational": RationalQuadratic(length_scale=1.0, alpha=1.0)
    }
    results = {}
    for kernel_name, kernel in kernels.items():
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
        gpr.fit(train_x_scaled, train_y)
        mean_prediction, std_prediction = gpr.predict(test_x_scaled, return_std=True)
        results[kernel_name] = {"mean": mean_prediction.ravel(), "std": std_prediction.ravel()}
    return results

# 3. GPy 모델 학습 및 예측
def train_and_predict_gpy():
    import GPy
    kernels = {
        "RBF": GPy.kern.RBF(input_dim=train_x_scaled.shape[1], variance=1.0, lengthscale=1.0),
        "Matern(1.5)": GPy.kern.Matern32(input_dim=train_x_scaled.shape[1], variance=1.0, lengthscale=1.0),
        "Matern(2.5)": GPy.kern.Matern52(input_dim=train_x_scaled.shape[1], variance=1.0, lengthscale=1.0),
        "Rational": GPy.kern.RatQuad(input_dim=train_x_scaled.shape[1], variance=1.0, lengthscale=1.0, power=1.0)
    }
    results = {}
    # 출력 데이터(train_y)를 2차원 배열로 변환
    train_y_reshaped = train_y.reshape(-1, 1)  # 1차원 배열을 2차원으로 변환
    for kernel_name, kernel in kernels.items():
        model = GPy.models.GPRegression(train_x_scaled, train_y_reshaped, kernel)
        model.optimize()
        mean_prediction, var_prediction = model.predict(test_x_scaled)
        results[kernel_name] = {"mean": mean_prediction.ravel(), "std": np.sqrt(var_prediction).ravel()}
    return results

# 4. GPflow 모델 학습 및 예측
def train_and_predict_gpflow():
    import gpflow
    kernels = {
        "RBF": gpflow.kernels.SquaredExponential(),
        "Matern(1.5)": gpflow.kernels.Matern32(),
        "Matern(2.5)": gpflow.kernels.Matern52(),
        "Rational": gpflow.kernels.RationalQuadratic()
    }
    results = {}
    # 출력 데이터를 2차원 배열로 변환
    train_y_reshaped = train_y.reshape(-1, 1)
    for kernel_name, kernel in kernels.items():
        model = gpflow.models.GPR(data=(train_x_scaled, train_y_reshaped), kernel=kernel, mean_function=None)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)
        mean_prediction, var_prediction = model.predict_f(test_x_scaled)
        results[kernel_name] = {"mean": mean_prediction.numpy().ravel(), "std": np.sqrt(var_prediction.numpy()).ravel()}
    return results

# 5. GPyTorch 모델 학습 및 예측
import torch
from gpytorch.kernels import Kernel  # Kernel 클래스 임포트
from gpytorch.constraints import Positive  # Positive 제약 조건 임포트

import torch
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel, RBFKernel
from gpytorch.constraints import Positive

# Rational Quadratic Kernel 구현
import torch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive

class RationalQuadraticKernel(Kernel):
    def __init__(self, lengthscale_prior=None, alpha_prior=None, **kwargs):
        super().__init__(**kwargs)

        # 길이 스케일(lengthscale) 파라미터
        self.register_parameter(
            name="raw_lengthscale",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )
        self.register_constraint("raw_lengthscale", Positive())

        # 매끄러움(alpha) 파라미터
        self.register_parameter(
            name="raw_alpha",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )
        self.register_constraint("raw_alpha", Positive())

        if lengthscale_prior is not None:
            self.register_prior(
                "lengthscale_prior",
                lengthscale_prior,
                lambda m: m.lengthscale,
                lambda m, v: m._set_lengthscale(v),
            )
        if alpha_prior is not None:
            self.register_prior(
                "alpha_prior",
                alpha_prior,
                lambda m: m.alpha,
                lambda m, v: m._set_alpha(v),
            )

    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    @lengthscale.setter
    def lengthscale(self, value):
        self._set_lengthscale(value)

    def _set_lengthscale(self, value):
        self.initialize(
            raw_lengthscale=self.raw_lengthscale_constraint.inverse_transform(value)
        )

    @property
    def alpha(self):
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value):
        self._set_alpha(value)

    def _set_alpha(self, value):
        self.initialize(
            raw_alpha=self.raw_alpha_constraint.inverse_transform(value)
        )

    def forward(self, x1, x2, diag=False, **params):
        # x1과 x2의 크기를 일치시켜 쌍(pairwise) 계산
        diff = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).pow(2).sum(dim=-1)  # Pairwise squared differences
        base = 1 + diff / (2 * self.alpha * self.lengthscale.pow(2))
        return base.pow(-self.alpha)

# GPyTorch 학습 및 예측
def train_and_predict_gpytorch():
    import gpytorch

    # GPyTorch 모델 정의
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, kernel_type):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()

            # 커널 종류에 따라 설정
            if kernel_type == "RBF":
                self.covar_module = ScaleKernel(RBFKernel())
            elif kernel_type == "Matern(1.5)":
                self.covar_module = ScaleKernel(MaternKernel(nu=1.5))
            elif kernel_type == "Matern(2.5)":
                self.covar_module = ScaleKernel(MaternKernel(nu=2.5))
            elif kernel_type == "Rational":
                self.covar_module = ScaleKernel(RationalQuadraticKernel())
            else:
                raise ValueError(f"Unknown kernel type: {kernel_type}")

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # 데이터 준비
    train_x_torch = torch.tensor(train_x_scaled, dtype=torch.float32)
    train_y_torch = torch.tensor(train_y.ravel(), dtype=torch.float32)
    test_x_torch = torch.tensor(test_x_scaled, dtype=torch.float32)

    kernels = ["RBF", "Matern(1.5)", "Matern(2.5)", "Rational"]
    results = {}

    for kernel_name in kernels:
        print(f"Training with kernel: {kernel_name}")
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x_torch, train_y_torch, likelihood, kernel_name)
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # 학습 루프
        for epoch in range(50):  # 50번 반복 학습
            optimizer.zero_grad()
            output = model(train_x_torch)
            loss = -mll(output, train_y_torch)
            loss.backward()
            optimizer.step()

            # 손실 값 출력
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

        # 예측 수행
        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = model(test_x_torch)
            mean_prediction = pred.mean.numpy()
            std_prediction = pred.variance.sqrt().numpy()

        # 결과 저장
        results[kernel_name] = {"mean": mean_prediction, "std": std_prediction}

    return results

# 6. 모든 모델 학습 및 결과 저장
results = {
    "Scikit-Learn": train_and_predict_sklearn(),
    "GPy": train_and_predict_gpy(),
    "GPflow": train_and_predict_gpflow(),
    "GPyTorch": train_and_predict_gpytorch()}

# 7. 시각화
fig, axes = plt.subplots(len(results), len(["RBF", "Matern(1.5)", "Matern(2.5)", "Rational"]))
# 결과를 순회하며 플롯
for i, (model_name, model_results) in enumerate(results.items()):
    for j, (kernel_name, kernel_results) in enumerate(model_results.items()):
        ax = axes[i, j]

        # 예측 결과
        pred_mean = kernel_results["mean"]
        pred_std = kernel_results["std"]

        # 95% 신뢰 구간 계산
        lower_bound = pred_mean - 1.96 * pred_std
        upper_bound = pred_mean + 1.96 * pred_std

        # 실제 값
        actual_values = test_y[:len(pred_mean)]  # 길이를 예측 값에 맞춤

        # 플롯
        ax.plot(cycle, actual_values, 'r-', label='Actual Value')  # 실제 값
        ax.plot(cycle, pred_mean, 'b-', label='Predicted Mean')  # 예측 평균
        ax.fill_between(cycle, lower_bound, upper_bound, color='blue', alpha=0.2, label='95% Confidence Interval')  # 신뢰 구간

        # 제목 및 레이블 설정
        if i == 0:
            ax.set_title(kernel_name, fontsize=12)
        if j == 0:
            ax.set_ylabel(model_name, fontsize=12)
        if i == len(results) - 1:
            ax.set_xlabel("Cycle", fontsize=10)

        # 첫 번째 서브플롯에만 범례 추가
        if i == 0 and j == 0:
            ax.legend(loc = "lower left", fontsize=8)
# 레이아웃 조정 및 출력
plt.tight_layout()



# 2. Paper에 맞게 setting
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. 데이터 준비
train_x = train_x_raw
train_y = train_y_raw
test_x = test_x_raw
test_y = test_y_raw
cycle = Broadband_cases["test"][B_test_list[-1]]["Cycle"].values

# 입력 데이터 스케일링 함수
def scale_input(train_x, test_x):
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)
    return train_x_scaled, test_x_scaled, scaler


# Scikit-learn GPR 함수
def sklearn_gpr(train_x, train_y, test_x, test_y):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel()
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.0)
    model.fit(train_x, train_y)
    y_pred, y_std = model.predict(test_x, return_std=True)
    mse = mean_squared_error(test_y, y_pred)
    return y_pred, mse, y_std


# GPy GPR 함수
def gpy_gpr(train_x, train_y, test_x, test_y):
    import GPy
    kernel = GPy.kern.Matern32(input_dim=train_x.shape[1])
    model = GPy.models.GPRegression(train_x, train_y.reshape(-1, 1), kernel)
    model.optimize(messages=False)
    mean, variance = model.predict(test_x)
    mse = mean_squared_error(test_y, mean.ravel())
    return mean.ravel(), mse, np.sqrt(variance.ravel())


# GPflow GPR 함수
def gpflow_gpr(train_x, train_y, test_x, test_y):
    import gpflow
    kernel = gpflow.kernels.Matern32()
    model = gpflow.models.GPR(data=(train_x, train_y.reshape(-1, 1)), kernel=kernel)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)
    mean, variance = model.predict_f(test_x)
    mse = mean_squared_error(test_y, mean.numpy().ravel())
    return mean.numpy().ravel(), mse, np.sqrt(variance.numpy().ravel())


# GPyTorch GPR 함수
def gpytorch_gpr(train_x, train_y, test_x, test_y):
    import torch
    import gpytorch

    train_x_torch = torch.tensor(train_x, dtype=torch.float32)
    train_y_torch = torch.tensor(train_y, dtype=torch.float32)
    test_x_torch = torch.tensor(test_x, dtype=torch.float32)

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=1.5)
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x_torch, train_y_torch, likelihood)
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(100):
        optimizer.zero_grad()
        output = model(train_x_torch)
        loss = -mll(output, train_y_torch)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(test_x_torch))
        mean = pred.mean
        lower, upper = pred.confidence_region()
    mse = mean_squared_error(test_y, mean.numpy().ravel())
    return mean.numpy().ravel(), mse, (upper - lower).numpy().ravel() / 2


# 전체 비교 함수
def compare_models(train_x, train_y, test_x, test_y):
    # 입력 데이터만 스케일링
    train_x_scaled, test_x_scaled, scaler = scale_input(train_x, test_x)

    # 각 모델 성능 평가
    models = {
        "Scikit-learn": sklearn_gpr,
        "GPy": gpy_gpr,
        "GPflow": gpflow_gpr,
        "GPyTorch": gpytorch_gpr,
    }
    results = {}
    for name, model_func in models.items():
        y_pred, mse, uncertainty = model_func(train_x_scaled, train_y, test_x_scaled, test_y)
        results[name] = {"MSE": mse, "Prediction": y_pred, "Uncertainty": uncertainty}
    return results
results_compare_models = compare_models(train_x, train_y, test_x, test_y)


# 서브플롯 시각화
fig, axes = plt.subplots(2, 2)
axes = axes.flatten()
# 각 모델의 결과를 서브플롯에 시각화
for idx, (model_name, result) in enumerate(results_compare_models.items()):
    y_pred = result["Prediction"]
    uncertainty = result["Uncertainty"]
    ax = axes[idx]
    # 실제값
    ax.plot(cycle, test_y, 'k-', label="True Value", linewidth=1.5)
    # 예측값
    ax.plot(cycle, y_pred, 'r-', label="Predicted Value", linewidth=1.5)
    # 95% 신뢰구간
    ax.fill_between(cycle, y_pred - 1.96 * uncertainty, y_pred + 1.96 * uncertainty, color='gray', alpha=0.3, label="95% Confidence Interval")
    # 서브플롯 제목 및 설정
    ax.set_title(f"{model_name} Model")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SoH(%)")
    ax.legend()
    ax.grid(True)
# 전체 플롯 레이아웃 정리
plt.tight_layout()


