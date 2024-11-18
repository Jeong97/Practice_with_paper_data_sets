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
import gpytorch
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


''' BroadBand '''
# SK-learn method
def train_and_predict_SK_Bayesian(train_x, train_y, test_x, test_y):
    # 스케일링 추가
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)

    # Bayesian Optimization
    def objective(trial):
        alpha = trial.suggest_loguniform("alpha", 1e-5, 1e-1)
        length_scale = trial.suggest_uniform("length_scale", 0.1, 10.0)
        constant_value = trial.suggest_uniform("constant_value", 0.1, 10.0)
        noise_level = trial.suggest_loguniform("noise_level", 1e-6, 1e-2)
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

# GPy Methode with Extended Optimization
def train_and_predict_GPy_Bayesian(train_x, train_y, test_x, test_y):
    # 스케일링 추가
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)

    # Bayesian Optimization
    def objective(trial):
        alpha = trial.suggest_loguniform("alpha", 1e-5, 1e-1)
        length_scale = trial.suggest_uniform("length_scale", 0.1, 10.0)
        variance = trial.suggest_uniform("variance", 0.1, 10.0)  # 추가: 커널 스케일
        noise_level = trial.suggest_loguniform("noise_level", 1e-6, 1e-2)  # 추가: 노이즈 수준
        nu = trial.suggest_categorical("nu", [1.5, 2.5])  # 추가: nu 최적화

        # 확장된 커널 조합
        kernel = GPy.kern.Matern32(input_dim=train_x.shape[1], lengthscale=length_scale, variance=variance) + GPy.kern.White(input_dim=train_x.shape[1], variance=noise_level)
        model = GPy.models.GPRegression(train_x_scaled, train_y.reshape(-1, 1), kernel)
        model.Gaussian_noise.variance = alpha  # 노이즈 최적화
        model.optimize(messages=False)
        y_pred, y_var = model.predict(test_x_scaled)
        mse = mean_squared_error(test_y, y_pred.ravel())
        return mse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    # 최적화된 GPR 모델
    best_params = study.best_params
    kernel = (
        GPy.kern.Matern32(input_dim=train_x.shape[1], lengthscale=best_params["length_scale"], variance=best_params["variance"])
        + GPy.kern.White(input_dim=train_x.shape[1], variance=best_params["noise_level"])
    )
    model = GPy.models.GPRegression(train_x_scaled, train_y.reshape(-1, 1), kernel)
    model.Gaussian_noise.variance = best_params["alpha"]
    model.optimize(messages=False)
    y_pred, y_var = model.predict(test_x_scaled)
    y_std = np.sqrt(y_var.ravel())

    lower_bound = y_pred.ravel() - 1.96 * y_std
    upper_bound = y_pred.ravel() + 1.96 * y_std

    return y_pred.ravel(), y_std, lower_bound, upper_bound, best_params


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
performance_metrics = {"SK-learn": [], "GPy": []}
predicted_results = {"SK-learn": [], "GPy": []}

for i in range(len(B_train_list)):
    # 데이터 분리
    train_x_raw = Broadband_cases["train"][B_train_list[i]].drop(columns=B_input_drop_col).values
    train_y_raw = Broadband_cases["train"][B_train_list[i]]["SoH"].values
    test_x_raw = Broadband_cases["test"][B_test_list[i]].drop(columns=B_input_drop_col).values
    test_y_raw = Broadband_cases["test"][B_test_list[i]]["SoH"].values

    # SK-learn 방식 예측
    y_pred_sk, y_std_sk, lower_sk, upper_sk, best_params_sk = train_and_predict_SK_Bayesian(train_x_raw, train_y_raw, test_x_raw, test_y_raw)
    metrics_sk = calculate_performance_metrics(test_y_raw, y_pred_sk, y_std_sk)
    metrics_sk["Method"] = "SK-learn"
    metrics_sk["Case"] = f"Cell {i + 1}"
    performance_metrics["SK-learn"].append(metrics_sk)
    predicted_results["SK-learn"].append({
        "Actual": test_y_raw.tolist(),
        "Predicted": y_pred_sk.tolist(),
        "Lower Bound": lower_sk.tolist(),
        "Upper Bound": upper_sk.tolist(),
        "Case": f"Cell {i + 1}",
    })

    # GPy 방식 예측
    y_pred_gpy, y_std_gpy, lower_gpy, upper_gpy, best_params_gpy = train_and_predict_GPy_Bayesian(train_x_raw, train_y_raw, test_x_raw, test_y_raw)
    metrics_gpy = calculate_performance_metrics(test_y_raw, y_pred_gpy, y_std_gpy)
    metrics_gpy["Method"] = "GPy"
    metrics_gpy["Case"] = f"Cell {i + 1}"
    performance_metrics["GPy"].append(metrics_gpy)
    predicted_results["GPy"].append({
        "Actual": test_y_raw.tolist(),
        "Predicted": y_pred_gpy.tolist(),
        "Lower Bound": lower_gpy.tolist(),
        "Upper Bound": upper_gpy.tolist()})

# 출력 결과를 데이터프레임으로 저장
B_predicted_results_df = {method: pd.DataFrame(predicted_results[method]) for method in ["SK-learn", "GPy"]}

# 두 결과를 하나의 데이터프레임으로 결합
B_performance_df = pd.concat([pd.DataFrame(performance_metrics["SK-learn"]), pd.DataFrame(performance_metrics["GPy"])])
B_performance_df[B_performance_df["Method"]=="GPy"]


# Calculate Cycle, Actual SoH, Predicted SoH, and confidence intervals for each method
colors = sns.color_palette("Spectral", 3)
actual_color = '#E74C3C'
predicted_color = '#2E86C1'
confidence_color = '#808080'

# Methods to loop over
methods = ["SK-learn", "GPy"]

for method in methods:
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
methods = ["SK-learn", "GPy"]
for method in methods:
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
colors = sns.color_palette("Set1", len(methods))

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
    axes[row, col].legend(title="Method", fontsize=8, loc="best")
# Hide empty subplot
axes[1, 2].axis("off")
# Set overall layout
plt.suptitle("Broadband Performance Metrics Comparison by Metric and Method", fontsize=16, fontweight='bold')
plt.tight_layout()



''' Fixed_Frequency '''
# Create a data frame to apply to ML Model
def train_and_predict_gpr_Fixed(train_x, train_y, test_x):
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)
    kernel = C(1.0, (1e-3, 1e4)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, alpha=0.1)
    gpr.fit(train_x_scaled, train_y)
    y_pred, y_std = gpr.predict(test_x_scaled, return_std=True)
    return y_pred, y_std


### Study_1 : Broadband Version
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
F_performance_metrics = {"SK-learn": [], "GPy": []}
F_predicted_results = {"SK-learn": [], "GPy": []}

# 각 fixed_case 반복
for num in range(len(fixed_cases_list)):
    # Train/Test 데이터 분리
    F_train_list = list(fixed_hold_out[num]["train"].keys())
    F_test_list = list(fixed_hold_out[num]["test"].keys())

    for method in ["SK-learn", "GPy"]:
        for i in range(len(F_train_list)):
            # 훈련 및 테스트 데이터 준비
            train_x_raw = fixed_hold_out[num]["train"][F_train_list[i]].drop(
                columns=["Cell_Name", "Cycle", "SoH"]).values
            train_y_raw = fixed_hold_out[num]["train"][F_train_list[i]]["SoH"].values
            test_x_raw = fixed_hold_out[num]["test"][F_test_list[i]].drop(columns=["Cell_Name", "Cycle", "SoH"]).values
            test_y_raw = fixed_hold_out[num]["test"][F_test_list[i]]["SoH"].values

            # 모델 학습 및 예측
            if method == "SK-learn":
                y_pred, y_std, lower_bound, upper_bound, best_params = train_and_predict_SK_Bayesian(
                    train_x_raw, train_y_raw, test_x_raw, test_y_raw
                )
            else:  # GPy 방식
                y_pred, y_std, lower_bound, upper_bound, best_params = train_and_predict_GPy_Bayesian(
                    train_x_raw, train_y_raw, test_x_raw, test_y_raw
                )

            # 성능 지표 계산
            metrics = calculate_performance_metrics(test_y_raw, y_pred, y_std)
            metrics["Method"] = method
            metrics["Case"] = f"Fixed {num + 1}"
            F_performance_metrics[method].append(metrics)

            # 예측 결과 저장
            F_predicted_results[method].append({
                "Case": f"Fixed {num + 1}",
                "Actual": test_y_raw.tolist(),
                "Predicted": y_pred.tolist(),
                "Lower Bound": lower_bound.tolist(),
                "Upper Bound": upper_bound.tolist(),})

# 성능 지표 데이터프레임으로 변환
F_performance_df = pd.concat([pd.DataFrame(F_performance_metrics["SK-learn"]),
                              pd.DataFrame(F_performance_metrics["GPy"])])

# 예측 결과 데이터프레임으로 변환
F_predicted_results_df = {method: pd.DataFrame(F_predicted_results[method]) for method in ["SK-learn", "GPy"]}


fixed_mean_result_df = F_performance_df[(F_performance_df["Method"]=="GPy")&(F_performance_df["Case"].str.contains("Fixed 4"))]



# Iterate over Fixed Cases
fixed_plot_cases = ["Fixed 1", "Fixed 2", "Fixed 3", "Fixed 4"]
methods = ["SK-learn", "GPy"]

# Seaborn 스타일 설정
sns.set(style="darkgrid")

# 색상 리스트 정의 (각 데이터 요소에 대해 일관된 색상 사용)
colors = sns.color_palette("Spectral", 3)
actual_color = '#E74C3C'  # 실제값 색상 (눈에 편안한 주황색 계열)
predicted_color = '#2E86C1'  # 예측값 색상 (진한 파란색 계열)
confidence_color = '#808080'  # 신뢰 구간 색상 (연한 회색 계열)


for f_case in ["Fixed 1", "Fixed 2", "Fixed 3", "Fixed 4"]:
    method = methods[0]
    # Filter results for the specific Fixed Case and method
    fixed_plot_df = F_predicted_results_df[methods[0]][F_predicted_results_df[methods[0]]["Case"].str.contains(f_case)]

    # Create a figure for the Fixed Case
    plt.figure(figsize=(14, 10))

    # Iterate over the rows in the filtered dataframe (one subplot per Cell)
    for n in range(len(fixed_plot_df)):
        plt.subplot(2, 2, n + 1)
        x = fixed_hold_out[n]["test"][F_test_list[n]]["Cycle"].values
        actual_soh = fixed_plot_df["Actual"].reset_index(drop=True)[n]
        predicted_soh = fixed_plot_df["Predicted"].reset_index(drop=True)[n]
        lower_bound = fixed_plot_df["Lower Bound"].reset_index(drop=True)[n]
        upper_bound = fixed_plot_df["Upper Bound"].reset_index(drop=True)[n]

        # Plot Actual SoH
        plt.plot(x, actual_soh, '*', color=actual_color, markersize=7, label='Actual SoH')

        # Plot Predicted SoH
        plt.plot(x, predicted_soh, '-', color=predicted_color, linewidth=3, label='Predicted SoH')

        # Plot Confidence Interval
        plt.fill_between(x, lower_bound, upper_bound, color=confidence_color, alpha=0.3,
                         label='95% Confidence Interval')

        # Add subplot title and labels
        plt.xlabel("Cycle", fontsize=12, fontweight='bold')
        plt.ylabel("SoH (%)", fontsize=12, fontweight='bold')
        plt.title(f"{method} - {f_case}", fontsize=14, fontweight='bold')
        plt.legend(loc="upper right", fontsize="small")

    # Set the overall title for the figure
    plt.suptitle(f"{method} - {f_case} SoH Predictions", fontsize=16, fontweight='bold')
    plt.tight_layout()


# 성능 비교를 위한 지표
metrics = ["Maximum Absolute Error (%)", "Mean Absolute Error (%)",
           "Root Mean Square Error (%)", "Coverage Probability (%)",
           "Mean Standard Deviation (%)"]

# 같은 모델 내에서 비교할 방법 (예: "SK-learn")
method = "SK-learn"

# 2x3 서브플롯 생성 (마지막 subplot은 비어 있게 유지)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# 각 성능 지표에 대해 반복
for i, metric in enumerate(metrics):
    row, col = divmod(i, 3)
    ax = axes[row, col]

    # 지표에 따라 Fixed 모델 비교 데이터 생성
    comparison_data = F_performance_df[(F_performance_df["Method"] == method)]
    metric_data = comparison_data[["Cell Name", metric]]

    # 막대 그래프 그리기
    sns.barplot(
        x="Cell Name",
        y=metric,
        data=metric_data,
        ax=ax,
        palette=sns.color_palette("Set2", len(comparison_data["Cell Name"].unique())),
        edgecolor='black',
        hue="Cell Name",  # `hue`를 추가하여 경고 제거
        dodge=False  # hue로 인해 발생하는 막대 간 위치 이동 방지
    )

    # 서브플롯 제목 및 레이블 설정
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_ylabel(metric, fontsize=10, fontweight='bold')
    ax.set_xlabel("Cell Name", fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)

    # 범례가 있을 경우 제거
    legend = ax.get_legend()
    if legend is not None:  # 범례 존재 여부 확인
        legend.remove()

# 빈 서브플롯 숨기기
if len(metrics) < 6:
    for j in range(len(metrics), 6):
        row, col = divmod(j, 3)
        axes[row, col].axis("off")


# 전체 제목 설정
plt.suptitle(f"Comparison of Performance Metrics for {method}", fontsize=16, fontweight='bold')
plt.tight_layout()

# 비교하고자 하는 특정 Cell Name
cell_name = "Cell 1"

# Cell Name에 따라 데이터 필터링
cell_data = F_performance_df[F_performance_df["Cell Name"] == cell_name]

# 2x3 서브플롯 생성 (마지막 subplot은 비어 있게 유지)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# 각 성능 지표에 대해 반복
for i, metric in enumerate(metrics):
    row, col = divmod(i, 3)
    ax = axes[row, col]

    # 지표 데이터 생성 (Case를 x축으로 사용)
    metric_data = cell_data[["Case", metric]]

    # 막대 그래프 그리기
    sns.barplot(
        x="Case",
        y=metric,
        data=metric_data,
        ax=ax,
        palette=sns.color_palette("Set2", len(metric_data["Case"].unique())),
        edgecolor='black'
    )

    # 서브플롯 제목 및 레이블 설정
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_ylabel(metric, fontsize=10, fontweight='bold')
    ax.set_xlabel("Fixed Models", fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)

# 빈 서브플롯 숨기기
if len(metrics) < 6:
    for j in range(len(metrics), 6):
        row, col = divmod(j, 3)
        axes[row, col].axis("off")

# 전체 제목 설정
plt.suptitle(f"Performance Metrics Comparison for {cell_name}", fontsize=16, fontweight='bold')
plt.tight_layout()







# 데이터셋 분리 함수
def separate_cases(df, cell_name):
    train_df = df[df['Cell_Name'] != cell_name].copy().reset_index(drop=True)
    test_df = df[df['Cell_Name'] == cell_name].copy().reset_index(drop=True)
    return train_df, test_df


# 성능 지표 계산 함수
def calculate_performance_metrics(y_true, y_pred, y_std):
    max_absolute_error = np.max(np.abs(y_true - y_pred) / np.abs(y_true)) * 100
    mean_absolute_error_perc = mean_absolute_error(y_true, y_pred) / np.mean(np.abs(y_true)) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) / np.mean(np.abs(y_true)) * 100
    lower_bound = y_pred - 1.96 * y_std
    upper_bound = y_pred + 1.96 * y_std
    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound)) * 100
    mean_std_dev = np.mean(y_std) / np.mean(np.abs(y_true)) * 100

    return {
        'Maximum Absolute Error (%)': max_absolute_error,
        'Mean Absolute Error (%)': mean_absolute_error_perc,
        'Root Mean Square Error (%)': rmse,
        'Coverage Probability (%)': coverage,
        'Mean Standard Deviation (%)': mean_std_dev
    }


# GPR 모델 정의
def create_gpr_pipeline():
    kernel = (
            C(1.0, (1e-3, 1e4)) *
            Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) +
            WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # 스케일링
        ('gpr', GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, alpha=0.1))
    ])
    return pipeline


# 데이터셋 준비
fixed_hold_out = {num: {"train": {}, "test": {}} for num in range(len(fixed_case_df))}
for num, case_key in enumerate(fixed_case_df.keys()):
    for i in range(1, 5):
        train_df, test_df = separate_cases(fixed_case_df[case_key], f'Cell_{i}')
        fixed_hold_out[num]["train"][f'case{i}_train_df'] = train_df
        fixed_hold_out[num]["test"][f'case{i}_test_df'] = test_df

# 성능 지표 저장 및 모델 학습
F_performance_metrics = []

for num in fixed_hold_out.keys():
    F_train_list = list(fixed_hold_out[num]["train"].keys())
    F_test_list = list(fixed_hold_out[num]["test"].keys())

    for i in range(len(F_train_list)):
        train_x_raw = fixed_hold_out[num]["train"][F_train_list[i]].drop(columns=["Cell_Name", "Cycle"]).values
        train_y_raw = fixed_hold_out[num]["train"][F_train_list[i]]["SoH"].values
        test_x_raw = fixed_hold_out[num]["test"][F_test_list[i]].drop(columns=["Cell_Name", "Cycle"]).values
        test_y_raw = fixed_hold_out[num]["test"][F_test_list[i]]["SoH"].values

        # Pipeline으로 GPR 모델 학습 및 예측
        pipeline = create_gpr_pipeline()
        pipeline.fit(train_x_raw, train_y_raw)
        y_pred, y_std = pipeline.predict(test_x_raw, return_std=True)

        # 성능 지표 계산
        metrics = calculate_performance_metrics(test_y_raw, y_pred, y_std)
        F_performance_metrics.append(metrics)

        from sklearn.model_selection import cross_val_score
        kernel = C(1.0, (1e-3, 1e4)) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1e-3)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, alpha=1)
        scores = cross_val_score(gpr, train_x_raw, train_y_raw, cv=5, scoring="neg_mean_squared_error")
        print("Cross-Validation Scores:", -scores)

# 성능 결과를 데이터프레임으로 변환
F_performance_df = pd.DataFrame(F_performance_metrics)











train_x_raw.shape



''''''

train_x_raw
train_y_raw
test_x_raw
test_y_raw


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
import optuna

input_demention = 6

# Standardize training and testing datasets
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x_raw)
test_x = scaler.transform(test_x_raw)

# Kernels for comparison
kernels_sklearn = {
    "RBF": RBF(length_scale=1.0),
    "Matern (ν=1.5)": Matern(length_scale=1.0, nu=1.5),
    "Matern (ν=2.5)": Matern(length_scale=1.0, nu=2.5),
    "Rational Quadratic": RationalQuadratic(length_scale=1.0, alpha=1.0),}

# Results storage
results = {
    "Scikit-learn": {},
    "GPy": {},
    "Pyro": {},
    "GPyTorch" : {}
}

# Step 2: Scikit-learn Implementation
for kernel_name, kernel in kernels_sklearn.items():
    try:
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1, random_state=42)
        gpr.fit(train_x, train_y_raw)
        y_pred, y_std = gpr.predict(test_x, return_std=True)
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
            kernel_gpy = GPy.kern.Matern32(input_dim=input_demention) if nu == 1.5 else GPy.kern.Matern52(input_dim=input_demention)
        elif kernel_name == "RBF":
            kernel_gpy = GPy.kern.RBF(input_dim=input_demention)
        elif kernel_name == "Rational Quadratic":
            kernel_gpy = GPy.kern.RatQuad(input_dim=input_demention)
        elif kernel_name == "Periodic":
            kernel_gpy = GPy.kern.PeriodicExponential(input_dim=input_demention)
        else:
            continue

        model = GPy.models.GPRegression(train_x, train_y_raw.reshape(-1, 1), kernel_gpy)
        model.optimize(messages=False)
        y_pred, y_var = model.predict(test_x)
        results["GPy"][kernel_name] = {
            "model": model,
            "y_pred": y_pred.ravel(),
            "y_std": np.sqrt(y_var.ravel()),
            "log_marginal_likelihood": model.log_likelihood()
        }
    except Exception as e:
        print(f"GPy - {kernel_name}: Error - {e}")

# Step 4: Pyro Implementation
train_x_torch = torch.tensor(train_x, dtype=torch.float32)
train_y_torch = torch.tensor(train_y_raw, dtype=torch.float32)
test_x_torch = torch.tensor(test_x, dtype=torch.float32)

for kernel_name, kernel in kernels_sklearn.items():
    try:
        if kernel_name == "RBF":
            kernel_pyro = gp.kernels.RBF(input_dim=input_demention)
        elif kernel_name.startswith("Matern"):
            kernel_pyro = gp.kernels.Matern32(input_dim=input_demention) if "1.5" in kernel_name else gp.kernels.Matern52(input_dim=input_demention)
        elif kernel_name == "Rational Quadratic":
            kernel_pyro = gp.kernels.RationalQuadratic(input_dim=input_demention)
        elif kernel_name == "Periodic":
            kernel_pyro = gp.kernels.Periodic(input_dim=input_demention)
        else:
            continue

        # Create GPRegression model
        gpr_pyro = gp.models.GPRegression(train_x_torch, train_y_torch, kernel_pyro, noise=torch.tensor(0.1))

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
        mean, var = gpr_pyro(test_x_torch, full_cov=False, noiseless=False)
        results["Pyro"][kernel_name] = {
            "model": gpr_pyro,
            "y_pred": mean.detach().numpy(),
            "y_std": torch.sqrt(var).detach().numpy(),
            "log_marginal_likelihood": -loss.item()
        }
        print(f"Pyro - {kernel_name}: Training complete, Log Marginal Likelihood = {-loss.item():.3f}")

    except Exception as e:
        print(f"Pyro - {kernel_name}: Error - {e}")


# Step 5: GPyTorch Implementation
# GPyTorch Implementation

# GPyTorch 커널 매핑 함수
def get_gpytorch_kernel(kernel_name):
    if kernel_name == "RBF":
        return gpytorch.kernels.RBFKernel()
    elif kernel_name == "Matern (ν=1.5)":
        return gpytorch.kernels.MaternKernel(nu=1.5)
    elif kernel_name == "Matern (ν=2.5)":
        return gpytorch.kernels.MaternKernel(nu=2.5)
    elif kernel_name == "Rational Quadratic":
        # Rational Quadratic 대체 (RBF + ScaleKernel 조합)
        return gpytorch.kernels.ScaleKernel(base_kernel=gpytorch.kernels.RBFKernel())
    else:
        raise ValueError(f"Unsupported kernel: {kernel_name}")

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = None  # 커널은 이후에 설정

    def set_kernel(self, kernel):
        self.covar_module = kernel  # base_kernel 전달 완료

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# 데이터 변환
train_x_torch = torch.tensor(train_x, dtype=torch.float32)
train_y_torch = torch.tensor(train_y_raw, dtype=torch.float32)
test_x_torch = torch.tensor(test_x, dtype=torch.float32)

# GPyTorch 학습 및 예측
for kernel_name, kernel in kernels_sklearn.items():
    try:
        # GPyTorch 커널 생성
        gpytorch_kernel = get_gpytorch_kernel(kernel_name)

        # 모델 및 likelihood 초기화
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x_torch, train_y_torch, likelihood)
        model.set_kernel(gpytorch_kernel)  # 커널 설정

        # 학습
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        training_iterations = 50
        for i in range(training_iterations):
            optimizer.zero_grad()
            output = model(train_x_torch)
            loss = -mll(output, train_y_torch)
            loss.backward()
            optimizer.step()

        # 예측
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = likelihood(model(test_x_torch))
            mean = predictions.mean
            lower, upper = predictions.confidence_region()

        # 결과 저장
        results["GPyTorch"][kernel_name] = {
            "model": model,
            "y_pred": mean.numpy(),
            "y_std": (upper - lower).numpy() / 2,
            "log_marginal_likelihood": -loss.item()
        }
        print(f"GPyTorch - {kernel_name}: Training complete, Log Marginal Likelihood = {-loss.item():.3f}")

    except Exception as e:
        print(f"GPyTorch - {kernel_name}: Error - {e}")



# Step 5: Visualization
plt.figure(figsize=(20, 10))
num_kernels = len(kernels_sklearn)
num_models = len(results)

for i, (model_name, model_results) in enumerate(results.items(), 1):
    for j, (kernel_name, result) in enumerate(model_results.items(), 1):
        plt.subplot(num_models, num_kernels, (i - 1) * num_kernels + j)
        y_pred = result["y_pred"]
        y_std = result["y_std"]
        plt.plot(test_y_raw, label="True Values", color='black', alpha=0.6)
        plt.plot(y_pred, label=f"{kernel_name} Predictions", alpha=0.8)
        plt.fill_between(
            range(len(test_y_raw)),
            y_pred - 1.96 * y_std,
            y_pred + 1.96 * y_std,
            color='blue', alpha=0.2, label="95% Confidence Interval"
        )
        plt.title(f"{model_name}: {kernel_name}")
        plt.legend(loc="upper right", fontsize=10)
plt.tight_layout()


# 성능 지표 계산 함수
def calculate_metrics(y_true, y_pred, y_std, confidence_level=1.96):
    """
    성능 지표 계산:
    - y_true: 실제값
    - y_pred: 예측값
    - y_std: 예측 표준편차
    - confidence_level: 신뢰구간 (default=95%, z=1.96)
    """
    # 절대 오차
    absolute_errors = np.abs(y_true - y_pred)
    max_abs_error = np.max(absolute_errors)
    mean_abs_error = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Coverage Probability 계산
    lower_bound = y_pred - confidence_level * y_std
    upper_bound = y_pred + confidence_level * y_std
    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound)) * 100

    # 평균 표준편차 계산
    mean_std = np.mean(y_std)

    # 결과 반환
    return {
        "Max Absolute Error (%)": (max_abs_error / np.mean(y_true)) * 100,
        "Mean Absolute Error (%)": (mean_abs_error / np.mean(y_true)) * 100,
        "Root Mean Square Error (%)": (rmse / np.mean(y_true)) * 100,
        "Coverage Probability (%)": coverage,
        "Mean Standard Deviation (%)": (mean_std / np.mean(y_true)) * 100,
    }

# 성능 지표 추가
metrics_results = {}

for model_name, model_results in results.items():
    metrics_results[model_name] = {}
    for kernel_name, result in model_results.items():
        try:
            # 예측값과 표준편차 가져오기
            y_pred = result["y_pred"]
            y_std = result["y_std"]

            # 성능 지표 계산
            metrics = calculate_metrics(test_y_raw, y_pred, y_std)
            metrics_results[model_name][kernel_name] = metrics

            # 출력 결과
            print(f"{model_name} - {kernel_name} Metrics:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.2f}")
        except Exception as e:
            print(f"Error calculating metrics for {model_name} - {kernel_name}: {e}")


# Step 6: Log Marginal Likelihood Comparison
print("\nLog-Marginal Likelihood Comparison:")
for model_name, model_results in results.items():
    print(f"\n{model_name}:")
    for kernel_name, result in model_results.items():
        print(f"  {kernel_name}: {result['log_marginal_likelihood']:.3f}")




# Optimization of hyperparameter
# Hyperparameter Optimization with Optuna
def optimize_hyperparameters(train_x, train_y_raw, test_x, test_y_raw):
    def objective(trial):
        alpha = trial.suggest_loguniform("alpha", 1e-5, 1e-1)
        length_scale = trial.suggest_uniform("length_scale", 0.1, 10.0)
        nu = trial.suggest_categorical("nu", [1.5])

        kernel = Matern(length_scale=length_scale, nu=nu)
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha, random_state=42)
        gpr.fit(train_x, train_y_raw)
        y_pred, y_std = gpr.predict(test_x, return_std=True)

        mse = mean_squared_error(test_y_raw, y_pred)
        return mse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    return study.best_params

# Unified GPR Method
def train_and_predict(train_x, train_y, test_x, method, best_params):
    alpha = best_params["alpha"]
    length_scale = best_params["length_scale"]
    nu = best_params["nu"]

    # Scikit-learn
    if method == "scikit-learn":
        kernel = C(1.0, (1e-3, 1e4)) * Matern(length_scale=length_scale, nu=nu) + WhiteKernel(noise_level=1e-3)
        model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, random_state=42)
        model.fit(train_x, train_y)
        y_pred, y_std = model.predict(test_x, return_std=True)
        return y_pred, y_std

    # GPy
    elif method == "gpy":
        kernel = GPy.kern.Matern32(input_dim=train_x.shape[1], lengthscale=length_scale) if nu == 1.5 else GPy.kern.Matern52(input_dim=train_x.shape[1], lengthscale=length_scale)
        model = GPy.models.GPRegression(train_x, train_y[:, None], kernel)
        model.Gaussian_noise.variance = alpha
        model.optimize(messages=False)
        y_pred, y_var = model.predict(test_x)
        return y_pred.ravel(), np.sqrt(y_var.ravel())

    # Pyro
    elif method == "pyro":
        train_x_torch = torch.tensor(train_x, dtype=torch.float32)
        train_y_torch = torch.tensor(train_y, dtype=torch.float32)
        test_x_torch = torch.tensor(test_x, dtype=torch.float32)
        kernel = gp.kernels.Matern32(input_dim=train_x.shape[1]) if nu == 1.5 else gp.kernels.Matern52(input_dim=train_x.shape[1])
        model = gp.models.GPRegression(train_x_torch, train_y_torch, kernel, noise=torch.tensor(alpha))
        optimizer = pyro.optim.Adam({"lr": 0.01})
        # loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        loss_fn = pyro.infer.Trace_ELBO()
        svi = pyro.infer.SVI(model.model, model.guide, optimizer, loss=loss_fn)
        for _ in range(100):  # Training iterations
            # optimizer.zero_grad()
            loss = svi.step()  # SVI에서 step 메서드를 사용하여 손실 계산 및 최적화 수행
            # loss = loss_fn(model.model, model.guide)
            # loss.backward()
            # optimizer.step()
        mean, var = model(test_x_torch, full_cov=False, noiseless=False)
        return mean.detach().numpy(), torch.sqrt(var).detach().numpy()

    # GPyTorch
    elif method == "gpytorch":
        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=nu))
            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        train_x_torch = torch.tensor(train_x, dtype=torch.float32)
        train_y_torch = torch.tensor(train_y, dtype=torch.float32)
        test_x_torch = torch.tensor(test_x, dtype=torch.float32)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x_torch, train_y_torch, likelihood)
        model.covar_module.base_kernel.lengthscale = length_scale
        likelihood.noise = alpha
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for _ in range(50):  # Training iterations
            optimizer.zero_grad()
            output = model(train_x_torch)
            loss = -mll(output, train_y_torch)
            loss.backward()
            optimizer.step()
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = likelihood(model(test_x_torch))
            mean = predictions.mean
            lower, upper = predictions.confidence_region()
        return mean.numpy(), (upper - lower).numpy() / 2

# Example usage
# best_params = optimize_hyperparameters()
# for method in ["scikit-learn", "gpy", "pyro", "gpytorch"]:
#     y_pred, y_std = train_and_predict(train_x, train_y, test_x, method, best_params)
#     print(f"Method: {method}, Predicted: {y_pred}, Std Dev: {y_std}")


# Initialize the list to store performance metrics for each method
methods = ["scikit-learn", "gpy", "pyro", "gpytorch"]
performance_metrics = {method: [] for method in methods}

# Initialize prediction results storage
all_actuals = {method: [] for method in methods}
all_predictions = {method: [] for method in methods}
all_std_devs = {method: [] for method in methods}

# Perform model training and prediction for each method and each dataset
# best_params = optimize_hyperparameters()  # Get optimized hyperparameters using Bayesian optimization

for method in methods:
    for i in range(len(B_train_list)):
        train_x_raw = Broadband_cases["train"][B_train_list[i]].drop(columns=B_input_drop_col).values
        train_y_raw = Broadband_cases["train"][B_train_list[i]]["SoH"].values
        test_x_raw = Broadband_cases["test"][B_test_list[i]].drop(columns=B_input_drop_col).values
        test_y_raw = Broadband_cases["test"][B_test_list[i]]["SoH"].values

        # 하이퍼파라미터 최적화
        best_params = optimize_hyperparameters(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

        # 4가지 방법으로 예측 수행
        for method in ["scikit-learn", "gpy", "pyro", "gpytorch"]:
            y_pred, y_std = train_and_predict(train_x_raw, train_y_raw, test_x_raw, method, best_params)
            print(f"Method: {method}, Predicted: {y_pred}, Std Dev: {y_std}")

# Convert performance metrics to DataFrames for each method
performance_dfs = {method: pd.DataFrame(performance_metrics[method]) for method in methods}

# Display performance metrics for all methods
for method, df in performance_dfs.items():
    print(f"Performance Metrics for {method}:")
    print(df)


