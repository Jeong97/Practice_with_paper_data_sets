'''Data PreProcessing'''
# Import Lybrary
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
# from mat4py import loadmat
import warnings
warnings.simplefilter("ignore")


# Graph Font setting
def setGraphFont():
    import matplotlib.font_manager as fm

    # 설치된 폰트 출력
    font_list = [font.name for font in fm.fontManager.ttflist]

    # default font 설정
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


# Split Cycle by Current
def Split_Cycle(Current):
    n = len(Current)  # assign n to Current index by
    Idx = {'SChg': [], 'SDis': [], 'EChg': [], 'EDis': []}  # set Idx as dictionary
    for i in np.arange(1, n):
        if ((Current[i - 1] == 0) | (Current[i - 1] > 0)) & (Current[i] < 0):
            Idx['SDis'].append(i)
        if ((Current[i - 1] == 0) | (Current[i - 1] < 0)) & (Current[i] > 0):
            Idx['SChg'].append(i)
        if (Current[i - 1] < 0) & ((Current[i] == 0) | (Current[i] > 0)):
            Idx['EDis'].append(i - 1)
        if (Current[i - 1] > 0) & ((Current[i] == 0) | (Current[i] < 0)):
            Idx['EChg'].append(i - 1)
    return Idx


# Set EIS Data Path
eis_path = "C:/Users/jeongbs1/오토실리콘/1. python_code/Practice_with_paper_data_sets/Accelerated SoH estimation of second life LIB via EIS and ML/DIB_Data/EIS_Test"
eis_list = os.listdir(eis_path) # Set EIS File list
eis_col = ["Frequency", "Real_Z", "Imaginary_Z"] # # Set Columns name

# Set EIS Dictionary by DataFrame
eis_raw_df = {}
for num in np.arange(len(eis_list)):
    eis_Cell = int(eis_list[num].split("_")[0].split("Cell")[1])
    eis_SoH = int(eis_list[num].split("_")[1].split("SOH")[0])
    eis_Temp = int(eis_list[num].split("_")[2].split("degC")[0])
    eis_SoC = int(eis_list[num].split("_")[3].split("SOC")[0])
    eis_raw_df[eis_list[num]] = pd.read_excel(eis_path+"/"+eis_list[num], header=None)
    eis_raw_df[eis_list[num]].columns = eis_col
    eis_raw_df[eis_list[num]]["Cell_name"] = eis_Cell
    eis_raw_df[eis_list[num]]["SoH"] = eis_SoH
    eis_raw_df[eis_list[num]]["Temp"] = eis_Temp
    eis_raw_df[eis_list[num]]["SoC"] = eis_SoC
len(eis_raw_df)


# Set Transform DataFrame for Stacking by Cell name
eis_df = pd.DataFrame
for num in np.arange(len(eis_list)):
    wide_format = {}
    for _, row in eis_raw_df[eis_list[num]].iterrows():
        freq = row['Frequency']
        wide_format["Cell_name"] = eis_raw_df[eis_list[num]]["Cell_name"][0]
        wide_format["SoH"] = eis_raw_df[eis_list[num]]["SoH"][0]
        wide_format["Temp"] = eis_raw_df[eis_list[num]]["Temp"][0]
        wide_format["SoC"] = eis_raw_df[eis_list[num]]["SoC"][0]
        wide_format[f'F{freq}_Real_Z'] = row['Real_Z']
        wide_format[f'F{freq}_Imaginary_Z'] = row['Imaginary_Z']
    wide_df = pd.DataFrame([wide_format]) # Transform DataFrame
    info_cols = [col for col in wide_df.columns if not 'Z' in col] # info Colums
    real_cols = [col for col in wide_df.columns if 'Real_Z' in col] # Real Colums
    imaginary_cols = [col for col in wide_df.columns if 'Imaginary_Z' in col] # Imaginary Colums
    ordered_cols = info_cols + real_cols + imaginary_cols # Make columns Order
    wide_df = wide_df[ordered_cols] # Set columns Order to DataFrame
    # Stacking DataFrame
    if num == 0:
        eis_df = wide_df
    else:
        eis_df = pd.concat([eis_df, wide_df],axis=0)

# Set Condition
Con_SoH, Con_Temp , Con_SoC = [100,95,90,85,80], [15, 25, 35], [5, 20, 50, 70, 95]
Con_F = list(eis_raw_df[eis_list[0]].loc[:,"Frequency"])
len(Con_F)
# Set Case 1 dataFrame
Case_1_df = eis_df.copy().reset_index(drop=True)

# Set Case 2 dataFrame
Case_2_df = eis_df.copy().reset_index(drop=True)


# Set SoH, Temp df by SoC
SoH95_Temp_25_df = Case_2_df[(Case_2_df['SoH']==95)&(Case_2_df['Temp']==25)].reset_index(drop=True)
SoH95_Temp_25_df.loc[:,"F10000.0_Real_Z":"F0.01_Real_Z"].iloc[0]

plt.figure()
for row in np.arange(len(SoH95_Temp_25_df)):
    re = SoH95_Temp_25_df.loc[:,"F10000.0_Real_Z":"F0.01_Real_Z"].iloc[row]
    im = -(SoH95_Temp_25_df.loc[:,"F10000.0_Imaginary_Z":"F0.01_Imaginary_Z"].iloc[row])
    plt.plot(re,im, "o-", label=str(SoH95_Temp_25_df.loc[:,"SoC"].iloc[row])+"%")
plt.xlabel("Real_Z")
plt.ylabel("Imaginary_Z")
plt.legend(loc="upper left")
plt.tight_layout()

plt.figure()
for row in np.arange(len(SoH95_Temp_25_df)):
    im = abs(SoH95_Temp_25_df.loc[:,"F10000.0_Imaginary_Z":"F0.01_Imaginary_Z"].iloc[row])
    plt.plot(Con_F, im, "o-", label=str(SoH95_Temp_25_df.loc[:,"SoC"].iloc[row])+"%")
plt.xlabel("Frequency")
plt.ylabel("Imaginary_Z")
plt.legend(loc="upper right")
plt.tight_layout()

plt.figure()
for row in np.arange(len(SoH95_Temp_25_df)):
    re = abs(SoH95_Temp_25_df.loc[:,"F10000.0_Real_Z":"F0.01_Real_Z"].iloc[row])
    plt.plot(Con_F, re, "o-", label=str(SoH95_Temp_25_df.loc[:,"SoC"].iloc[row])+"%")
plt.xlabel("Frequency")
plt.ylabel("Real_Z")
plt.legend(loc="upper right")
plt.tight_layout()


SoH95_df = Case_2_df[(Case_2_df['SoC']==20)].reset_index(drop=True)
plt.figure()
plt.plot((SoH95_df.loc[:,"F10000.0_Real_Z":"F0.01_Real_Z"].idxmin(axis=1)).str.extract('(\d+)').astype(float), "o-")


Case_2_df.loc[:,"F10000.0_Real_Z":"F0.01_Real_Z"].iloc[0]==min(Case_2_df.loc[:,"F10000.0_Real_Z":"F0.01_Real_Z"].iloc[0])
