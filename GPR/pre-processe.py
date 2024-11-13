# import library
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    cap_df[soh_column] = round((cap_df[cap_column].astype(float) / cap_df[cap_column].dropna().iloc[0]) * 100, 2)
# Set SoH Dataframe column
SoH_Colunms = ['cycle_1_3', 'Cell1_SoH', 'Cell3_SoH', 'cycle_2_4', 'Cell2_SoH', 'Cell4_SoH']

# Make SoH Dataframe
SoH_df = cap_df[SoH_Colunms]

# Make SoH DataFrame by Cell
SoH_df_cell1 = SoH_df[['cycle_1_3', 'Cell1_SoH']].dropna().copy()
SoH_df_cell2 = SoH_df[['cycle_2_4', 'Cell2_SoH']].dropna().copy()
SoH_df_cell3 = SoH_df[['cycle_1_3', 'Cell3_SoH']].dropna().copy()
SoH_df_cell4 = SoH_df[['cycle_2_4', 'Cell4_SoH']].dropna().copy()

SoH_df_cell1 = SoH_df_cell1.rename(columns={'cycle_1_3': 'Cycle'})
SoH_df_cell2 = SoH_df_cell2.rename(columns={'cycle_2_4': 'Cycle'})
SoH_df_cell3 = SoH_df_cell3.rename(columns={'cycle_1_3': 'Cycle'})
SoH_df_cell4 = SoH_df_cell4.rename(columns={'cycle_2_4': 'Cycle'})

SoH_df_cell1 = SoH_df_cell1.rename(columns={'Cell1_SoH': 'SoH'})
SoH_df_cell2 = SoH_df_cell2.rename(columns={'Cell2_SoH': 'SoH'})
SoH_df_cell3 = SoH_df_cell3.rename(columns={'Cell3_SoH': 'SoH'})
SoH_df_cell4 = SoH_df_cell4.rename(columns={'Cell4_SoH': 'SoH'})

SoH_df_cell1["Cell_Name"] = "Cell_1"
SoH_df_cell2["Cell_Name"] = "Cell_2"
SoH_df_cell3["Cell_Name"] = "Cell_3"
SoH_df_cell4["Cell_Name"] = "Cell_4"

SoH_df_cell1 = SoH_df_cell1[["Cell_Name", "Cycle", "SoH"]]
SoH_df_cell2 = SoH_df_cell2[["Cell_Name", "Cycle", "SoH"]]
SoH_df_cell3 = SoH_df_cell3[["Cell_Name", "Cycle", "SoH"]]
SoH_df_cell4 = SoH_df_cell4[["Cell_Name", "Cycle", "SoH"]]


'''Set the DataFrame from raw_data of EIS'''
# Set Capacity Raw data path
eis_forder_cell_path = "C:/Users/jeongbs1/오토실리콘/1. python_code/Practice_with_paper_data_sets/A comparative study of different features extracted from EIS in SoH for LIB/raw_data(SDI)"
# Make folder list
eis_forder_cell_list = [forder for forder in os.listdir(eis_forder_cell_path) if os.path.isdir(os.path.join(eis_forder_cell_path, forder))]
eis_raw_df = {cell: {} for cell in ["Cell1", "Cell2", "Cell3", "Cell4"]}

for cell_name in eis_forder_cell_list:
    cell_path = Path(eis_forder_cell_path) / cell_name
    cycle_list = [cycle for cycle in os.listdir(cell_path) if "cycle" in cycle]
    # 각 사이클에 대해 파일 읽기
    for cycle_name in cycle_list:
        cycle_path = cell_path / cycle_name / "50soc"
        # "50soc_25d" 파일 검색
        eis_file = next((file for file in os.listdir(cycle_path) if "25d" in file), None)
        # 파일이 있는 경우 데이터프레임으로 읽어 저장
        if eis_file:
            df_name = (cycle_name.split("cycle")[1])
            file_path = cycle_path / eis_file
            eis_raw_df[cell_name][df_name] = pd.read_csv(file_path, sep=r'\s+', header=None, names=["Frequency", "Real_Z", "Imaginary_Z"])


cell_num = list(eis_raw_df.keys())
stack_eis_row_df = {cell: pd.DataFrame() for cell in cell_num}
for cell in range(len(cell_num)):
    cycle_num = list((eis_raw_df[cell_num[cell]].keys()))
    for cyc in range(len(cycle_num)):
        df_dummy = eis_raw_df[cell_num[cell]][cycle_num[cyc]]
        tran_f_col = {}
        for i, row in df_dummy.iterrows():
            freq = f"{row['Frequency']:.2f}"  # 소수 둘째 자리까지 표현
            tran_f_col["Cycle"] = float(cycle_num[cyc])
            tran_f_col["SoC"] = int(50)
            tran_f_col["Temp"] = int(25)
            tran_f_col[f"F_{freq}_Re_Z"] = row['Real_Z']
            tran_f_col[f"F_{freq}_Im_Z"] = row['Imaginary_Z']
        transformed_df = pd.DataFrame([tran_f_col])
        info_cols = [col for col in transformed_df.columns if not 'Z' in col]
        real_cols = [col for col in transformed_df.columns if '_Re_Z' in col]
        imaginary_cols = [col for col in transformed_df.columns if '_Im_Z' in col]
        ordered_cols = info_cols + real_cols + imaginary_cols
        transformed_df = transformed_df[ordered_cols]
        stack_eis_row_df[cell_num[cell]] = pd.concat([stack_eis_row_df[cell_num[cell]], transformed_df], axis=0, ignore_index=True)
    stack_eis_row_df[cell_num[cell]].sort_values(by="Cycle", ascending=True, ignore_index=True, inplace=True)

merged_cell1_df = pd.merge(SoH_df_cell1, stack_eis_row_df[cell_num[0]], on="Cycle", how="inner")
merged_cell2_df = pd.merge(SoH_df_cell2, stack_eis_row_df[cell_num[1]], on="Cycle", how="inner")
merged_cell3_df = pd.merge(SoH_df_cell3, stack_eis_row_df[cell_num[2]], on="Cycle", how="inner")
merged_cell4_df = pd.merge(SoH_df_cell4, stack_eis_row_df[cell_num[3]], on="Cycle", how="inner")
broadband_df = pd.concat([merged_cell1_df, merged_cell2_df, merged_cell3_df, merged_cell4_df], axis=0, ignore_index=True)
broadband_df["Cycle"] = broadband_df["Cycle"].astype(int)
# broadband_df.to_excel("C:/Users/jeongbs1/Downloads/broadband_df.xlsx", index=False)



