import pandas as pd
from sys import argv
from math import isnan

def get_total_time(row):
    total_time = float(row["SolverTotalTime"]) + float(row["SavileRowTotalTime"])
    if row["SavileRowTimeOut"] == 1:
        total_time *= 10
    if not isnan(float(row["SolverTimeOut"])) and row["SolverTimeOut"] == 1:
        total_time *= 10
    return total_time


def check_and_update(row:pd.Series):
    row = row.to_dict()
    if 'or-tools8' == row['solver']:

        row1 = row.copy()
        row2 = row.copy()

        row1['solver'] = 'or-tools8-wallTime'
        row2['solver'] = 'or-tools8-cpuTime'

        row2['SolverTotalTime'] = float(row2['SolverTotalTime']) * 8

        return [row1, row2]
    return [row]

def duplicate_ortools(df:pd.DataFrame):
    new_rows = []
    for _, row in df.iterrows():
        new_rows += check_and_update(row)
    
    return pd.DataFrame(new_rows)


def add_stats(data:pd.DataFrame, save:bool, csv_file:str) -> pd.DataFrame:
    data = duplicate_ortools(data)
    data["TotalTime"] = data.apply(get_total_time, axis=1)
    if save:
        data.to_csv(csv_file, index=False)
    return data


def main():
    if len(argv) < 2:
        print("please provide a valid tsv file to parse")
        return

    data = pd.read_csv(argv[1])
    add_stats(data, True, argv[1])

if __name__ == "__main__":
    main()