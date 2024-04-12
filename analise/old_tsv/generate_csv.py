import pandas as pd
from sys import argv

def transform(data: pd.DataFrame, columns: list)-> pd.DataFrame:
    dict_data = data.to_dict()
    keys = list(dict_data.keys())
    elements = list(dict_data[keys[0]].keys())
    dict_data = [{k: dict_data[k][e] for k in keys} for e in elements] 
    n_elements = len(dict_data)
    i = 0
    grouped = []
    extra_columns = [c for c in keys if c not in columns]
    while i < n_elements:
        current = dict_data[i]
        column_values = [current[c] for c in columns]
        extra = {}
        while column_values == [current[c] for c in columns] and i < n_elements:
            extra[current[extra_columns[0]]] = current[extra_columns[1]]
            i += 1
            if i < n_elements:
                current = dict_data[i]
        new_value = dict_data[i-1].copy()
        for c in extra_columns:
            del new_value[c]
        new_value.update(extra)
        grouped.append(new_value)
    return pd.DataFrame(grouped)

def generate_csv(file_name:str, save:bool = True, save_file_name = "") -> pd.DataFrame:
    if save and save_file_name == "":
        raise Exception("Error: provide save file name if you want to save the recap")

    data = pd.read_csv(file_name, sep="\t")
    data.columns=['problem', 'model', 'parameter', 'conjure_mode', 'optimization_level', 'heuristic', 'solver', 'var', 'value']
    df = transform(data, ['problem', 'model', 'parameter', 'conjure_mode', 'optimization_level', 'heuristic', 'solver'])
    if save:
        df.to_csv(save_file_name, index=False)
    return df

def main():
    if len(argv) < 2:
        print("please provide a valid tsv file to parse")
        return

    save_file = argv[1].replace('tsv','csv')
    if len(argv) == 3 and  "--csv-name=" in argv[2]:
        save_file = argv[2].replace("--csv-name=", '')

    generate_csv(argv[1], save_file)

if __name__ == "__main__":
    main()