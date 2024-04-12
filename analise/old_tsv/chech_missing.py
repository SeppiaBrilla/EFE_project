import pandas as pd
import numpy as np
from sys import argv
from devtools import pprint
from json import dump
import math

def check_missing(data:'pd.DataFrame|str', verbose:bool = True, save:bool = True, save_name:str = ""):

    if save and save_name == "":
        raise Exception("Error: provide save file name if you want to save the recap")
    if type(data) == str:
        data = pd.read_csv(data)

    instances = np.unique(data["parameter"])  
    inst_data = {inst: {} for inst in instances}
    for i in range(len(data)):
        row = data.iloc[i, :]
        inst_data[row["parameter"]][f"{row['heuristic']}_{row['solver']}"] = {"t":float(row["SolverTotalTime"]), "i": row["parameter"]}
    
    number_of_elements = np.unique([len(list(inst_data[inst].keys())) for inst in inst_data.keys()]).tolist()
    max_elements = max(number_of_elements)
    if verbose:
        print(f"There are {max_elements} maximum combination for a single instance")
        print(f"There are {number_of_elements} distinct number of combination in the provided data file")
        print(f"There are {len(instances)} instances in the file")


    complete_list = []

    for inst in inst_data.keys():
        combs = list(inst_data[inst].keys())
        if len(combs) == max_elements:
            complete_list = combs
            break

    if verbose:
        complete_list_str = '\n\t-'.join(complete_list)
        print(f"A complete instance has the following combinations: \n\t-{complete_list_str}")
    recap = {
        "max_elements": max_elements,
        "number_of_elements": number_of_elements,
        "number_of_instances": len(instances),
        "not_complete": []
    }

    for inst in inst_data.keys():
        combs = list(inst_data[inst].keys())
        for i in reversed(range(len(combs))):
            if math.isnan(inst_data[inst][combs[i]]["t"]):
                del combs[i]

        if len(combs) < max_elements:
            elem = {
                "instance_name":inst,
                "missing_combinations": []
            }
            for comb_to_have in complete_list:
                if not comb_to_have in combs:
                    elem["missing_combinations"].append(comb_to_have)

            recap["not_complete"].append(elem)

    if verbose:
        print("Here's the full list of instances with missing elements:")
        pprint(recap["not_complete"])

        print("complete json recap:")
        pprint(recap)
    if save:
        with open(save_name, 'w') as f:
            dump(recap, f)
    return recap

def parse_args() -> dict:
    args = {
        "verbose":False,
        "save":False,
    }

    for arg in argv:
        if arg == "--verbose" or arg == "-v":
            args["verbose"] = True
        elif arg == "--save" or arg == "-s":
            args["save"] = True
        elif "--json-name=" in arg:
            args["save_name"] = arg.replace("--json-name=", "")
        else:
            args["data"] = arg

    if args["save"] and not "save_name" in args:
        args["save_name"] = args["data"].replace("csv", "json")

    return args

def main():
    args = parse_args()
    check_missing(**args)   



if __name__ == "__main__":
    main()
