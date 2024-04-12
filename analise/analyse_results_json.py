from load_jsons import load_jsons
from make_stats import make_stats
from sys import argv
from os.path import isdir
from os import makedirs
from typing import Any

def rebuild(data:list[dict]) -> list[dict]:
    out = []
    for datapoint in data:
        solver = datapoint["solver"]
        if solver == "or-tools":
            continue
            if "--threads=8" in datapoint["solverOptions"][0]:
                solver = "or-tools-8"
            else:
                solver = "or-tools-1"
        out.append({
            "instance": datapoint["essenceParams"][0],
            "model": datapoint["useExistingModels"][0].split("/")[-1],
            "solver": solver,
            "total_time": datapoint["totalTime"] if datapoint["status"] == "OK" else 36000,
            "status": datapoint["status"],
            "solver_options": datapoint["solverOptions"][0]
        })
    return out

def extract_combination_data(inst_data:dict[str,dict], combinations:list[str])-> list:
    out = []
    for key in inst_data:
        out_point = {'inst': key}
        for combination in combinations:
            out_point[combination] = inst_data[key][combination]
        out.append(out_point)
    return out

def parse_args() -> dict:
    args:dict[str,Any] = {
        "verbose":False,
        "save":False,
    }

    for arg in argv:
        if arg == "--help":
            return { "help":True }
        if arg == "--verbose" or arg == "-v":
            args["verbose"] = True
        elif "stats-folder" in arg:
            args["stats-folder"] = arg.replace("stats-folder=", "")
        else:
            args["folder"] = arg

    return args

def print_verbose(msg:str, verbose:bool):
    if verbose:
        print(msg)
def print_help():
    print(f"""
usage: python {argv[0]} folder [options]
--help          shows this message
--verbose/-v    print to stdout the results of the scripts
--stats-folder  save the stats into a particular folder
        """)

def main():
    args = parse_args()
    if "help" in args:
        print_help()   
        return
    if not "folder" in args:
        print("invalid usage, please see --help for reference")
        return

    verbose = args["verbose"]
    if not isdir(args["stats-folder"]):
        print_verbose("folder missing, generating it", verbose)
        makedirs(args["stats-folder"])

    data = load_jsons(args["folder"], verbose=verbose)
    print_verbose("JSONs loaded", verbose)
    data = rebuild(data)
    
    print_verbose("making graphs", verbose)
    make_stats(data, args['stats-folder'])
    print_verbose("graphs generated", verbose)

if __name__ == "__main__":
    main()
