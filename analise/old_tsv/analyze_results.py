from generate_csv import generate_csv
from chech_missing import check_missing
from add_aggregate_stats import add_stats
from generate_plots import make_stats
from sys import argv
from os.path import join, isdir
from os import makedirs

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
            args["json_name"] = arg.replace("--json-name=", "")
        elif "--csv-name=" in arg:
            args["csv_name"] = arg.replace("--csv-name=", "")
        elif "--folder=" in arg:
            args["folder"] = arg.replace("--folder=", "")
        else:
            args["data"] = arg

    if args["save"] and not "folder" in args:
        args["folder"] = "."
    if args["save"] and not "json_name" in args:
        args["json_name"] = args["data"].replace("csv", "json")
    if args["save"] and not "csv_name" in args:
        args["csv_name"] = args["data"].replace("tsv", "csv")

    return args

def print_verbose(msg:str, verbose:bool):
    if verbose:
        print(msg)

def main():

    args = parse_args()
    verbose = args["verbose"]
    save = args["save"]
    if not isdir(args["folder"]):
        print_verbose("folder missing, generating it", verbose)
        makedirs(args["folder"])

    print_verbose("generating the csv", verbose)
    data = generate_csv(args["data"], save=save, save_file_name=join(args["folder"], args["csv_name"]))
    print_verbose("csv generated", verbose)
    
    print_verbose("checking for missing", verbose)
    missings = check_missing(data, verbose, save=save, save_name=join(args["folder"], args["json_name"]))
    print_verbose("missing checked", verbose)

    print_verbose("dropping non-complete instances", verbose)
    not_complete = [val["instance_name"] for val in missings["not_complete"]]
    idxs = []

    for i in range(len(data)):
        row = data.iloc[i,:]
        if row["parameter"] in not_complete:
            idxs.append(i)
    data = data.drop(idxs)


    print_verbose("adding extra stats", verbose)
    data = add_stats(data, save, join(args["folder"], args["csv_name"]))
    print_verbose("stats added", verbose)

    print_verbose("making graphs", verbose)
    make_stats(data, args['folder'])
    print_verbose("graphs generated", verbose)

if __name__ == "__main__":
    main()