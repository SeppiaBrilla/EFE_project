from gen_features import gen_features
from sys import argv
from typing import Any
import os


def gen_all(eprime_file, param_folder, file_folder="features", save=True, verbose=True):
    
    files = [f for f in os.listdir(param_folder) if os.path.isfile(os.path.join(param_folder, f)) and ".param" in f]
    for param in files:
        instance = os.path.join(param_folder, param)
        gen_features(eprime_file, instance, save=save, verbose=verbose, file_name=f"{os.path.join(file_folder, param)}.json")

def parse_args():
    args: dict[str,Any] = {
        "save":False,
        "verbose": False,
        "file_folder": "features"
    }
    for i in range(1, len(argv)):
        arg = argv[i]
        if arg == "--save" or arg == "-s":
            args["save"] = True
        elif "--folder" in arg:
            args["file_folder"] = arg.replace("--folder=","")
        elif arg == "--verbose" or arg == "-v":
            args["verbose"] = True
        elif not "eprime_file" in args:
            args["eprime_file"] = arg
        else:
            args["param_folder"] = arg
    return args

def main():

    if len(argv) < 2:
        print("error, please pass sthe necessary parameters. Use --help if needed")
        return
    if argv[1] == "--help" or argv[1] == "-h":
        print(f"""usage: {argv[0]} [options] eprime_file param_folder
--save/-s       save the features as a file (default False)
--folder        personalize the saved file name (default is "features")
--verbose/-v    print the results (default False)
--help/-h       shows this message""")
        return
    if len(argv) < 3:
        print("error, please pass sthe necessary parameters. Use --help if needed")
        return

    args = parse_args()
    if not os.path.exists(args["file_folder"]):
        os.mkdir(args["file_folder"])
    gen_all(**args)

if __name__ == "__main__":
    main()
