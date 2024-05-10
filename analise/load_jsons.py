import json
from sys import argv
from typing import Any
import os 

def load_folder(folder:str) -> list[dict]:
    files = [os.path.join(folder, f) for f in os.listdir(folder) if ".stats.json" in f]
    jsons = []
    
    for file in files:
        with open(file) as f:
            jsons.append(json.loads(f.read().replace("\t","")))

    return jsons

def find_sub_folders(folder:str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    dirs = []
    for file in os.listdir(folder):
        complete_File = os.path.join(folder, file)
        if os.path.isdir(complete_File):
            dirs.append(complete_File)
            dirs += find_sub_folders(complete_File)
    return dirs

def load_jsons(folder:str, verbose:bool, save:'str|None' = None)-> list[dict]:
    folders = find_sub_folders(folder)
    print_verbose(f"found {len(folders)} folders.", verbose)
    jsons = []
    for folder in folders:
        print_verbose(f"analysing {folder}:", verbose)
        files = load_folder(folder)
        print_verbose(f"    found {len(files)} json stats files in it", verbose)
        jsons += files

    if save != None:
        str_jsons = json.dumps(jsons)
        f = open(save, "w")
        f.write(str_jsons)
        f.close()
    
    return jsons

def print_verbose(string:str, verbose:bool) -> None:
    if not verbose:
        return
    print(string)

def main():
    if len(argv) < 2:
        print("error, please provide a folder to load. Use --help to have more info")
        return
    if argv[1] == "--help":
        print(f"""Usage: {argv[1]} folder [params]
--verbose/-v    print information about the loading process
--save          save the loaded files in an unified json file passed as parameter
""")
    args = load_args(argv[1:])
    load_jsons(args["foder"], args["verbose"], args["save"] if "save" in args else None)

def load_args(args:list[str]) -> dict:
    args_dict:dict[str,Any] = {"verbose": False}

    for arg in args:
        if "--verbose" == arg or "-v" == arg:
            args_dict["verbose"] = True
        elif "--save" in arg:
            args_dict["save"] = arg.replace("--save=", "")
        else:
            args_dict["folder"] = arg
    return args_dict

if __name__ == "__main__":
    main()
