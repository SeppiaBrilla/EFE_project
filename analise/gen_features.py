from subprocess import run, PIPE, STDOUT
from devtools import pprint
from json import loads
from re import compile
from typing import Any
from sys import argv
import os

TEAMP_FILENAME = "feat-temp"
TEMP_FILE = f"./.cache/{TEAMP_FILENAME}"

def clean():
    current_dir = os.path.join(os.getcwd(), ".cache")

    files = [f for f in os.listdir(current_dir) if os.path.isfile(os.path.join(current_dir, f))]
    for file in files:
        if TEAMP_FILENAME in file:
            os.remove(os.path.join(current_dir, file))

def call_savilerow(eprime, param):
    command = ["savilerow", eprime, param, "-chuffed"]
    process = run(command, stdout=PIPE, stderr=STDOUT, check=True, encoding="UTF-8")
    pattern = "Created output file (.*fzn)"
    prog = compile(pattern)
    for line in process.stdout.splitlines():
        match = prog.match(line)
        if match:
            return match.group(1)

def call_conjure(eprime, param):
    command = ["conjure", "translate-parameter", f"--eprime={eprime}", f"--essence-param={param}", f"--eprime-param={TEMP_FILE}.eprime-param"]
    process = run(command, stdout=PIPE, stderr=STDOUT, check=True, encoding="UTF-8")
    if process.stdout != "":
        raise Exception(process.stdout)

def call_fzn2feat(model_file):
    command = ["fzn2feat", model_file, "dict"]
    process = run(command, stdout=PIPE, stderr=STDOUT, check=True, encoding="UTF-8")
    return process.stdout

def gen_features(eprime_file, param_file, file_name=None, save=True, verbose=True):

    if not os.path.exists(".cache"):
        os.mkdir(".cache")

    call_conjure(eprime_file, param_file)
    generated_file = call_savilerow(eprime_file, f"{TEMP_FILE}.eprime-param")
    res = call_fzn2feat(generated_file)
    res = res.replace("'", '"')

    if verbose:
        pprint(loads(res))
    if save:
        file_name = file_name if file_name != None else f"{eprime_file.split('/')[-1]}_{param_file.split('/')[-1]}.json"
        f = open(file_name, "w")
        f.write(res)
        f.close()
    clean()

def parse_args():
    args: dict[str,Any] = {
        "save":False,
        "verbose": False
    }
    for i in range(1, len(argv)):
        arg = argv[i]
        if arg == "--save" or arg == "-s":
            args["save"] = True
        elif arg == "--name":
            args["file_name"] = arg.replace("--name","")
        elif arg == "--verbose" or arg == "-v":
            args["verbose"] = True
        elif not "eprime_file" in args:
            args["eprime_file"] = arg
        else:
            args["param_file"] = arg
    return args

def main():
    if len(argv) < 2:
        print("error, please pass the necessary parameters. Use --help if needed")
        return
    if argv[1] == "--help" or argv[1] == "-h":
        print(f"""usage: {argv[0]} [options] eprime_file param_file
--save/-s       save the features as a file (default False)
--name          personalize the saved file name
--verbose/-v    print the results (default False)
--help/-h       shows this message""")
        return
    if len(argv) < 3:
        print("error, please pass the necessary parameters. Use --help if needed")
        return

    args = parse_args()
    gen_features(**args)

if __name__ == "__main__":
    main()
