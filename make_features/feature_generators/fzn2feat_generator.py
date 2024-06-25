from .base_generator import Generator
from subprocess import run, PIPE, STDOUT
import json
from re import compile
import os

class Fzn2feat_generator(Generator):

    TEAMP_FILENAME = "feat-temp"
    TEMP_FILE = f"./.cache/{TEAMP_FILENAME}"
    
    def __init__(self, eprime) -> None:
        super().__init__()
        self.eprime = eprime

    def generate(self, instance:'str') -> 'dict[str,float]':
        features = self.__gen_features(self.eprime, instance)
        return json.loads(features)
    
    def __clean(self):
        current_dir = os.path.join(os.getcwd(), ".cache")

        files = [f for f in os.listdir(current_dir) if os.path.isfile(os.path.join(current_dir, f))]
        for file in files:
            if self.TEAMP_FILENAME in file:
                os.remove(os.path.join(current_dir, file))

    def __call_savilerow(self, eprime, param):
        command = ["runsolver", "-R", "16384", "savilerow", eprime, param, "-chuffed"]
        process = run(command, stdout=PIPE, stderr=STDOUT, check=True, encoding="UTF-8")
        pattern = "Created output file (.*fzn)"
        prog = compile(pattern)
        for line in process.stdout.splitlines():
            match = prog.match(line)
            if match:
                return match.group(1)
        raise Exception(process.stdout)

    def __call_conjure(self, eprime, param):
        command = ["conjure", "translate-parameter", f"--eprime={eprime}", f"--essence-param={param}", f"--eprime-param={self.TEMP_FILE}.eprime-param"]
        process = run(command, stdout=PIPE, stderr=STDOUT, encoding="UTF-8")
        if process.returncode != 0:
            raise Exception(process.stdout)

    def __call_fzn2feat(self, model_file):
        command = ["fzn2feat", model_file, "dict"]
        process = run(command, stdout=PIPE, stderr=STDOUT, check=True, encoding="UTF-8")
        return process.stdout

    def __gen_features(self, eprime_file, param_file):
        if not os.path.exists(".cache"):
            os.mkdir(".cache")

        self.__call_conjure(eprime_file, param_file)
        run(['ls', '.cache'])
        generated_file = self.__call_savilerow(eprime_file, f"{self.TEMP_FILE}.eprime-param")
        res = self.__call_fzn2feat(generated_file)
        res = res.replace("'", '"')
        self.__clean()
        return res


