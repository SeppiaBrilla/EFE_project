import re
from sys import argv
import os 

def find_variables(data:str):
    vars = re.findall("letting (.*) be", data)
    return vars

def remove_comments(content):
    comments = re.findall("(\$.*\\n)", content)
    for comment in comments:
        content = content.replace(comment,"")

    return content

def main():
    in_folder = argv[1]
    out_folder = argv[2]

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    params = [f for f in os.listdir(in_folder) if ".param" in f]
    for param in params:
        f = open(os.path.join(in_folder, param))
        content = f.read()
        f.close()

        content = remove_comments(content)
        content = content.replace("language Essence 1.3\n","")
        vars = find_variables(content)
        for i in range(len(vars)):
            var = vars[i]
            content = content.replace(f"letting {var} be", f"{i+1} =")

        f = open(os.path.join(out_folder, param.replace(".param",".bert")),"w")
        f.write(content)
        f.close()

if __name__ == "__main__":
    main()
