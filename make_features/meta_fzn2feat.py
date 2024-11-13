import json, subprocess, tqdm, pandas, sys

f = open(sys.argv[1])
data = json.load(f)
f.close()

features = []

for d in tqdm.tqdm(data):
    value = d["instance_value"]
    name = d["instance_name"].split("/")[-1]
    f = open(f"instances/{name}", "w")
    f.write(value)
    f.close()
    out = \
        subprocess.run(["python", "make_features/generate.py", "-t", "fzn2feat", "-i", f"instances/{name}", "-e", "01_compact.eprime", "--output", "json", "--time"], stdout=subprocess.PIPE)

    out = json.loads(out.stdout.decode())
    new_out = out["features"]
    new_out["time"] = out["time"]
    new_out["inst"] = d["instance_name"]
    features.append(new_out)

f = open("backup", "w")
json.dump(features, f)
f.close()
df = pandas.DataFrame(features)
df.to_csv("fzn2feat_EFPA.csv", index=False)
