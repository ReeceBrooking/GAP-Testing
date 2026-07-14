import sys

INPUT  = sys.argv[1] if len(sys.argv) > 1 else "out.xyz"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "trimmed.xyz"

kept = 0
with open(INPUT) as fin, open(OUTPUT, "w") as fout:
    for line in fin:
        if line.startswith("AT"):
            fout.write(line[3:])   # drop the "AT " prefix -> valid xyz
            kept += 1

print(f"Wrote {kept} lines from {INPUT} to {OUTPUT}")
