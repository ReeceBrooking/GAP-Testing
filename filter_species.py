import sys
from ase.io import read, write

INPUT   = sys.argv[1]                       # e.g. train_waterbulk.xyz
OUTPUT  = sys.argv[2]                        # e.g. filtered.xyz
ALLOWED = set(int(z) for z in sys.argv[3:])  # e.g. 1 6 8  ->  {1, 6, 8}

dataset = read(INPUT, index=":")

# keep a structure only if every atom in it is one of the allowed species
kept = [s for s in dataset if set(s.get_atomic_numbers()) <= ALLOWED]

write(OUTPUT, kept)

print(f"Allowed species: {sorted(ALLOWED)}")
print(f"Kept {len(kept)} / {len(dataset)} structures -> {OUTPUT}")
