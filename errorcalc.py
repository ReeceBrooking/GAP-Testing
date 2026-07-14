import numpy as np
from ase.io import read
from ase import Atoms
import sys

INPUT_FILE = str(sys.argv[1])
REFERENCE = str(sys.argv[2])
PREDICTION = str(sys.argv[3])

dataset = read(INPUT_FILE, index=":")

errors = []
references = []
predictions = []

for structure in dataset:
    references.append(structure.info[REFERENCE])
    predictions.append(structure.calc.results[PREDICTION])
    
references = np.array(references)
predictions = np.array(predictions)
diff = references - predictions
ss_res = np.sum(diff ** 2)
ss_tot = np.sum((references - references.mean(axis=0)) ** 2)

r2 = 1 - ss_res / ss_tot
rmse = np.sqrt(np.mean(diff ** 2))

print(f"RMSE: {rmse:.6f}  ({len(dataset)} structures)")
print(f"R2: {r2:.6f}  ({len(dataset)} structures)")