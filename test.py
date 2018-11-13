import target.debug.libpyn5 as pyn5
from pathlib import Path
import shutil

dataset = Path("test.n5")
if dataset.is_dir():
    shutil.rmtree(dataset)

pyn5.create_dataset("test.n5", "piggies", [10,10,10],[2,2,2])
piggies = pyn5.Dataset("test.n5", "piggies")

try:
    piggies.write_block([0,0,0],[0,1,2,3])
    raise Exception("FAILED! This block needs 8 values, not 4!")
except ValueError as e:
    print(e)

try:
    piggies.write_block([0,0,0],[0,1,2,3,4,5,6,7,7,6,5,4,3,2,1,0])
    raise Exception("FAILED! This block needs 8 values, not 16!")
except ValueError as e:
    print(e)

try:
    piggies.write_block([-1,-1,-1], [0,1,2,3,4,5,6,7])
    raise Exception("FAILED! Allowed writing blocks with negative indicies")
except Exception as e:
    print(e)

try:
    piggies.write_block([10,10,10], [0,1,2,3,4,5,6,7])
    raise Exception("FAILED! Allowed writing blocks outside boundaries")
except Exception as e:
    print(e)