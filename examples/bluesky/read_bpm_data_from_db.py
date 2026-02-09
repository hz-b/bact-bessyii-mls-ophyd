import logging
logging.basicConfig(level=logging.WARNING)

import datetime
import os
import numpy as np
import math
from concurrent.futures import ProcessPoolExecutor

import jsons
from databroker import catalog
from tqdm import tqdm

logger = logging.getLogger("bpm-read-test")


from bact_device_models.devices.bpm_turn_by_turn_data import BPMTurnByTurnDataCollection

def _load_one(d):
    # d is an already-decoded dict
    return jsons.load(d, BPMTurnByTurnDataCollection)

def parallel_load(run):
    items = list(run["col-col"].values)
    n = len(items)
    if n == 0:
        return []

    num_workers = min(4, n)
    # tune chunksize: small enough to keep processes busy, not too small to cause overhead
    chunksize = max(1, math.ceil(n / num_workers))

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        # ex.map preserves order; wrap with tqdm to show progress
        bpm_data = list(tqdm(ex.map(_load_one, items, chunksize=chunksize), total=n))
    return bpm_data


def main():
    uuid = "f77c1dd1"
    start = datetime.datetime.now()
    db = catalog["heavy_local"]
    data = db[uuid]
    data_accessed = datetime.datetime.now()
    run = data.primary.read()
    run_read = datetime.datetime.now()
    # bpm_data = [
    #    jsons.load(one_reading, BPMTurnByTurnDataCollection)
    #    for one_reading in run["col-col"].values
    #]
    bpm_data = parallel_load(run)
    data_converted = datetime.datetime.now()
    txt = f"""
Time required for
 
access to data      {data_accessed - start}  
run read            {run_read - data_accessed} 
bpm data converted  {data_converted - run_read}
total time          {data_converted - start} 
"""
    print(txt)


if __name__ == "__main__":
    main()