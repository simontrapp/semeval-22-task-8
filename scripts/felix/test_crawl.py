from util import CSV_PATH
import pandas
import math
import os
import subprocess

DATA = pandas.read_csv(CSV_PATH)
size= 200
dir = "../data/subsets"
if not os.path.exists(dir):
    os.mkdir(dir)
for i in range(math.ceil(len(DATA)/size)):
    start = i*size
    end = ((i+1)*size)-1
    subset = DATA.iloc[start: end]

    path = os.path.join(dir, f"s{i}.csv")
    subset.to_csv(path, index=False)
    subprocess.call(f'start python -m semeval_8_2022_ia_downloader.cli --links_file={path} --dump_dir=../data/downloader_output_subsets', shell=True)
