import requests
import time
import csv
import shutil
from tqdm import tqdm
from pathlib import Path
from typing import Tuple


def check_url(source_info):
    with open(source_info, 'r', encoding='utf-8') as fh:
        csv_reader = csv.reader(fh, delimiter='\t')
        rows = [x for x in csv_reader]

    fail_download = []
    for row in tqdm(rows):
        lang = row[0]
        pid = row[1]
        urls = (row[2], row[3])
        filename = pid + '.html'
        target = Path(__file__).parent.resolve() / 'html' / lang / pid / filename
        if not target.exists():
            items = '\t'.join(row) + '\n'
            fail_download.append(items)

    print(f'Failed scraping: {len(fail_download)}')
    if fail_download:
        with open('failed1.txt', 'w', encoding='utf-8') as fh:

            fh.writelines(fail_download)


def main():
    check_url(TARGET_DATA)


if __name__ == '__main__':
    TARGET_DATA = 'source_train.tsv'
    main()
