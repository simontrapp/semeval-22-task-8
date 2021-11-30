import requests
import time
import csv
import shutil
from tqdm import tqdm
from pathlib import Path
from typing import Tuple


def check_download(source_info, ftype='.html'):
    with open(source_info, 'r', encoding='utf-8') as fh:
        csv_reader = csv.reader(fh, delimiter=',')
        rows = [x for x in csv_reader]

    fail_download = []
    for row in tqdm(rows[1:]):
        ids = row[2].split('_')
        for i, pid in enumerate(ids):
            folder = pid[-2:]
            fname = pid + ftype
            target = Path(__file__).parent.resolve() / 'data' / folder / fname
            if not target.exists():
                items = '\t'.join([row[i], pid, row[3+i], row[5+i]]) + '\n'
                fail_download.append(items)

    print(f'Failed scraping: {len(fail_download)}')
    if fail_download:
        if ftype == '.html':
            fname = 'failed_scrapy_html.txt'
        else:
            fname = 'failed_scrapy_json.txt'
        with open(fname, 'w', encoding='utf-8') as fh:
            fh.writelines(fail_download)


def create_new_source_info():
    fname = 'failed_scrapy_html.txt'
    with open(fname, 'r', encoding='utf-8') as fh:
        csv_reader = csv.reader(fh, delimiter='\t')
        rows = [x for x in csv_reader]

    header = ['url1_lang', 'url2_lang', 'pair_id', 'link1', 'link2',
              'ia_link1', 'ia_link2', 'Geography', 'Entities', 'Time',
              'Narrative', 'Overall', 'Style', 'Tone']
    new_source = [header]
    length = len(rows)
    if len(rows) % 2 != 0:
        length = len(rows) + 1
        rows.append(rows[-1])
    for i in range(0, length, 2):
        items1 = rows[i]
        items2 = rows[i+1]
        lang1 = items1[0]
        lang2 = items2[0]
        ids = items1[1] + '_' + items2[1]
        url1 = items1[2]
        url2 = items2[2]
        ia1 = items1[3]
        ia2 = items2[3]
        row = [lang1, lang2, ids, url1, url2, ia1, ia2, '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']
        new_source.append(row)

    #with open('failed_scrapy_batch.tsv', 'w', encoding='utf-8') as fh:
    #    fh.writelines(new_source)

    #with open('failed_scrapy_batch.tsv', 'r', encoding='utf-8') as fh:
    #    csv_reader = csv.reader(fh, delimiter='\t')
    #    rows = [x for x in csv_reader]

    with open('failed_scrapy_batch.csv', 'w', encoding='utf-8', newline='') as fh:
        writer = csv.writer(fh, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(new_source)


def main():
    # Module to check failed downloads (scrapings)
    check_download(TARGET_DATA, '.json')

    # Module to create new data_batch file for re-scraping
    # create_new_source_info()


if __name__ == '__main__':
    TARGET_DATA = 'semeval-2022_task8_train-data_batch.csv'
    main()
