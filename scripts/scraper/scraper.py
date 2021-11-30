import requests
import time
import csv
import shutil
from tqdm import tqdm
from pathlib import Path
from typing import Tuple


def scrape_html(page_urls: Tuple[str, str], target: Path) -> int:
    # Return code:
    # -1 : failed download
    # 0 or 1 : download success with 1st or 2nd url
    # 99 : html file already exist, skip re-download
    encoding = 'utf-8'
    pid = target.name
    filename = pid + '.html'
    target_name = target / filename
    page = None
    success = -1
    if not target_name.exists():
        try:
            page = requests.get(page_urls[0], timeout=5)
            if page.status_code == 200:
                success = 0
            else:
                page = requests.get(page_urls[1], timeout=5)
                if page.status_code == 200:
                    success = 1
            if success >= 0:
                target.mkdir(parents=True, exist_ok=True)
                with open(target_name, 'w', encoding=encoding) as fh:
                    fh.write(page.text)
        except Exception as e:
            print(f'Error Scraping {pid}: {e}')

        return success
    else:
        return 99


def prepare_scrape(source_info, rescrape=False):
    with open(source_info, 'r', encoding='utf-8') as fh:
        csv_reader = csv.reader(fh, delimiter='\t')
        rows = [x for x in csv_reader]

    if rows:
        fail_download = []
        for row in tqdm(rows):
            lang = row[0]
            pid = row[1]
            urls = (row[2], row[3])
            target = Path(__file__).parent.resolve() / 'html' / lang / pid
            if rescrape and target.exists():
                shutil.rmtree(target)
            # target.mkdir(parents=True, exist_ok=True)
            rc = scrape_html(urls, target)
            if rc == -1:
                items = '\t'.join(row) + '\n'
                fail_download.append(items)
                # print(f'Download failed - {id}')
            elif rc == 1:
                time.sleep(1)

        print(f'Failed scraping: {len(fail_download)}')
        if fail_download:
            with open(source_info, 'w', encoding='utf-8') as fh:
                fh.writelines(fail_download)


def get_url_info(source, output):
    with open(source, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = [x for x in csv_reader]

    source_info = []
    # Skip header, thus the rows[1:]
    for row in rows[1:]:
        lang1 = row[0]
        lang2 = row[1]
        ids = row[2].split('_')
        url1 = row[3]
        url2 = row[4]
        ia1 = row[5]
        ia2 = row[6]
        temp1 = '\t'.join([lang1, ids[0], url1, ia1]) + '\n'
        temp2 = '\t'.join([lang2, ids[1], url2, ia2]) + '\n'
        source_info.append(temp1)
        source_info.append(temp2)

    with open(output, 'w', encoding='utf-8') as fh:
        fh.writelines(source_info)


def main():
    # 1 - Parse url info from training data (csv file)
    # get_url_info(SOURCE_DATA, TARGET_DATA)

    # 2 - Scrape HTML files parsed from training data
    # prepare_scrape(TARGET_DATA)

    # 3 - Rescrape HTML files that failed before
    # Backup the failed.txt first in case some errors happens
    filename = 'failed.txt'
    # shutil.copyfile(filename, 'failed_backup.txt')
    for _ in range(RETRY_SCRAPE):
        prepare_scrape(filename, rescrape=True)


if __name__ == '__main__':
    SOURCE_DATA = 'semeval-2022_task8_train-data_batch.csv'
    TARGET_DATA = 'source_train.tsv'
    RETRY_SCRAPE = 1
    main()
