import os
import json
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


def parse_raw_info(files):
    data: Dict[str, str, str, str, List[str], List[str]] = dict()
    for file in tqdm(files):
        with open(file, 'r', encoding='utf-8') as fh:
            raw = json.load(fh)

        meta = raw['meta_data']
        # Get title
        title = raw['title']
        if not title:
            if 'og' in meta.keys() and 'title' in meta['og'].keys():
                title = meta['og']['title']
            elif 'twitter' in meta.keys() and 'title' in meta['twitter'].keys():
                title = meta['twitter']['title']
        data['title'] = title

        # Get description
        desc = raw['meta_description']
        if not desc:
            if 'description' in meta.keys() and not meta['description']:
                desc = meta['description']
            elif 'og' in meta.keys() and 'description' in meta['og'].keys():
                desc = meta['og']['description']
            elif 'twitter' in meta.keys() and 'description' in meta['twitter'].keys():
                desc = meta['twitter']['description']
        data['description'] = desc

        # Get published date
        pdate = raw['publish_date']
        if not pdate and 'article' in raw['meta_data'].keys() and \
                'published_time' in raw['meta_data']['article'].keys():
            pdate = raw['meta_data']['article']['published_time']
        if pdate is None:
            pdate = ''
        data['publish_date'] = pdate

        # Get text
        text = raw['text']
        data['text'] = text

        # Get keywords, news_keywords
        kw = raw['meta_keywords']
        if '' in kw:
            kw.remove('')
        if 'keywords' in meta.keys():
            if type(meta['keywords']) is str:
                keywords = [x.strip() for x in meta['keywords'].split(',')]
                kw.extend(keywords)
        if 'news_keywords' in meta.keys():
            if type(meta['news_keywords']) is str:
                keywords = [x.strip() for x in meta['news_keywords'].split(',')]
                kw.extend(keywords)
        data['keywords'] = list(set(kw))

        # Get tags
        tags = raw['tags']
        if '' in tags:
            tags.remove('')
        data['tags'] = tags

        fname = file[file.rfind('\\')+1:].split('.')[0] + '.json'
        fpath = Path('./training_data') / fname
        with open(fpath, 'w', encoding='utf-8') as fh:
            fh.write(json.dumps(data, indent=2))


def correcting_keywords():
    raw_source = Path('./training_data')
    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(raw_source)
                 for f in filenames if os.path.splitext(f)[1].lower() == '.json']

    for file in tqdm(files):
        with open(file, 'r', encoding='utf-8') as fh:
            data = json.load(fh)

        if '' in data['keywords']:
            data['keywords'].remove('')

            with open(file, 'w', encoding='utf-8') as fh:
                fh.write(json.dumps(data, indent=2))


def correcting_date():
    raw_source = Path('./training_data')
    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(raw_source)
             for f in filenames if os.path.splitext(f)[1].lower() == '.json']

    for file in tqdm(files):
        with open(file, 'r', encoding='utf-8') as fh:
            data = json.load(fh)

        if data['publish_date'] is None:
            data['publish_date'] = ''

            with open(file, 'w', encoding='utf-8') as fh:
                fh.write(json.dumps(data, indent=2))


def rename_files():
    raw_source = Path('./training_data')
    all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(raw_source)
                 for f in filenames if os.path.splitext(f)[1].lower() == '.dat']
    for file in twdm(all_files):
        os.rename(file, file.replace('.dat', '.json'))


def main():
    raw_source = Path('./scapy_scraped')
    all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(raw_source)
                 for f in filenames if os.path.splitext(f)[1].lower() == '.json']
    parse_raw_info(all_files)


if __name__ == '__main__':
    # main()
    # correcting_keywords()
    correcting_date()
    # rename_files()
