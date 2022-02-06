from utils_eval import calculate_result
from pathlib import Path
import csv


def get_diff():
    # Gold
    with open(GOLD_CSV, 'r', encoding='utf-8') as fh:
        csv_reader = csv.reader(fh)
        rows = [x for x in csv_reader]
    rows = rows[1:]
    ids_gl = [x[2] for x in rows]

    # Input
    with open(INPUT_CSV, 'r', encoding='utf-8') as fh:
        csv_reader = csv.reader(fh)
        rows = [x for x in csv_reader]
    if rows[0][0] == 'pair_id':
        rows = rows[1:]
    ids_pl = [x[0] for x in rows]

    diffs = []
    for ids in ids_pl:
        if ids not in ids_gl:
            diffs.append(ids)

    print(diffs)


if __name__ == '__main__':
    # INPUT_CSV = '../final/predictions-cnn.csv'
    INPUT_CSV = '../lingson/eval_result/sb128-default.csv'
    GOLD_CSV = Path('../../data/final_evaluation_data.csv')

    get_diff()