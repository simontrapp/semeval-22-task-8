from utils_eval import calculate_result
from pathlib import Path
import csv


def load_labels(data, fname, key) -> tuple:
    keys = {
        'geo': 0, 'entity': 1, 'time': 2,
        'overall': 3, 'narrative': 4,
        'style': 5, 'tone': 6
    }
    index = keys[key]
    # Load prediction
    with open(fname, 'r') as fh:
        csv_reader = csv.reader(fh)
        rows = [x for x in csv_reader]
    if rows[0][0] == 'pair_id':
        rows = rows[1:]

    gold_ids = [x[0] for x in data]
    gold = []
    pred = []
    for row in rows:
        try:
            pos = gold_ids.index(row[0])
        except ValueError:
            pos = None
        if pos:
            pred.append(float(row[1]))
            gold.append(float(data[pos][1][index]))

    return gold, pred


def load_gold(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as fh:
        csv_reader = csv.reader(fh)
        rows = [x for x in csv_reader]

    data = []
    for row in rows[1:]:
        lang = [row[0], row[1]]
        # Scores: Geography, Entities, Time, Overall, Narrative, Style, Tone
        scores = [row[7], row[8], row[9], row[11], row[10], row[12], row[13]]
        data.append((row[2], scores, lang))
    return data


if __name__ == '__main__':
    # INPUT_CSV = 'sne03-sbert-cos-combined_time.csv'
    # INPUT_CSV = 'sbert128-best-title_text-combined_time.csv'
    INPUT_CSV = 'predictions-validation_cnn-pred-combined_time.csv'
    GOLD_CSV = Path('gold_test.csv')

    gold_data = load_gold(GOLD_CSV)
    gl, pl = load_labels(gold_data, INPUT_CSV, 'overall')
    print(len(gl))
    n, r2, mse, rmse, mae, pearson = calculate_result(gl, pl, pout=False)
    print(f'n:    {n}')
    print(f'R2:   {r2:.3f}')
    print(f'MSE:  {mse:.3f}')
    print(f'RMSE: {rmse:.3f}')
    print(f'MAE:  {mae:.3f}')
    print(f'PCC:  {pearson:.3f}')
