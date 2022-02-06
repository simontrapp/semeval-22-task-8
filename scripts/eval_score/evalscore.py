from utils_eval import calculate_result
from pathlib import Path
import csv


def load_gold(csv_path, key):
    keys = {
        'geo': 0, 'entity': 1, 'time': 2,
        'narrative': 3, 'overall': 4,
        'style': 5, 'tone': 6
    }
    index = keys[key]
    with open(csv_path, 'r', encoding='utf-8') as fh:
        csv_reader = csv.reader(fh)
        rows = [x for x in csv_reader]

    data = []
    for row in rows[1:]:
        lang = [row[0], row[1]]
        # Scores: Geography, Entities, Time, Overall, Narrative, Style, Tone
        score = [row[7], row[8], row[9], row[11], row[10], row[12], row[13]]
        data.append((row[2], score, lang))

    scores = [float(x[1][index]) for x in data]
    return data, scores


def load_scores(gold_data, csv_path, avg=2.5):
    with open(csv_path, 'r', encoding='utf-8') as fh:
        csv_reader = csv.reader(fh)
        rows = [x for x in csv_reader]
    scores = []
    if rows[0][0] == 'pair_id':
        rows = rows[1:]
    ids = [x[0] for x in rows]
    scores = [x[1] for x in rows]
    zero_value = False
    new_scores = []
    for row in gold_data:
        id = row[0]
        index = ids.index(id)
        score = float(scores[index])
        if score == 0:
            zero_value = True
            score = avg
        new_scores.append(score)
    if zero_value:
        print(f'Rows with zero value detected. Changed to {avg}')
    return new_scores


if __name__ == '__main__':
    # INPUT_CSV = 'sne03-sbert-cos-combined_time.csv'
    # INPUT_CSV = 'sbert128-best-title_text-combined_time.csv'
    INPUT_CSV = '../final/predictions_cleaned_rounded1.csv'
    # INPUT_CSV = '../lingson/eval_result/sbert128-bestmix-finetune-combined_time1.csv'
    GOLD_CSV = Path('../../data/final_evaluation_data.csv')

    gdata, gl = load_gold(GOLD_CSV, 'overall')
    pl = load_scores(gdata, INPUT_CSV, 2.5)
    if len(gl) != len(pl):
        print(f'Different row numbers - Gold: {len(gl)} vs Input: {len(pl)}')
    else:
        n, r2, mse, rmse, mae, pearson = calculate_result(gl, pl, pout=False)
        print(f'n:    {n}')
        print(f'R2:   {r2:.3f}')
        print(f'MSE:  {mse:.3f}')
        print(f'RMSE: {rmse:.3f}')
        print(f'MAE:  {mae:.3f}')
        print(f'PCC:  {pearson:.4f}')
