from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import math


def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)


def pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff


    return diffprod / math.sqrt(xdiff2 * ydiff2)


def calculate_result(gold_list, pred_list, fpath='', pout=True):
    combined = list(zip(gold_list, pred_list))
    new_gold = []
    new_pred = []
    for row in combined:
        if row[1] != 0:
            new_gold.append(row[0])
            new_pred.append(row[1])
    gold_list = new_gold
    pred_list = new_pred

    r2 = r2_score(gold_list, pred_list)
    mse = mean_squared_error(gold_list, pred_list)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(gold_list, pred_list)
    # pearson = pearson_def(gold_list, pred_list)
    pearson, _ = pearsonr(gold_list, pred_list)

    if pout:
        print(f'n:    {len(gold_list)}')
        print(f'R2:   {r2:.3f}')
        print(f'MSE:  {mse:.3f}')
        print(f'RMSE: {rmse:.3f}')
        print(f'MAE:  {mae:.3f}')
        print(f'PCC:  {pearson:.3f}')
    if fpath:
        fname = fpath.replace('.csv', '.txt')
        with open(fname, 'w', encoding='utf-8') as fh:
            print(f'n:    {len(gold_list)}', file=fh)
            print(f'R2:   {r2:.3f}', file=fh)
            print(f'MSE:  {mse:.3f}', file=fh)
            print(f'RMSE: {rmse:.3f}', file=fh)
            print(f'MAE:  {mae:.3f}', file=fh)
            print(f'PCC:  {pearson:.3f}', file=fh)

    return len(gold_list), r2, mse, rmse, mae, pearson
