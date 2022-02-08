import pandas
from scipy.stats import pearsonr


def calculate_pearson(reference_csv: str, predictions_csv: str):
    data_ref = pandas.read_csv(reference_csv, index_col='pair_id')
    data_pred = pandas.read_csv(predictions_csv, index_col='pair_id')
    data = data_ref.join(data_pred, lsuffix='_ref', rsuffix='_pred').dropna()    # only keeps rows from reference, omits rest
    return pearsonr(data['Overall_ref'], data['Overall_pred'])


# on 10% train split
print('10% split:   ----------------------------------------')
print(f"CNN only:           {calculate_pearson('../../data/split/test.csv', '../../models/cnn_pred/predictions_cnn_validation.csv')}")
print(f"+ Random Forest:    {calculate_pearson('../../data/split/test.csv', '../../models/ft-cnn/predictions-validation.csv')}")
print('-----------------------------------------------------')

# on eval set
print('Eval set:    ----------------------------------------')
print(f"CNN only:           {calculate_pearson('../../data/final_evaluation_data.csv', '../../models/final_predictions-cnn.csv')}")
print(f"+ Random Forest:    {calculate_pearson('../../data/final_evaluation_data.csv', '../../models/final_predictions-rf.csv')}")
print(f"+ Pub Date:         {calculate_pearson('../../data/final_evaluation_data.csv', '../eval_score/predictions-rf-combined_time.csv')}")
print('-----------------------------------------------------')


def calculate_performance_per_language(reference_csv: str, predictions_csv: str, index: str):
    data_ref = pandas.read_csv(reference_csv, index_col='pair_id')
    data_pred = pandas.read_csv(predictions_csv, index_col='pair_id')
    data = data_ref.join(data_pred, lsuffix='_ref', rsuffix='_pred').dropna()    # only keeps rows from reference, omits rest
    data['language'] = data['url1_lang'] + '-' + data['url2_lang']
    results_per_lang = dict()
    for lang in set(data['language'].tolist()):
        filtered_data = data[data['language'] == lang]
        results_per_lang[lang] = [round(pearsonr(filtered_data['Overall_ref'], filtered_data['Overall_pred'])[0], 2)]
    return pandas.DataFrame(results_per_lang, index=[index])


# 10% train split
lang_10_cnn = calculate_performance_per_language('../../data/split/test.csv', '../../models/cnn_pred/predictions_cnn_validation.csv', '10 TextCNN')
lang_10_rf = calculate_performance_per_language('../../data/split/test.csv', '../../models/ft-cnn/predictions-validation.csv', '10 Random Forest')

# eval set
lang_eval_cnn = calculate_performance_per_language('../../data/final_evaluation_data.csv', '../../models/final_predictions-cnn.csv', 'Eval TextCNN')
lang_eval_rf = calculate_performance_per_language('../../data/final_evaluation_data.csv', '../../models/final_predictions-rf.csv', 'Eval Random Forest')
lang_eval_pd = calculate_performance_per_language('../../data/final_evaluation_data.csv', '../eval_score/predictions-rf-combined_time.csv', 'Eval Publish Date')

# append to final latex table
final_lang_table = lang_10_cnn.append(lang_10_rf).append(lang_eval_cnn).append(lang_eval_rf).append(lang_eval_pd)
print(final_lang_table.to_latex(na_rep=''))
