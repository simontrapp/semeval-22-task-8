import pandas


def calculate_language_count(reference_csv: str):
    data = pandas.read_csv(reference_csv, index_col='pair_id')
    data['language'] = data['url1_lang'] + '-' + data['url2_lang']
    results_per_lang = dict()
    for lang in set(data['language'].tolist()):
        count = len(data[data['language'] == lang])
        results_per_lang[lang] = [count, round(count / len(data) * 100, 2)]
    print(pandas.DataFrame(results_per_lang, index=['Absolute Count', 'Percentage [%]']).sort_values('Absolute Count', axis=1, ascending=False).to_latex())


calculate_language_count('../../data/semeval-2022_task8_train-data_batch.csv')
calculate_language_count('../../data/final_evaluation_data.csv')
