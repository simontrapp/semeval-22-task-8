## Training

When your scraped training data is set up (set `DATA_DIR` and `CSV_PATH` accordingly), run:

```shell
python calculate_article_similarty.py
```

This calculates the sentence- and keyword-similarities of all document pairs with a pre-trained multi-lingual SBERT model. Then run:

```shell
python train_classifier.py
```

This trains a random forest model (n=100) and evaluates the performance on a test set (20% of training data). To train with the full training data, set `create_test_set` to `False`. The trained random forest model is saved as a joblib file.