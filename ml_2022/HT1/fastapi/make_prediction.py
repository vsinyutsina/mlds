import pickle
import pandas as pd
import numpy as np

def normalize_data(sample):

    cols = ['mileage', 'engine', 'max_power', 'seats']

    sample = sample.drop(columns=['torque'])

    sample['mileage'] = sample['mileage'].str.split()
    sample['mileage'] = sample['mileage'].apply(lambda x: x if isinstance(x, float) else x[0]) \
                                                                                    .astype(float)

    sample['engine'] = sample['engine'].str.split()
    sample['engine'] = sample['engine'].apply(lambda x: x if isinstance(x, float) else x[0])\
                                                                                    .astype(float)

    sample['max_power'] = sample['max_power'].str.split()
    sample['max_power'] = sample['max_power'].apply(lambda x: np.nan if isinstance(x, float) else x[0])
    sample['max_power'] = sample['max_power'].apply(lambda x: np.nan if x == 'bhp' else x) \
                                                                                    .astype(float)

    for col in cols:
        sample.loc[sample[col].isna(), col] = sample[col].quantile(0.5)
        if col in ['engine', 'seats']:
            sample[col] = sample[col].astype(int)

    categories = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
    num_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power']

    X_test_nums = sample[num_cols]
    X_test_cat = sample[categories]

    # Сгенерировать новые признаки на основе уже существующих

    X_test_nums.loc[:, 'max_power_2'] = (sample.loc[:, 'max_power']) ** 2
    X_test_nums.loc[:, 'mileage_2'] = sample.loc[:, 'mileage'] ** 2
    X_test_nums.loc[:, 'year_2'] = sample.loc[:, 'year'] ** 2

    # Добыть новые признаки
    num_owner = {
        'First': 1,
        'Second': 2,
        'Third': 3,
        'Fourth': 4,
        'Test': 0
    }

    X_test_nums.loc[:, 'owner_num'] = sample['owner'].str.split().apply(lambda x: num_owner[x[0]])

    X_test_cat.loc[:, 'model'] = sample.name.str.split().apply(lambda x: x[0])
    X_test_cat.seats = X_test_cat.seats.astype(str)

    sample = pd.concat([X_test_nums, X_test_cat], axis=1)
    return sample


def make_prediction(df_sample):
    categories = ['fuel', 'seller_type', 'transmission', 'seats', 'model']

    cols = ['year', 'km_driven', 'mileage_2', 'max_power', 'engine',
            'max_power_2', 'year_2', 'mileage', 'owner_num']

    with open('../model_weights/ridge_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('../model_weights/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('../model_weights/one_hot_encoder.pkl', 'rb') as f:
        one_hot_encoder = pickle.load(f)

    df_sample = normalize_data(df_sample)

    # вещественные признаки
    df_sample_standard = scaler.transform(df_sample[cols])
    df_sample_standard = pd.DataFrame(data=df_sample_standard, columns=scaler.get_feature_names_out())

    # категориальные признаки
    df_sample_cat = one_hot_encoder.transform(df_sample[categories])
    df_sample_cat = pd.DataFrame(data=df_sample_cat, columns=one_hot_encoder.get_feature_names_out())

    # предсказание
    df_sample_final = pd.concat([df_sample_standard, df_sample_cat], axis=1)

    y_sample_pred = model.predict(df_sample_final)

    return y_sample_pred