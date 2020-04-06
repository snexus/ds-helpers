import pytest
import pandas as pd

from feature_tools.base_feature import BaseFeatures


@pytest.fixture
def df1():
    d = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [1, 2, 3, 4, 5],
        'feature3': [1, 2, 3, 4, 5],
        'feature4': [1, 2, 3, 4, 5],
        'meta_column1': [1, 2, 3, 4, 5],
        'meta_column2': [1, 1, 1, 1, 1],
        'outcome': [1, 1, 1, 0, 0]
    }

    return pd.DataFrame.from_dict(d), ['feature1',
                                       'feature2', 'feature3', 'feature4'], \
           ['meta_column1', 'meta_column2']


@pytest.fixture
def df2():
    d = {'feature1': [1, 2, 3, 4, 5],
         'feature2': [1, 2, 3, 4, 5],
         'meta_column1': [1, 2, 3, 4, 5],
         'meta_column2': [1, 1, 1, 1, 1],
         'outcome': [1, 1, 1, 0, 0]
         }

    return pd.DataFrame.from_dict(d), ['feature1',
                                       'feature2'], \
           ['meta_column1', 'meta_column2']


def test_validate_data_invalid_features(df1):
    feature_cols = ['feature1', 'feature2', 'featurex']
    with pytest.raises(KeyError):
        b = BaseFeatures(df1[0], feature_cols)


def test_validate_data_invalid_meta(df1):
    feature_cols = ['feature1', 'feature2', 'feature3']
    meta_cols = ['meta_column1', 'meta_column3']
    with pytest.raises(KeyError):
        b = BaseFeatures(df1[0], feature_cols, meta_columns=meta_cols)


def test_validate_outcome_column1(df1):
    class Features(BaseFeatures):
        def create_target(self):
            return 123

    df, feature_cols, meta_cols = df1
    f = Features(df, feature_cols, meta_columns=meta_cols)
    with pytest.raises(TypeError):
        print(f.outcome_column)


def test_validate_outcome_column2(df1):
    class Features(BaseFeatures):
        def create_target(self):
            return 'outcome1'

    df, feature_cols, meta_cols = df1
    f = Features(df, feature_cols, meta_columns=meta_cols)
    with pytest.raises(KeyError):
        print(f.outcome_column)


def test_validate_output_columns1(df1):
    class Features(BaseFeatures):
        def create_target(self):
            return 'outcome'

    df, feature_cols, meta_cols = df1
    f = Features(df, feature_cols, meta_columns=meta_cols)
    cols = f.data.columns
    assert len(cols) == 7

    for col in feature_cols + meta_cols:
        assert col in cols

    assert 'outcome' in cols


def test_validate_output_columns_suffix(df1):
    class Features(BaseFeatures):
        def create_target(self):
            return 'outcome'

    df, feature_cols, meta_cols = df1
    f = Features(df, feature_cols, meta_columns=meta_cols, feature_suffix='_my_suffix')
    cols = f.data.columns
    assert len(cols) == 7

    # Assert all feature columns have suffix
    for col in feature_cols:
        assert col + "_my_suffix" in cols

    # Assert meta columns don't have suffix
    for col in meta_cols:
        assert col in cols

    # Outcome should have suffix
    assert 'outcome' in cols


def test_validate_custom_features(df1):
    class Features(BaseFeatures):
        def create_target(self):
            return 'outcome'

        def transform(self):
            with self.data_accessor() as data:
                data['new_feat1'] = 2
                data['new_feat2'] = 3
            self._additional_features += ['new_feat1', 'new_feat2']

    df, feature_cols, meta_cols = df1
    f = Features(df, feature_cols, meta_columns=meta_cols)
    cols = f.data.columns
    assert len(cols) == 9

    for col in feature_cols + meta_cols + ['new_feat1', 'new_feat2']:
        assert col in cols

    assert 'outcome' in cols


def test_add_two_instances_mismatch_metacolumns(df1, df2):
    class Features(BaseFeatures):
        def create_target(self):
            return 'outcome'

        def transform(self):
            with self.data_accessor() as data:
                data['new_feat1'] = 2
                data['new_feat2'] = 3
            self._additional_features += ['new_feat1', 'new_feat2']

    df_1, feature_cols_1, meta_cols_1 = df1
    df_2, feature_cols_2, meta_cols_2 = df2

    f1 = Features(df_1, feature_cols_1, meta_columns=meta_cols_1, feature_suffix="_1")
    f2 = Features(df_2, feature_cols_2, meta_columns=[meta_cols_2[0]], feature_suffix="_2")

    with pytest.raises(TypeError):
        f3 = f1 + f2


def test_add_two_instances(df1, df2):
    class Features(BaseFeatures):
        def create_target(self):
            return 'outcome'

        def transform(self):
            with self.data_accessor() as data:
                data['new_feat1'] = 2
                data['new_feat2'] = 3
            self._additional_features += ['new_feat1', 'new_feat2']

    df_1, feature_cols_1, meta_cols_1 = df1
    df_2, feature_cols_2, meta_cols_2 = df2

    f1 = Features(df_1, feature_cols_1, meta_columns=meta_cols_1, feature_suffix="_1")
    f2 = Features(df_2, feature_cols_2, meta_columns=meta_cols_2, feature_suffix="_2")

    f3 = f1 + f2
    cols = f3.data.columns
    assert len(cols) == 13

    for col in ['new_feat1_2', 'new_feat1_1', 'new_feat2_1','new_feat2_2']:
        assert col in cols

    for col in ['feature1_1', 'feature2_1', 'feature3_1', 'feature4_1',
                'feature1_2', 'feature2_2']:
        assert col in cols

    for col in ['meta_column1', 'meta_column2']:
        assert col in cols

    assert 'outcome' in cols