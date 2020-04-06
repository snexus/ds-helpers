import pandas as pd
from contextlib import contextmanager


class BaseFeatures:
    """Base Features Class"""

    def __init__(self, data, feature_columns, feature_suffix="", meta_columns=False, parse_dates=False,
                 exclude_columns=False):
        """

        Arguments
        ---------
        data (str or Pandas DataFrame): file name pointing to data, or pd.DataFrame.
        feature_columns (list): list of columns to be considered as features.
        feature_suffix (str): suffix to add to all feature.
        meta_columns (list): list of columns to be considered as meta-columns.
        parse_dates (bool or list): if list, specifyis columns that will be converted to pd.timestamp.
        exclude_columns (bool or list): if list, specifies columns to be excluded from the list.
        """

        self._base_feature_columns = feature_columns
        self.feature_suffix = feature_suffix
        self.meta_columns = meta_columns or []
        self.exclude_columns = exclude_columns or []
        self._outcome_column = None
        self.custom_transformer = None

        self.parse_dates = parse_dates
        self._additional_features = []

        self._all_data = self._load_validate(data)
        self._clean_data = None

    def _create_outcome_column(self):
        """
        Creates outcome column by called create_target from inherited class.
        It is expected that created_target() returns name of the target column

        Raises:
        ------

        KeyError - if returned column not in the data
        TypeError - if returned column name is not of type string
        """

        if self._outcome_column is None:
            self._outcome_column = []
            col = self.create_target()

            if col is not None:
                if not isinstance(col, str):
                    raise TypeError("Outcome column should be string or None.")

                if col not in self._all_data.columns:
                    raise KeyError(f"Couldn't find {col} in the data")

                self._outcome_column = [col]
        return self._outcome_column

    def set_custom_transformer(self, fn):
        """Sets custom transformer

        Arguments:
        fn (function) - function to use as custom transformer
        """

        self.custom_transformer = fn
        self._clean_data = None  # Invalidate cache

    def _invoke_custom_transformer(self):
        """
        Invokes custom transformer, passing raw data to the transformer
        """

        with self.data_accessor() as data:
            prev_features = data.columns
            feature_names = self.custom_transformer(data)

            if feature_names is not None:
                if not isinstance(feature_names, list):
                    raise TypeError("Customer transformer should return list of new feature names")

                self._check_columns(data, feature_names)
                self._additional_features += feature_names

            new_features = data.columns
            added = set(new_features) - set(prev_features)

            if len(added) > 0:
                print(f"New transformed features detected in DataFrame: {added}")
                print(f"Registered features: {feature_names}\n")

    @property
    def outcome_column(self):
        return self._create_outcome_column()

    @property
    def data(self):
        """Property to access clean dataset containing only necessary columns.

        Following set of operations performed:
        1) Transform data
        2) Impute data
        3) Create target column
        4) Add suffix to the feature column (if necessary)
        5) Extract meta_columns + feature_columns + outcome_column
        """

        if self._clean_data is None:

            if self.custom_transformer is None:
                self.transform()
            else:
                self._invoke_custom_transformer()

            self.impute()
            self._create_outcome_column()

            # Feature name transformation to add suffix
            temp_data = self._all_data.copy()
            suffix_mapper = {feat: feat_with_suffix for feat, feat_with_suffix in
                             zip(self.raw_feature_columns, self.feature_columns)}
            temp_data = temp_data.rename(columns=suffix_mapper)

            if len(self._additional_features) == 0:
                print("Warning - no additional features exists. Perhaps transformation wasn't properly applied?")

            self._clean_data = temp_data.loc[:, self.meta_columns + self.feature_columns + self.outcome_column]

        return self._clean_data

    @property
    def raw_feature_columns(self):
        """
        Returns feature columns by combining base features and additional features and removing exclusions
        """

        all_features = self._base_feature_columns + self._additional_features

        if self.exclude_columns:
            all_features = [feature for feature in all_features if feature not in self.exclude_columns]
        return all_features

    @property
    def feature_columns(self) -> list:
        """
        Returns list of feature names
        """

        all_features = self.raw_feature_columns
        if self.feature_suffix:
            all_features = [feature + self.feature_suffix for feature in all_features]

        return all_features

    def _load_validate(self, data_obj) -> pd.DataFrame:
        """ Loads and validates data.

        Arguments:
        ---------
        data_obj (str or pd.DataFrame): data object. If string, filename is assumed, data will be
         loaded from this location.

        Returns:
        -------
        Pandas DataFrame, if all checks passed.

        Raises:
        ------
        TypeError - if data is not a string or pd.DataFrame.
        KeyError - if feature columns or data columns don't exist.
        """

        if isinstance(data_obj, str):
            data = pd.read_csv(data_obj, parse_dates=self.parse_dates)
        elif isinstance(data_obj, pd.DataFrame):
            data = data_obj.copy()
        else:
            raise TypeError("Data should be either string or pd.DataFrame.")

        self._check_columns(data, self._base_feature_columns)
        self._check_columns(data, self.meta_columns)
        return data

    def _check_columns(self, df: pd.DataFrame, col_list: list, additional_str=""):
        """
        Checks if columns specified in col_list exist in dataframe
        """

        df_cols = df.columns
        for col in col_list:
            if col not in df_cols:
                print("Available colunms:", df_cols)
                raise KeyError(f"Column '{col}' doesn't exist in data. {additional_str}")

    def __str__(self):
        return f"Dataset size: {len(self.data)}, Base features: {len(self._base_feature_columns)}, " \
               f"Transformed features: {len(self._additional_features)}, " \
               f"Metadata columns: {len(self.meta_columns)}, Outcome: {len(self.outcome_column)}"

    @contextmanager
    def data_accessor(self):
        """
        Context manager to make convinient copy of data.
        """

        data = self._all_data.copy()
        try:
            yield data

        finally:
            self._all_data = data

    def __add__(self, other):
        """
        Class addition implementation
        """

        if not isinstance(other, BaseFeatures):
            raise TypeError("Incompatible types. Both object should inherit from BaseFeatures")

        if len(self.data) != len(other.data):
            print("Warning: features for differently sized dataset. Performing inner join.")

        if set(self.meta_columns) != set(other.meta_columns):
            raise TypeError("Meta-columns don't match. Can't add two objects.")

        # print(set(self.feature_columns), set(other.feature_columns))
        if set(self.feature_columns) - set(other.feature_columns) != set(self.feature_columns):
            raise KeyError("Duplicate feature columns. Consider adding suffix.")

        new_feature_columns = self.feature_columns + other.feature_columns
        # print(new_feature_columns, self.meta_columns + self.outcome_column, other.data)
        data = self.data.merge(other.data, on=self.meta_columns + self.outcome_column, how="inner")

#         if len(data)!=len(self.data):
#           raise RuntimeError("Join didn't succeed. Length of data after combinig different from original data.")

        obj = BaseFeatures(data=data, feature_columns=new_feature_columns, meta_columns=self.meta_columns)
        obj._outcome_column = self._outcome_column.copy()  # Monkey patch to assign proper target
        return obj

    def transform(self):
        pass

    def impute(self):
        pass

    def create_target(self):
        pass
