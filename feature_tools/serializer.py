import json
import datetime as dt
from base_feature import BaseFeatures


class FeatureSerializer:
    """Serialize features and metadata"""

    def __init__(self, features: BaseFeatures, process_uuid: str, additional_meta: dict = None):
        if not isinstance(features, BaseFeatures):
            raise TypeError(f"Expected instance of BaseFeatures. Got {type(features)} instead.")

        self.features = features
        self.additional_meta = additional_meta
        self.process_uuid = process_uuid

    def _get_meta(self, subset):
        current_timestamp = str(dt.datetime.now())

        feature_cols = self.features.feature_columns
        if subset is not None:
            if not set(subset).issubset(set(feature_cols)):
                raise ValueError("Not all subset features exist in original features")
            feature_cols = subset

        meta = {
            'features': feature_cols,
            'meta_columns': self.features.meta_columns,
            'target': self.features.outcome_column,
            'process_uuid': self.process_uuid,
            'created_at': current_timestamp
        }

        if self.additional_meta is not None:
            meta = {**meta, **self.additional_meta}
        return meta

    def save(self, base_folder, meta_fn: str, data_fn: str, add_timestamp_prefix=False, subset=None):
        """
        Saves data together with metadata

        Arguments
        ---------
        base_folder (str):
        meta_fn (str): filename for metadata.
        data_fn (str): filename for the data.
        add_timestamp_prefix (bool): If true, adds current timestamp to the filename
        subset (bool or list): list of features to include to final dataset. useful after feature selection.

        Returns
        -------
        metadata - dictionary containing details of saved model.
        """

        prefix = ''
        if isinstance(add_timestamp_prefix, str):
            prefix = add_timestamp_prefix

        elif add_timestamp_prefix:
            prefix = dt.datetime.now().strftime("%Y%m%d_%H%M")

        meta_fn = f"{prefix}_{meta_fn}"
        data_fn = f"{prefix}_{data_fn}"

        meta = self._get_meta(subset)
        meta['meta_fn'] = base_folder + meta_fn
        meta['data_fn'] = base_folder + data_fn

        print("Saving metadata to {}".format(meta_fn))
        print(meta)
        with open(meta['meta_fn'], 'w') as outfile:
            json.dump(meta, outfile)

        all_columns = meta['meta_columns'] + meta['features'] + meta['target']
        print("\nSaving features to {}".format(data_fn))
        self.features.data.loc[:, all_columns].to_csv(meta['data_fn'], index=False, header=True)
        print("\nDONE.\n")
        return meta
