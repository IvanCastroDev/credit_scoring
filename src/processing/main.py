import pandas as pd
from typing import Tuple, Any
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class CreditDataProcessor:
    def __init__(self):
        self.numerical_features = ['Age', 'Job', 'Credit amount', 'Duration']
        self.categorical_features = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
        self.targe_feature = 'Risk'

    def fit_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        """
        Build a pipeline for preprocessing the data
        1.- Scale numerical features (StandardScaler)
        2.- Codify categorical features (OneHotEncoder)
        """

        numeric_tf = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_tf = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_tf, self.numerical_features),
                ('cat', categorical_tf, self.categorical_features)
            ],
            remainder='passthrough'
        )

        x_train = df.drop(self.targe_feature, axis=1)
        preprocessor.fit(x_train)
        return preprocessor
    
    def process_data(self, df: pd.DataFrame, preprocessor: ColumnTransformer):
        """
        Apply processing and separate features
        Args:
            df (pd.DataFrame): Dataframe to process
            preprocessor (ColumnTransformer): Fit preprocessing
        """

        df_copy = df.copy()
        df_copy[self.targe_feature] = df_copy[self.targe_feature].map({'bad': 0, 'good': 1})

        y = df_copy[self.targe_feature]
        x = df_copy.drop(self.targe_feature, axis=1)

        x_processed = preprocessor.transform(x)

        return x_processed, y