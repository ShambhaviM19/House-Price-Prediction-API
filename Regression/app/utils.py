from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
class Preprocessor(BaseEstimator, TransformerMixin):
    #Train our custom preprocessors
    def fit(self,X,y=None):
        self.imputer=SimpleImputer()
        self.imputer.fit(X[['HouseAge','DistanceToStation','NumberOfPubs']])

        self.scaler=StandardScaler()
        self.scaler.fit(X[['HouseAge','DistanceToStation','NumberOfPubs']])

        self.onehot=OneHotEncoder(handle_unknown='ignore')
        self.onehot.fit(X[['PostCode']])
        return self

    #Apply our custom preprocessors
    def transform(self,X):
        imputed_cols=self.imputer.transform(X[['HouseAge','DistanceToStation','NumberOfPubs']]) 
        onehot_cols=self.onehot.transform(X[['PostCode']])

        transformed_df=X.copy()

        transformed_df['Year']=transformed_df['TransactionDate'].apply(lambda x: x[:4]).astype(int)
        transformed_df['Month']=transformed_df['TransactionDate'].apply(lambda x: x[5:]).astype(int)
        transformed_df=transformed_df.drop('TransactionDate',axis=1)

        transformed_df[['HouseAge','DistanceToStation','NumberOfPubs']]=imputed_cols
        transformed_df[['HouseAge','DistanceToStation','NumberOfPubs']]=self.scaler.transform(transformed_df[['HouseAge','DistanceToStation','NumberOfPubs']])

        transformed_df=transformed_df.drop('PostCode',axis=1)
        transformed_df[self.onehot.get_feature_names_out()]=onehot_cols.toarray().astype(int)

        return transformed_df