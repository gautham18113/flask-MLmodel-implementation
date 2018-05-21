from sklearn.base import BaseEstimator, TransformerMixin


class PreProcessing(BaseEstimator, TransformerMixin):
    """Custom Pre-Processing estimator for this use-case
    """
    def __init_(self):
        pass

    def transform(self, df):
        pred_var = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
                    'CoapplicantIncome','LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']

        df = df[pred_var]
        df['Dependents'] = df['Dependents'].fillna(0)
        df['Self_Employed'] = df['Self_Employed'].fillna('No')
        df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(self.term_mean_)
        df['Credit_History'] = df['Credit_History'].fillna(1)
        df['Married'] = df['Married'].fillna('No')
        df['Gender'] = df['Gender'].fillna('Male')
        df['LoanAmount'] = df['LoanAmount'].fillna(self.amt_mean_)
        gender_values = {'Female': 0, 'Male': 1}
        married_values = {'No': 0, 'Yes': 1}
        education_values = {'Graduate': 0, 'Not Graduate': 1}
        employed_values = {'No': 0, 'Yes': 1}
        property_values = {'Rural': 0, 'Urban': 1, 'Semiurban': 2}
        dependent_values = {'3+': 3, '0': 0, '2': 2, '1': 1}
        df.replace({'Gender': gender_values, 'Married': married_values, 'Education': education_values,
                    'Self_Employed': employed_values, 'Property_Area': property_values,
                    'Dependents': dependent_values}, inplace=True)

        return df.as_matrix()

    def fit(self, df, y=None, **fit_params):
        """Fitting the Training dataset and calculating required values
        from train
        """
        self.term_mean_ = df['Loan_Amount_Term'].mean()
        self.amt_mean_ = df['LoanAmount'].mean()
        return self

