from sklearn.linear_model import LinearRegression
import file_helpers as fh

def sci_lr(df):

    lr: LinearRegression = LinearRegression()

    ### sklearn requires a 2-dimensional X and 1 dimensional y. The below yeilds shapes of:
    ### skl_X = (n,1); skl_Y = (n,)
    skl_X = df[['GrLivArea']]
    skl_Y = df['SalePrice']

    lr.fit(skl_X,skl_Y)
    print("Intercept:", lr.intercept_)
    print("Coefficient:", lr.coef_)