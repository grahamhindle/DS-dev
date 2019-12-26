
import numpy as np
import matplotlib.pyplot as plt
import file_helpers as fh
import linear_regression as lr
import lr_scikit as lrs

tr_path = '../data/train.csv'
test_path = '../data/test.csv'

df = fh.read_to_df(tr_path)

df_sub = fh.select_columns(df, ['SalePrice', 'GrLivArea', 'YearBuilt'])

cutoffs = [('SalePrice', 50000, 1e10), ('GrLivArea', 0, 4000)]
df_sub_cutoff = fh.column_cutoff(df_sub, cutoffs)

X = df_sub_cutoff['GrLivArea'].values
Y = df_sub_cutoff['SalePrice'].values

# reshaping for input into function
training_y = np.array([Y])
training_x = np.array([X])

weights = lr.least_squares_weights(training_x, training_y)
print(weights)

max_X = np.max(X) + 500
min_X = np.min(X) - 500

# Choose points evenly spaced between min_x in max_x
reg_x = np.linspace(min_X, max_X, 1000)

# Use the equation for our line to calculate y values
reg_y = weights[0][0] + weights[1][0] * reg_x

plt.plot(reg_x, reg_y, color='#58b970', label='Regression Line')
plt.scatter(X, Y, c='k', label='Data')

plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.legend()
plt.show()

# calculate rmse
rmse = 0

b0 = weights[0][0]
b1 = weights[1][0]

for i in range(len(Y)):
    y_pred = b0 + b1 * X[i]
    rmse += (Y[i] - y_pred) ** 2
rmse = np.sqrt(rmse/len(Y))
print(rmse)

# calculate R^^2
ss_t = 0
ss_r = 0

mean_y = np.mean(Y)

for i in range(len(Y)):
    y_pred = b0 + b1 * X[i]
    ss_t += (Y[i] - mean_y) ** 2
    ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r/ss_t)

print(r2)

lrs.sci_lr(df_sub_cutoff)
