import numpy as np
import matplotlib.pyplot as plt
import ising as ising
from sklearn import linear_model
from sklearn.model_selection import train_test_split

#------------------#
# DATA PREPARATION #
#------------------#

# Ising states parameters
nsample = 1000   # number of data samples
nsites = 20       # number of ising sites
rndseed = 12      # random number generator seed

# generate random ising configurations
ising_states = ising.generate_states(nsample, nsites, rand_seed=rndseed)
# generate ising interaction matrix
ising_intmat = ising.interaction_matrix(ising_states)
# calculate total energy for ising states (y)
energy = ising.total_energy(ising_states)


# prepare data for model training
# features (X) : pair interactions for each configuration with size (nsample, nsite*nsite)
# labels (y) : total energy for each configuration (nsample, 1)
X = ising_intmat.reshape(1, -1, nsample).swapaxes(0, 2).squeeze()
y = energy

#--------------------------------#
# MODEL SELECTION AND EVALUATION #
#--------------------------------#

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.33, random_state=rndseed)

# instantiate models
linreg_ols = linear_model.LinearRegression()    # ordinary least square regression
linreg_rdg = linear_model.Ridge()               # Ridge regression
linreg_las = linear_model.Lasso()               # Lasso regression

# train old model
linreg_ols.fit(X_train, y_train)
err_train_ols = linreg_ols.score(X_train, y_train)
err_test_ols = linreg_ols.score(X_test, y_test)
coef_ols = linreg_ols.coef_

# train ridge and lasso models
err_train_las = []
err_train_reg = []
err_test_las = []
err_test_reg = []
coef_las = []
coef_reg = []
regpara = np.logspace(-4, 5, 10)    # regularization parameter

for alpha in regpara:   
    # ridge
    linreg_rdg.set_params(alpha=alpha)
    linreg_rdg.fit(X_train, y_train)
    # errors and coeffs
    err_train_reg.append(linreg_rdg.score(X_train, y_train))
    err_test_reg.append(linreg_rdg.score(X_test, y_test))    
    coef_reg.append(linreg_rdg.coef_)

    # lasso
    linreg_las.set_params(alpha=alpha)
    linreg_las.fit(X_train, y_train)
    # errors and coeffs
    err_train_las.append(linreg_las.score(X_train, y_train))
    err_test_las.append(linreg_las.score(X_test, y_test))
    coef_las.append(linreg_las.coef_)


#-----------------------#
# ANALYZE MODEL OUTPUTS #
#-----------------------#

# plot train and test errors
plt.hlines(err_train_ols, min(regpara), max(regpara), colors='k', linestyles='dotted', label='Train OLS')
plt.hlines(err_test_ols, min(regpara), max(regpara), colors='k', linestyles='solid', label='Test OLS')
plt.semilogx(regpara, err_train_reg, '--ro', label="Train Ridge")
plt.semilogx(regpara, err_test_reg, 'ro', label="Test Ridge")
plt.semilogx(regpara, err_train_las, '--bo', label="Train Lasso")
plt.semilogx(regpara, err_test_las, 'bo', label="Test Lasso")
plt.xlim(min(regpara), max(regpara))
plt.xlabel("Regularization parameter")
plt.ylabel("Model score")
plt.legend()

# linear regression coefficients (interaction strength of ising pairs)
npara = regpara.size
fig, ax = plt.subplots(nrows=npara, ncols=3)

coef_reg = np.array(coef_reg).reshape(npara, nsites, nsites)
coef_las = np.array(coef_las).reshape(npara, nsites, nsites)
coef_ols = coef_ols.reshape(nsites, nsites)

for iax in range(npara):
    # Ridge
    ax[iax, 0].imshow(coef_reg[iax, :, :])
    # Lasso
    ax[iax, 1].imshow(coef_las[iax, :, :])
    # Ols
    ax[iax, 2].imshow(coef_ols)

plt.show()


# train and test dataset scores
# print("linreg_ols train set score:{:2.2f}".format(linreg_ols.score(X_train, y_train)))
# print("linreg_ols test set score:{:2.2f}".format(linreg_ols.score(X_test, y_test)))

# linear regression coefficients (interaction strength of ising pairs)
# coef_ols = linreg_ols.coef_
# coef_ols = coef_ols.reshape(nsites, nsites)

# imgplt = plt.imshow(coef_ols)
# imgplt.set_clim(-1, 1)
# plt.colorbar()
# plt.show()

# new sample
# new_state = ising.generate_states(1,nsites)
# new_energy = ising.total_energy(new_state)
# new_X = new_state.reshape(nsites, 1) * new_state
# new_X = new_X.reshape(1,-1)
# linreg_ols.predict(new_X)

# paper example
# X_train_exmpl = X[:400, :]
# y_train_exmpl = y[:400]
# X_test_exmpl = X[400:3*400//2, :]
# y_test_exmpl = y[400:3*400//2]

# linreg_ols_exmpl = linear_model.LinearRegression()
# linreg_ols_exmpl.fit(X_train_exmpl, y_train_exmpl)
# linreg_ols_exmpl.score(X_train_exmpl, y_train_exmpl)
# linreg_ols_exmpl.score(X_test_exmpl, y_test_exmpl)


# print scores
# print("model 1 train score: {:4.2f}".format(linreg_ols.score(X_train, y_train)))
# print("model 1 test score: {:4.2f}".format(linreg_ols.score(X_test, y_test)))
# print("model 2 train score: {:4.2f}".format(linreg_ols_exmpl.score(X_train_exmpl, y_train_exmpl)))
# print("model 2 test score: {:4.2f}".format(linreg_ols_exmpl.score(X_test_exmpl, y_test_exmpl)))
# 