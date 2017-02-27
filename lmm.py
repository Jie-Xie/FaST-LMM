__author__ = 'Haohan Wang'

import scipy.optimize as opt
# import dataLoader
import time
from sklearn.linear_model import Lasso

import sys

sys.path.append('../')

from helpingMethods import *

def train(X, K, Kva, Kve, y, numintervals=100, ldeltamin=-5, ldeltamax=5, discoverNum=50, mode='linear'):
    """
    train linear mixed model lasso
    Input:
    X: Snp matrix: n_s x n_f
    y: phenotype:  n_s x 1
    K: kinship matrix: n_s x n_s
    mu: l1-penalty parameter
    numintervals: number of intervals for delta linesearch
    ldeltamin: minimal delta value (log-space)
    ldeltamax: maximal delta value (log-space)
    rho: augmented Lagrangian parameter for Lasso solver
    alpha: over-relatation parameter (typically ranges between 1.0 and 1.8) for Lasso solver
    Output:
    results
    """
    time_start = time.time()
    [n_s, n_f] = X.shape
    assert X.shape[0] == y.shape[0], 'dimensions do not match'
    assert K.shape[0] == K.shape[1], 'dimensions do not match'
    assert K.shape[0] == X.shape[0], 'dimensions do not match'
    if y.ndim == 1:
        y = scipy.reshape(y, (n_s, 1))              # Ensure it's a n_s * 1

    X0 = np.ones(len(y)).reshape(len(y), 1)

    if mode != 'linear': # LMM
        S, U, ldelta0, monitor_nm = train_nullmodel(y, K, S=Kva, U=Kve, numintervals=numintervals, ldeltamin=ldeltamin, ldeltamax=ldeltamax, mode=mode)

        delta0 = scipy.exp(ldelta0)
        # print("delta0: {0}".format(delta0))
        print("S: {0}".format(S))
        print("U: {0}".format(U))
        Sdi = 1. / (S + delta0)
        Sdi_sqrt = scipy.sqrt(Sdi)
        print("Sdi_sqrt: {0}".format(Sdi_sqrt))
        SUX = scipy.dot(U.T, X)
        SUX = SUX * scipy.tile(Sdi_sqrt, (n_f, 1)).T
        # print("SUX: {0}".format(SUX))
        SUy = scipy.dot(U.T, y)
        SUy = SUy * scipy.reshape(Sdi_sqrt, (n_s, 1))
        print("SUy: {0}".format(SUy))
        SUX0 = scipy.dot(U.T, X0)
        SUX0 = SUX0 * scipy.tile(Sdi_sqrt, (1, 1)).T
    else: # linear models
        SUX = X
        SUy = y
        ldelta0 = 0
        monitor_nm = {}
        monitor_nm['ldeltaopt'] = 0
        monitor_nm['nllopt'] = 0
        SUX0 = None

    w1 = hypothesisTest(SUX, SUy, X, SUX0, X0) # hypothesis testing case
    regs = []
    for i in range(5, 30):
        for j in range(1, 10):
            regs.append(j*np.power(10.0, -i))

    print("regs: {0}".format(regs))
    breg, w2, ss = cv_train(SUX, SUy.reshape([n_s, 2]), regMin=1e-30, regMax=1e30, K=discoverNum)
    print w2
    time_end = time.time()
    time_diff = time_end - time_start
    print '... finished in %.2fs' % (time_diff)

    res = {}
    res['ldelta0'] = ldelta0
    res['single'] = w1
    res['combine'] = w2
    res['combine_ss'] = ss
    res['combine_reg'] = regs
    res['time'] = time_diff
    res['monitor_nm'] = monitor_nm
    return res


def train_lasso(X, y, mu):
    lasso = Lasso(alpha=mu)
    lasso.fit(X, y)
    return lasso.coef_

def hypothesisTest(UX, Uy, X, UX0, X0):
    [m, n] = X.shape
    p = []
    for i in range(n): # for every SNP
    	# lmm
        if UX0 is not None:
            UXi = np.hstack([UX0 ,UX[:, i].reshape(m, 1)])
            XX = matrixMult(UXi.T, UXi)
            XX_i = linalg.pinv(XX)
            beta = matrixMult(matrixMult(XX_i, UXi.T), Uy)
            Uyr = Uy - matrixMult(UXi, beta)
            Q = np.dot( Uyr.T, Uyr)
            sigma = Q * 1.0 / m # what is sigma?
        # lasso
        else:
            Xi = np.hstack([X0 ,UX[:, i].reshape(m, 1)])
            XX = matrixMult(Xi.T, Xi)
            XX_i = linalg.pinv(XX)
            beta = matrixMult(matrixMult(XX_i, Xi.T), Uy)
            Uyr = Uy - matrixMult(Xi, beta)
            Q = np.dot(Uyr.T, Uyr)
            sigma = Q * 1.0 / m
        ts, ps = tstat(beta[1], XX_i[1, 1], sigma, 1, m)
        if -1e10 < ts < 1e10:
            p.append(ps)
        else:
            p.append(1)
    return p

def nLLeval(ldelta, Uy, S, REML=True):
    """
    evaluate the negative log likelihood of a random effects model:
    nLL = 1/2(n_s*log(2pi) + logdet(K) + 1/ss * y^T(K + deltaI)^{-1}y,
    where K = USU^T.
    Uy: transformed outcome: n_s x 1
    S:  eigenvectors of K: n_s
    ldelta: log-transformed ratio sigma_gg/sigma_ee
    """
    # print("S: {0}".format(S))
    n_s = Uy.shape[0]
    delta = scipy.exp(ldelta)

    # evaluate log determinant
    Sd = S + delta
    ldet = scipy.sum(scipy.log(Sd))

    # evaluate the variance
    Sdi = 1.0 / Sd
    # Uy_temp=Uy*Uy
    # print Uy_temp.shape()
    # Sdi1=Sdi*Uy_temp[0]
    # Sdi2=Sdi*Uy_temp[1]
    # dot_temp=[Sdi1,Sdi2]
    #dot_temp
    # print Uy
    # print Sdi
    # Uy_temp= Uy*Uy*Sdi
    # print Uy_temp
    # print type(Uy_temp)
    ss = 1. / n_s * (Uy*Uy*Sdi).sum()
    #print ss
    # ss=0.697269502547
    # evalue the negative log likelihood
    nLL = 0.5 * (n_s * scipy.log(2.0 * scipy.pi) + ldet + n_s + n_s * scipy.log(ss))

    if REML:
        pass

    return nLL

def train_nullmodel(y, K, S=None, U=None, numintervals=500, ldeltamin=-5, ldeltamax=5, scale=0, mode='lmm'):
    """
    train random effects model:
    min_{delta}  1/2(n_s*log(2pi) + logdet(K) + 1/ss * y^T(K + deltaI)^{-1}y,
    Input:
    X: Snp matrix: n_s x n_f
    y: phenotype:  n_s x 1
    K: kinship matrix: n_s x n_s
    mu: l1-penalty parameter
    numintervals: number of intervals for delta linesearch
    ldeltamin: minimal delta value (log-space)
    ldeltamax: maximal delta value (log-space)
    """
    ldeltamin += scale
    ldeltamax += scale

    if S is None or U is None:
        S, U = linalg.eigh(K)

    Uy = scipy.dot(U.T, y)
    # grid search
    nllgrid = scipy.ones(numintervals + 1) * scipy.inf
    ldeltagrid = scipy.arange(numintervals + 1) / (numintervals * 1.0) * (ldeltamax - ldeltamin) + ldeltamin
    for i in scipy.arange(numintervals + 1):
        nllgrid[i] = nLLeval(ldeltagrid[i], Uy, S)

    nllmin = nllgrid.min()
    ldeltaopt_glob = ldeltagrid[nllgrid.argmin()]

    for i in scipy.arange(numintervals - 1) + 1:
        if (nllgrid[i] < nllgrid[i - 1] and nllgrid[i] < nllgrid[i + 1]):
            ldeltaopt, nllopt, iter, funcalls = opt.brent(nLLeval, (Uy, S),
                                                          (ldeltagrid[i - 1], ldeltagrid[i], ldeltagrid[i + 1]),
                                                          full_output=True)
            if nllopt < nllmin:
                nllmin = nllopt
                ldeltaopt_glob = ldeltaopt

    monitor = {}
    monitor['ldeltaopt'] = ldeltaopt_glob
    monitor['nllopt'] = nllmin

    return S, U, ldeltaopt_glob, monitor

def cv_train(X, Y, regMin=1e-30, regMax=1.0, K=100):
    betaM = None
    breg = 0
    iteration = 0
    patience = 100
    ss = []

    while regMin < regMax and iteration < patience:
        iteration += 1
        reg = np.exp((np.log(regMin)+np.log(regMax)) / 2.0)
        # print("Iter:{}\tlambda:{}".format(iteration, lmbd), end="\t")
        clf = Lasso(alpha=reg)
        clf.fit(X, Y)
        k = len(np.where(clf.coef_ != 0)[0])
        # print reg, k
        ss.append((reg, k))
        if k < K:   # Regularizer too strong
            regMax = reg
        elif k > K: # Regularizer too weak
            regMin = reg
            betaM = clf.coef_
        else:
            betaM = clf.coef_
            break
        #print betaM
    return breg, betaM, ss # should be reg?

def run_synthetic():
    discoverNum = 50
    numintervals = 500
    # snps, Y, Kva, Kve, causal = dataLoader.load_data_synthetic() # this is just an example, write your own loading method. refer to numpy.loadtxt or numpy.load
#     snps = np.array([0.8147,    0.1576,    0.6557,    0.7060,    0.4387,
# 0.9058,    0.9706,    0.0357,    0.0318,    0.3816,
# 0.1270,    0.9572,    0.8491,    0.2769,    0.7655,
# 0.9134,    0.4854,    0.9340,    0.0462,    0.7952,
# 0.6324,    0.8003,    0.6787,    0.0971,    0.1869,
# 0.0975,    0.1419,    0.7577,    0.8235,    0.4898,
# 0.2785,    0.4218,    0.7431,    0.6948,    0.4456,
# 0.5469,    0.9157,    0.3922,    0.3171,    0.6463,
# 0.9575,    0.7922,    0.6555,    0.9502,    0.7094,
# 0.9649,    0.9595,    0.1712,    0.0344,    0.7547]).reshape(10, 5)
#     Y = np.array([0.4173,
# 0.0497,
# 0.9027,
# 0.9448,
# 0.4909,
# 0.4893,
# 0.3377,
# 0.9001,
# 0.3692,
# 0.1112]).reshape(10, 1)
    snps = np.array([[0, 1], [2, 1]])
    Y = np.array([5, 1]).reshape(2, 1)
    Kva = None
    Kve = None
    K = np.dot(snps, snps.T)
    n_s, n_f = snps.shape

    # TEST BEGINS

    # note S from linalg.eigh(K) is a row vector!
    # S, U = linalg.eigh(K)
    # S = S.reshape(S.size(), 1)
    S = np.array([3, 1]).reshape(2, 1)
    U = np.array([[1, 0], [1, 1]])
    # print("U: {0}".format(U))
    # print("S: {0}".format(S))
    ldelta = 1
    # test_f(Y, S, U, ldelta)
    # test_train_nullmodel(Y, U, S, numintervals=500, ldeltamin=-5, ldeltamax=5)
    SUX, SUy, SUX0 = test_train_params(snps, Y, S, U, numintervals=500, ldeltamin=-5, ldeltamax=5)
    p = test_hypo(SUX, SUy, snps, SUX0)
    test_cv(SUX, SUy)

    # TEST ENDS

    # res = train(X = snps, K=K, y=Y, Kva=Kva, Kve=Kve, numintervals=numintervals, ldeltamin=-5, ldeltamax=5, discoverNum=discoverNum, mode='lmm')
    # print res['ldelta0'], res['monitor_nm']['nllopt']

    # # hypothesis weights
    # result_hypo = np.array(res['single'])

    # # lasso weights
    # result_lasso = res['combine']

# jiexie
def test_f(y, S, U, ldelta):
    Uy = scipy.dot(U.T, y)
    result = nLLeval(ldelta, Uy, S)
    print "f value: {0}".format(result)

# jiexie
def test_train_nullmodel(y, U, S, numintervals=500, ldeltamin=-5, ldeltamax=5):
    Uy = scipy.dot(U.T, y)
    # grid search
    nllgrid = scipy.ones(numintervals + 1) * scipy.inf
    ldeltagrid = scipy.arange(numintervals + 1) / (numintervals * 1.0) * (ldeltamax - ldeltamin) + ldeltamin
    for i in scipy.arange(numintervals + 1):
        nllgrid[i] = nLLeval(ldeltagrid[i], Uy, S)

    nllmin = nllgrid.min()
    ldeltaopt_glob = ldeltagrid[nllgrid.argmin()]
    # jiexie
    print("nllmin: {0}".format(nllmin))
    print("ldeltaopt_glob: {0}".format(ldeltaopt_glob))
    return (ldeltaopt_glob, nllmin)

# jiexie
def test_train_params(X, y, S, U, numintervals, ldeltamin, ldeltamax):
    [n_s, n_f] = X.shape
    if y.ndim == 1:
        y = scipy.reshape(y, (n_s, 1))              # Ensure it's a n_s * 1

    X0 = np.ones(len(y)).reshape(len(y), 1)

    ldelta0, nllmin = test_train_nullmodel(y, U, S, numintervals, ldeltamin, ldeltamax)

    delta0 = scipy.exp(ldelta0)
    Sdi = 1. / (S + delta0)
    Sdi_sqrt = scipy.sqrt(Sdi)
    print("Sdi_sqrt: {0}".format(Sdi_sqrt))
    SUX = scipy.dot(U.T, X)
    SUX = SUX * scipy.tile(Sdi_sqrt, (1, n_f))
    print("SUX: {0}".format(SUX))
    SUy = scipy.dot(U.T, y)
    SUy = SUy * Sdi_sqrt
    print("SUy: {0}".format(SUy))
    SUX0 = scipy.dot(U.T, X0)
    SUX0 = SUX0 * Sdi_sqrt
    print("SUX0: {0}".format(SUX0))
    return SUX, SUy, SUX0
# jiexie
def test_hypo(UX, Uy, X, UX0):
    [m, n] = X.shape
    p = []
    for i in range(n): # for every SNP
        UXi = np.hstack([UX0 ,UX[:, i].reshape(m, 1)])
        XX = matrixMult(UXi.T, UXi) # must be 2*2
        XX_i = linalg.pinv(XX) # XX_i[1, 1] is variance of beta
        beta = matrixMult(matrixMult(XX_i, UXi.T), Uy) # must be 2*1, beta[1] is beta for feature i
        Uyr = Uy - matrixMult(UXi, beta)
        Q = np.dot(Uyr.T, Uyr)
        sigma = Q * 1.0 / m # genetic variance sigma_g

        ts, ps = tstat(beta[1], XX_i[1, 1], sigma, 1, m)
        if -1e10 < ts < 1e10:
            p.append(ps)
        else:
            p.append(1)
    print("p: {0}".format(p))
    return p

# working
def test_cv(X, Y, regMin=1e-30, regMax=1.0, K=1):
    # print("X: {0}".format(X))
    # print("Y: {0}".format(Y))
    betaM = None
    breg = 0
    iteration = 0
    patience = 100
    ss = []
    while regMin < regMax and iteration < patience:
        iteration += 1
        reg = np.exp((np.log(regMin)+np.log(regMax)) / 2.0)
        # print("Iter:{}\tlambda:{}".format(iteration, lmbd), end="\t")
        clf = Lasso(alpha=reg)
        clf.fit(X, Y)
        k = len(np.where(clf.coef_ != 0)[0])
        # print reg, k
        ss.append((reg, k))
        if k < K:   # Regularizer too strong
            regMax = reg
        elif k > K: # Regularizer too weak
            regMin = reg
            betaM = clf.coef_
        else:
            betaM = clf.coef_
            break
        #print betaM
    print("breg: {0}".format(breg))
    print("betaM: {0}".format(betaM))
    print("ss: {0}".format(ss))
    return breg, betaM, ss # should be reg?

if __name__ == '__main__':
    run_synthetic()
