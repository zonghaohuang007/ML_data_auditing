# from https://github.com/WannabeSmith/confseq_wor/tree/main/cswor

import math
import numpy as np
from scipy.special import beta, betaln, comb, gammaln

# Probability mass functions

def beta_binomial_pmf(k, n, a, b, log_scale=False):
    '''
    The pmf of a beta-binomial distribution.

    Parameters
    ----------
    k, integer
        number of "successes".
    n, integer
        number of draws.
    a, positive real
        alpha parameter for the beta-binomial.
    b, positive real
        beta parameter for the beta-binomial.
    log_scale, boolean
        should the pmf be returned on the log scale?

    Returns
    -------
    prob, array-like of positive reals
        probability mass of k
    '''
    if log_scale:
        return (np.log(comb(n, k)) +
                np.log(beta(k + a, n - k + b)) -
                np.log(beta(a, b)))
    else:
        return (comb(n, k) *
                beta(k + a, n - k + b) /
                beta(a, b))


def dirichlet_multinomial_pmf(x, n, a, log_scale=False):
    '''
    The pmf of a Dirichlet-multinomial distribution.

    Parameters
    ----------  
    x, array-like of integers
        number of "successes". This can be a list of integers 
        if this is just one set of observations, or it can be 
        a KxS list of lists or numpy array if there are K 
        categories and S samples.
    n, array-like of integers
        number of draws.
    a, array-like of positive reals
        alpha parameters for the Dirichlet multinomial.
        This can be a K-list of alpha values if we have one 
        set of observations, or it can be KxS list of lists 
        or numpy array if there are K categories and S samples.
    log_scale, boolean
        should the pmf be returned on the log scale?

    Returns
    -------
    prob, array-like of positive reals
        probability mass of k
    '''
    np.seterr(divide='ignore', invalid='ignore')
    x = np.array(x)
    a = np.array(a)
    n = np.array(n)

    # x and a must have the same dimensions.
    assert(np.shape(x) == np.shape(a))

    sum_a = a.sum(axis=0)

    # 1 if the observation was valid (i.e. all x >= 0) and 0 otherwise
    valid_obs = (x >= 0).prod(axis=0)

    if log_scale:
        log_constant = np.log(n) + betaln(sum_a, n)
        # In product in pmf, we want to include only x's that are non-zero.
        summands = np.log(x) + betaln(a, x)
        summands[x == 0] = 0
        # \log (\prod_{k : x_k > 0} x_k B(alpha_k, x_k))
        log_product = summands.sum(axis=0)

        # turn back into np array just in case it was cast to float
        log_pmf = np.array(log_constant - log_product)
        # if the observation was invalid, it has a probability of 0,
        # thus log-prob of -inf
        log_pmf[valid_obs == 0] = -math.inf
        # if there are "no draws" (i.e. everything already observed)
        # then the pmf is 1 iff all of the x's are 0.
        # This is just by convention, since n = 0 is technically
        # an invalid input to the pmf.
        # sum_of_abs_x is 0 iff all x are 0
        sum_of_abs_x = np.abs(x).sum(axis=0)
        log_pmf[n == 0] = np.log(sum_of_abs_x[n == 0] == 0)
        return log_pmf
    else:
        constant = n * beta(sum_a, n)
        # In product in pmf, we want to include only x's that are non-zero.
        multiplicands = x * beta(a, x)
        multiplicands[x == 0] = 1
        # \prod_{k : x_k > 0} x_k B(alpha_k, x_k)
        product = multiplicands.prod(axis=0)

        # turn back into np array just in case it was cast to float
        pmf = np.array(constant / product)
        # if the observation was invalid, it has a probability of 0
        pmf[valid_obs == 0] = 0

        # if there are "no draws" (i.e. everything already observed)
        # then the pmf is 1 iff all of the x's are 0.
        # This is just by convention, since n = 0 is technically
        # an invalid input to the pmf.
        # sum_of_abs_x is 0 iff all x are 0
        sum_of_abs_x = np.abs(x).sum(axis=0)
        pmf[n == 0] = sum_of_abs_x[n == 0] == 0
        # Get an array back from the 1xL matrix where L is the length of x
        return np.squeeze(np.multiply(constant, product))


def mvt_hypergeo_pmf(k, T, log_scale=False):
    '''
    The pmf of a multivariate hypergeometric distribution.

    Parameters
    ----------
    k, array-like of integers
        number of "successes". This can be a list of integers
        if this is just one set of observations, or it can be
        a KxS list of lists or numpy array if there are K
        categories and S samples.
    T, integer
        true number of "successes"
    log_scale, boolean
        should the pmf be returned on the log scale?

    Returns
    -------
    prob, array-like of positive reals
        probability mass of k
    '''
    k = np.array(k)
    T = np.array(T)
    assert(np.shape(k) == np.shape(T))

    # Turn into column vector if only one observation
    k = k[:, None] if np.ndim(k) == 1 else k
    T = T[:, None] if np.ndim(T) == 1 else T

    # Total balls in urn
    N = T.sum(axis=0)
    # Total number of balls sampled from urn
    n = k.sum(axis=0)

    valid_inputs = np.logical_and(k >= 0, T - k >= 0).prod(axis=0)

    if log_scale:
        log_numerator = (gammaln(T + 1) -
                         gammaln(T - k + 1) -
                         gammaln(k + 1)).sum(axis=0)
        log_denominator = gammaln(N + 1) -\
            gammaln(N - n + 1) -\
            gammaln(n + 1)

        log_pmf = log_numerator - log_denominator
        log_pmf[np.logical_not(valid_inputs)] = -math.inf
        return log_pmf
    else:
        numerator = comb(T, k).prod(axis=0)
        denominator = comb(N, n)
        pmf = numerator / denominator
        pmf[np.logical_not(valid_inputs)] = 0
        return pmf


def beta_pdf(x, a, b, log_scale=False):
    x = np.array(x)
    a = np.array(a)
    b = np.array(b)

    if log_scale:
        log_numer = (a - 1) * np.log(x) +\
            (b - 1) * np.log(1 - x)
        log_denom = betaln(a, b)
        return(log_numer - log_denom)
    else:
        numer = np.power(x, a - 1) *\
            np.power(1 - x, b - 1)
        denom = beta(a, b)
        return(numer/denom)

def pval_from_martingale(martingale, running_min=False):
    '''
    Get a p-value from a martingale with initial expectation 1

    Parameters
    ----------
    martingale, array-like
        the martingale
    running_min, boolean
        should the running minimum be taken?


    Returns
    -------
    p, array-like
        p-values
    '''
    pval = np.minimum(1/martingale, 1)
    pval = np.minimum.accumulate(pval) if running_min else pval

    return pval