# from https://github.com/WannabeSmith/confseq_wor/tree/main/cswor

import numpy as np
from utils import *

def BBWR_martingale_twosided(x, B_alpha, B_beta, T_null):
    '''
    Beta-binomial with-replacement martingale with 2 categories

    Parameters
    ----------
    x, matrix-like
        2 by S matrix where S is the number of samples taken.
        Entries are the number of balls of color k in sample s,
        k in {1, ..., K}, s in {1, ... , S}. A list of values 
        for a single observation is also acceptable.
    B_alpha, positive real
        alpha parameter for beta-binomial prior
    B_beta, positive real
        beta parameter for beta-binomial prior
    T_null, list of integers
        null vector for number of balls of colors 1 through K 
        in the urn.

    Returns
    -------
    mart_vec, array
        martingale as balls are sampled from the urn
    '''
    # put into numpy column vector format
    T_null = np.array(T_null)[:, None]

    # Make x a numpy array
    x = np.array(x)

    # if there's just a single observation, turn into column vector
    if x.ndim == 1:
        x = x[:, None]

    # Number of groups (categories)
    K = np.shape(x)[0]
    assert(K == 2)

    # Total number of balls in urn 
    N = int(np.sum(T_null))

    # Cumulative sum
    S_t = np.cumsum(x[0, :])
    t = np.arange(1, len(S_t) + 1)
    
    # Two-sided BRAVO is just a prior-posterior Beta-Bernoulli
    # martingale with a uniform prior.
    prior = beta_pdf(T_null[0] / N, B_alpha, B_beta)
    posterior = beta_pdf(T_null[0] / N, B_alpha + S_t, B_beta + t - S_t)

    return prior/posterior



def BBWR_confseq(x, n, N, B_alpha, B_beta, alpha=0.05,
                 running_intersection=False):
    '''
    Confidence sequence for the number of balls in an urn
    of a particular color based on the with-replacement 
    beta-binomial martingale

    Parameters
    ----------
    x, array-like
        array of observations indicating balls of color 1.
        A single value is also acceptable
    n, array-like
        number of samples at each time
    N, integer
        number of total balls in the urn 
    B_alpha, positive real
        alpha parameter for beta-binomial
    B_beta, positive real
        beta parameter for beta-binomial
    alpha, positive real
        confidence level
    running_intersection, boolean
        should the running intersection of the confidence sequence be taken?

    Returns
    -------
    CIs_lower, list of positive reals
        lower part of CIs
    CIs_upper, list of positive reals
        upper part of CIs
    '''
    # We need x and n to be numpy arrays
    x = np.array(x)
    n = np.array(n)

    # cumulative sum
    S_t = np.cumsum(x)
    # intrinsic time
    t = np.cumsum(n)

    T_hats = N * S_t / t
    CIs_lower = np.zeros(len(x))
    CIs_upper = np.repeat(N, len(x))

    overparam_x = np.vstack((x, n - x))
    # All possible values of T
    possible_T = np.arange(0, N+1)
    confseq_mtx = np.zeros((len(possible_T), len(x)))
    # TODO: This can be sped up by doing a binary search, effectively
    # converting from O(N) time at each step to O(log N) time. However,
    # this is already quite fast for real-world use, so we'll leave it
    # as-is until we have free time to speed up.
    for i in np.arange(0, len(possible_T)):
        T = possible_T[i]

        confseq_mtx[i, :] = \
            BBWR_martingale_twosided(overparam_x, 
                                     B_alpha, B_beta, 
                                     [T, N-T]) <= 1/alpha
        # if 0 (the boundary) didn't reject, include it in the CI

    lower = np.zeros(len(x))
    upper = np.repeat(N, len(x))
    for j in np.arange(0, len(x)):
        where_in_cs = np.where(confseq_mtx[:, j])
        lower[j] = possible_T[where_in_cs[0][0]]
        upper[j] = possible_T[where_in_cs[0][-1]]

    lower = np.maximum.accumulate(lower) if running_intersection else lower
    upper = np.minimum.accumulate(upper) if running_intersection else upper 
    return lower, upper 

def hoeffding_ptwise(x, alpha):
    x = np.array(x)
    t = np.arange(1, len(x)+1)
    margin = np.sqrt(0.5/t * np.log(2/alpha))

    mu_hat_t = np.cumsum(x)/t

    return mu_hat_t-margin, mu_hat_t+margin


def corollary1_prob_GW(eps, n, N, sigma2):
    eps=np.array(eps)
    n=np.array(n)
    f_n = (n-1)/(N-1)
    prob=np.exp(-n*np.power(eps, 2) / (sigma2 * 2 * (1-f_n) + eps/3))
    return(prob)

def hypergeo2_prob_BM(eps, n, N):
    n=np.array(n)
    return(np.exp(-2*n*np.power(eps,2)/((1-n/N)*(1+1/n))))

def prop3_1_prob_WSR(eps, n, N):
    i = np.arange(1, n)
    W_n_minus_1 = np.sum(i/(N-i))
    
    prob = np.exp(-2*np.power(eps * (n+W_n_minus_1), 2)/n)
    return(prob)

def theorem3_prob_GW(eps, t, N, mu_null):
    t = np.array(t)
    eps = np.array(eps)

    assert(len(t) == 1 or len(eps) == 1)
    
    h = lambda y : y*(np.log(y) - 1) + 1
    psi = lambda y : (2/np.power(y,2)) * h(1+y)
    f_t = (t-1) / (N-1)
    sigma2 = mu_null*(1-mu_null)
    prob = np.exp(- (np.power(eps,2) * t) / (2 * sigma2 * (1-f_t)) *\
                  psi(eps * np.sqrt(t) / (np.sqrt(t) * sigma2 * (1-f_t))))
    
    return prob

def theorem3_5_prob_BM(eps, t, n, N, mu_null, delta):
    t = np.array(t)
    eps = np.array(eps)

    assert(len(t) == 1 or len(eps) == 1)

    c = lambda delta, n0 : sigma * np.sqrt(2 * np.log(1/delta) / n0)
    f = lambda n0 : n0/N
    
    sigma=mu_null*(1-mu_null)

    gamma2 = (1-f(n-1))*np.power(sigma, 2) + f(n-1)*c(delta, n-1)
    gamma2_bar = (1-f(n)) * ((n+1)/n * np.power(sigma, 2) + \
                 (N-n-1)/n * c(delta, N-n-1))

    first_prob = np.exp((-n * np.power((eps * t * ( N-n ))/(n * (n-t)), 2)) /\
            (2 * gamma2 + (2/3) * (eps * t * (N-n))/(n * (N-t))))

    second_prob = np.exp(-n*np.power(eps, 2)/ (2 * (gamma2_bar + (2/3) * eps)))
   
    probs = np.ones(len(t))
    
    for i in np.arange(0, len(t)):
        if t[i] <= n:
            probs[i] = first_prob[i]
        else:
            probs[i] = second_prob

    return(probs)
    if len(t) > 1:
        probs = np.append(first_prob[0:(n-1)], second_prob[n:(len(t)-1)])
    else:
        if t <= n:
            return first_prob
        else:
            return second_prob

def hoeffding_wor_BM(x, lower_bd, upper_bd, N, n, alpha=0.05):
    t = np.arange(1, len(x) + 1)
    S_t = np.cumsum(x)

    epsilon_1 = (np.sqrt(np.log(4 / alpha) *
                         (1 - (n - 1) / N) / (2 * n)) *
                 np.abs(upper_bd - lower_bd))

    CI_bound_1 = (N - t) / (t * (N - n)) * n * epsilon_1
    lower_CI_1 = S_t / t - CI_bound_1
    upper_CI_1 = S_t / t + CI_bound_1

    epsilon_2 = (np.sqrt(np.log(4 / alpha) * (1 - n / N) *
                         (1 + 1 / n) / (2 * n)) *
                 np.abs(upper_bd - lower_bd))

    CI_bound_2 = epsilon_2

    lower_CI_2 = S_t / t - CI_bound_2
    upper_CI_2 = S_t / t + CI_bound_2

    lower_CI = np.append(lower_CI_1[0:n], lower_CI_2[n:len(x)])
    upper_CI = np.append(upper_CI_1[0:n], upper_CI_2[n:len(x)])

    return(lower_CI, upper_CI)


def hoeffding_wor_BM_stitch(x, lower_bd, upper_bd,
                         N, epoch_times, alpha=0.05,
                         running_intersect=False):

    assert(np.max(epoch_times) <= N)
    assert(np.min(epoch_times) >= 1)
    num_epochs = len(epoch_times)

    lower_CI_max = np.repeat(lower_bd, len(x))
    upper_CI_min = np.repeat(upper_bd, len(x))
    for epoch_time in epoch_times:
        lower_CI_i, upper_CI_i = hoeffding_wor_BM(x, lower_bd,
                                               upper_bd, N,
                                               n=epoch_time,
                                               alpha=alpha/num_epochs)
        lower_CI_max = np.maximum(lower_CI_max, lower_CI_i)
        upper_CI_min = np.minimum(upper_CI_min, upper_CI_i)

        lower_CI_max = np.maximum.accumulate(lower_CI_max)\
            if running_intersect else lower_CI_max
        upper_CI_min = np.minimum.accumulate(upper_CI_min)\
            if running_intersect else upper_CI_min

    return(lower_CI_max, upper_CI_min)

