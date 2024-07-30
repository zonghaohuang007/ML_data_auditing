# from https://github.com/WannabeSmith/confseq_wor/tree/main/cswor

import matplotlib.pyplot as plt
import math
import numpy as np
import random
import pandas as pd
from scipy import optimize
from scipy.special import beta, betaln, comb, gammaln
from utils import *
from misc import *
import multiprocessing as mp
import time

from multiprocess import Pool
import multiprocess

def DMHG_martingale(x, DM_alpha, T_null):
    '''
    Dirichlet-multinomial-hypergeometric martingale
    with possibly more than 2 categories (colors)

    Parameters
    -----------
    x, matrix-like
        K by S matrix where K is the number of
        categories and S is the number of samples taken. Entries
        are the number of balls of color k in sample s,
        k in {1, ..., K}, s in {1, ... , S}. A list of
        values for a single observation is also acceptable.
    DM_alpha, array-like of positive reals
        alpha parameters for
        the Dirichlet-multinomial distribution. Must have same
        dimensions as x
    T_null, K-list of integers
        null vector for number of balls of colors 1 through K

    Returns
    -------
    martingale, array-like
        martingale as balls are sampled from the urn
    '''
    # put into numpy column vector format
    T_null = np.array(T_null)[:, None]

    # Make x a numpy array
    x = np.array(x)

    # if there's just a single observation, turn into column vector
    if x.ndim == 1:
        x = x[:, None]

    # Total number of balls in urn
    N = np.sum(T_null)

    # number of samples at each time
    n = x.sum(axis=0)
    # intrinsic time
    t = n.cumsum()

    # The cumulative sum process
    S_t = x.cumsum(axis=1)

    # Convert DM_alpha into column vector format.
    DM_alpha = np.array(DM_alpha)[:, None]

    N_t = N - t
    DM_alpha_t = DM_alpha + S_t

    log_prior_0 = dirichlet_multinomial_pmf(T_null,
                                            [N], DM_alpha,
                                            log_scale=True)
    log_posterior_0t = dirichlet_multinomial_pmf(T_null - S_t,
                                                 N_t, DM_alpha_t,
                                                 log_scale=True)
    log_M_t = log_prior_0 - log_posterior_0t
    log_M_t = np.float128(log_M_t) 
    martingale = np.exp(log_M_t)

    return martingale

def logical_cs(x, N):
    '''
    The 1-dimensional logical confidence sequence for sampling without
    replacement. This is essentially the CS that would be 
    known regardless of the underlying martingale being used.
    Specifically, if the cumulative sum at time t, S_t is equal to
    5 and N is 10, then the true mean cannot be any less than 0.5, assuming
    all observations are between 0 and 1.

    Parameters
    ----------
    x, array-like of reals between 0 and 1
        The observed bounded random variables.
    N, integer
        The size of the finite population
    '''
    t = np.arange(1, len(x) + 1)

    S_t = np.cumsum(x)

    l = S_t/N
    u = 1-(t-S_t)/N

    return l, u 

def ci_from_martingale_2D(mu_hat, N, mart_vec,
                          possible_m,
                          alpha=0.05,
                          search_step=1):
    '''
    Gets a single confidence interval from within a CS,
    given a martingale vector. This is mostly a helper-function
    for cs_from_martingale_2D.

    Parameters
    ----------
    mu_hat, real number
        Estimates of the mean at each time
    N, integer
        Total population size
    mart_vec, array-like of positive reals
        Martingale values for each candidate null
    possible_m, array-like of real numbers
        Candidate nulls (e.g. 0, 1, 2, 3, ...) for the number
        of green balls in an urn.
    alpha, real between 0 and 1
        Confidence level (e.g. 0.05)
    search_step, positive integer
        How much to step when searching the parameter space

    Returns
    -------
    lower, array-like of reals
        Lower confidence interval. A lower bound on the smallest
        value not rejected
    upper, array-like of reals
        Upper confidence interval. An upper bound on the largest
        value not rejected
    '''
    where_in_cs = np.where(mart_vec < 1/alpha)

    # If can't find anything in the CS, we'll need to
    # return something that is a superset of the CS
    if len(where_in_cs[0]) == 0:
        lower = np.floor(mu_hat)
        upper = np.ceil(mu_hat)
    else:
        '''
        If the user is trying to search a subset of [0, N], 
        they will need to be slightly conservative and report
        a superset of the confidence set at each time.
        '''
        if search_step != 1:
            # If the boundaries are not rejected, no point
            # in searching for the confidence bound
            if mart_vec[0] < 1/alpha:
                lower = 0
            else:
                lower = possible_m[where_in_cs[0][0]-1]
            if mart_vec[len(possible_m)-1] < 1/alpha:
                upper = 1
            else:
                upper = possible_m[where_in_cs[0][-1]+1]
        else:
            lower = possible_m[where_in_cs[0][0]]
            upper = possible_m[where_in_cs[0][-1]]
        
    return lower, upper


def cs_from_martingale_2D(x, N, mart_fn, n=None,
                          alpha=0.05,
                          search_step=1,
                          running_intersection=False):
    '''
    Confidence sequence from an array of data, `x` and a function
    which produces a martingale, `mart_fn`. 

    Parameters
    ----------
    x, array-like of real numbers
        The observed data points
    N, positive integer
        Population size
    mart_fn, ([Real], Real -> [Real])
        Martingale function which takes an array-like of observations
        and a candidate null, and produces an array-like of positive
        martingale values.
    n, array-like of positive integers
        The total number of samples at each time. If left as `None`,
        n is assumed to be all ones.
    alpha, real between 0 and 1
        Confidence level
    search_step, positive integer
        The search step to be used when testing the different possible
        candidate values of m. 
    running_intersection, boolean
        If True, the running intersection of the confidence sequence
        is returned.
    '''
    possible_m = np.arange(0, N+1+search_step, step=search_step)/N
    possible_m = possible_m[possible_m <= 1]
    mart_mtx = np.zeros((len(possible_m), len(x)))

    if n is None:
        n = np.ones(len(x))
    t = np.cumsum(n)
    mu_hat_t = np.cumsum(x) / t
    # TODO: This can be sped up by doing a binary search, effectively
    # converting from O(N) time at each step to O(log N) time. However,
    # this is already quite fast for real-world use, so we'll leave it
    # as-is until we have free time to speed up.
    for i in np.arange(0, len(possible_m)):
        m = possible_m[i]
        
        mart_mtx[i, :] = mart_fn(x, m)

    lower = np.repeat(0.0, len(x))
    upper = np.repeat(1.0, len(x))
    for j in np.arange(0, len(x)):
        lower[j], upper[j] =\
            ci_from_martingale_2D(mu_hat_t[j], N,
                                      mart_vec=mart_mtx[:, j], 
                                      possible_m=possible_m,
                                      alpha=alpha,
                                      search_step=search_step)
        
    lgcl_l, lgcl_u = logical_cs(x, N)
    lower = np.maximum(lower, lgcl_l)
    upper = np.minimum(upper, lgcl_u)

    lower = np.maximum.accumulate(lower) if running_intersection else lower
    upper = np.minimum.accumulate(upper) if running_intersection else upper

    return lower, upper


def BBHG_confseq(x, N, BB_alpha, BB_beta, n=None, alpha=0.05,
                 running_intersection=False, search_step=1,
                 times=None):
    '''
    Confidence sequence for the total number of ones in
    an urn with ones and zeros exclusively. Based on the
    beta-binomial-hypergeometric martingale

    Parameters
    ---------- 
    x, array-like
        array of observations with ones and zeros
    n, array-like
        number of samples at each time. If left `None`,
        then n is assumed to be a list of ones.
    N, integer
        total number of objects in the urn
    BB_alpha, positive real
        alpha parameter for beta-binomial
    BB_beta, positive real
        beta parameter for beta-binomial
    alpha, positive real
        error level
    running_intersection, boolean
        should the running intersection of the confidence 
        sequence be taken?
    search_step, integer
        The step to take when searching through all 
        possible values of N^+, the parameter of interest.
        A search_step of 1 will search all possible values, a 
        value of 2 will search every other value, and so on.
    times, array-like of integers,
        The times at which to compute the confidence sequence.
        Leaving this as None will simply compute the CS at all
        times. To compute the CS at every other time, for example,
        simply set times=np.arange(0, N, step=2).


    Returns
    -------
    CIs_lower, list of positive reals
        lower part of CIs
    CIs_upper, list of positive reals
        upper part of CIs
    '''
    if n is None:
        n = np.ones(len(x))
    # We need x and n to be numpy arrays
    x = np.array(x)
    n = np.array(n)
    
    if times is not None:
        x = np.add.reduceat(x, times)
        n = np.add.reduceat(n, times)

    # cumulative sum
    S_t = np.cumsum(x)
    # intrinsic time
    t = np.cumsum(n)

    # Get x into "overparameterized" form as we usually do with multinomials,
    # for example.
    DM_x = np.vstack((x, n - x))
    
    mart_fn = lambda x, m: DMHG_martingale(np.vstack((x, n-x)),
                                           [BB_alpha, BB_beta],
                                           [int(N*m), N-int(N*m)])
    
    l_01, u_01 =\
        cs_from_martingale_2D(x, N, mart_fn, n=n, alpha=alpha, 
                              search_step=search_step,
                              running_intersection=running_intersection)

    return N*l_01, N*u_01


def plot_power(martingale_dict, data, T_true,
               nsim=1000, alpha=0.05, title=''):
    '''
    Plot the power for a dictionary of martingales given some
    true total number of votes for each party

    Parameters  
    ----------
    martingale_dict, dict of {string : function}
        dictionary of various martingales where the key is a 
        string (for the name of the martingale), and the 
        values are univariate functions that take a univariate 
        numpy array (or simply a list) of size S and produce 
        an S-vector/list of martingale values. 
    data, array-like
        array of 1s and 0s indicating samples of balls of color 1 or 2
    T_true, 2-list of integers
        the true number of balls of each color.
    nsim, integer
        number of simulations to perform
    alpha, positive real
        confidence level
    title, string
        title for the produced plot
    '''
    N = np.sum(len(data))

    # Get things ready for the plot
    t = np.ones(N).cumsum()
    threshold = np.repeat(alpha, len(t))
    plt.figure(0)

    # For each martingale, do a power simulation
    for mart_name in martingale_dict:
        mart_closure = martingale_dict[mart_name]

        mart_values = np.zeros((nsim, N))
        for sim in range(nsim):
            np.random.shuffle(data)
            mart_values[sim, :] = mart_closure(data)

        cumul_max_mart = np.maximum.accumulate(mart_values, axis=1)
        cumul_rejections = cumul_max_mart > 1/alpha
        cumul_reject_prob = cumul_rejections.sum(axis=0)/nsim

        plt.plot(t, cumul_reject_prob, label=mart_name)

        plt.xlabel("Number of ballots sampled")
        plt.ylabel("Rejection probability")
        plt.legend(loc="best")

    plt.title(title)
    plt.plot(t, threshold, linestyle="--",
             label="$\\alpha = " + str(alpha) + "$",
             color="grey")

def stopping_times(martingale_dict, data,
                   nsim=100, alpha=0.05,
                   num_proc=1, multiple_of=1):
    '''
    Get stopping times and simulation times

    Parameters
    ----------
    martingale_dict, dict of {string : function}
        dictionary of various martingales where the key is a string (for the name of the martingale), and the values are univariate functions
        that take KxS matrices (or np.arrays, or just lists of lists)
        and produce an S-vector/list of martingale values. Here, K is
        the number of categories, and S is the number of samples
    data, array-like
        array of the ones and zeros indicating votes for candidates
        1 and 2, respectively
    nsim, integer
        number of simulations to perform
    alpha, positive real
        confidence level
    title, string
        title for the produced plot
    num_proc, integer
        Number of CPU processes to spawn. This should be less than or
        equal to the number of CPU cores on the user's machine.
    '''
    N = np.sum(len(data))

    # Get things ready for the plot
    t = np.ones(N).cumsum()

    mart_names = list(martingale_dict.keys())
    stopping_times_dict = {}
    simulation_times_dict = {}

    def get_stopping_times(i):
        np.random.seed()
        np.random.shuffle(data)
        
        result_array = np.array([])
        for mart_name in mart_names:
            mart_closure = martingale_dict[mart_name]
            mart_value = mart_closure(data)
            mart_value[-1] = math.inf
            
            stopping_time =\
                np.where(np.logical_and(mart_value > 1/alpha,
                                        t%multiple_of == 0))[0][0] + 1
            result_array = np.append(result_array, stopping_time)
        return result_array
            
    with Pool(processes=num_proc) as pool:
        results =\
            np.array(pool.map(get_stopping_times,
                                range(nsim)))
            
    for j in range(len(mart_names)):
        mart_name = mart_names[j]
        stopping_times_dict[mart_name] = results[:, j]
                
    return stopping_times_dict


def plot_stopping_time(martingale_dict, data,
                       nsim=1000, alpha=0.05, title='',
                       num_proc=1):
    '''
    Plot the power for a dictionary of martingales 
    given some true total number of balls of each color in an urn

    Parameters
    ----------
    martingale_dict, dict of {string : function}
        dictionary of various martingales where the key is a string (for the name of the martingale), and the values are univariate functions
        that take KxS matrices (or np.arrays, or just lists of lists)
        and produce an S-vector/list of martingale values. Here, K is
        the number of categories, and S is the number of samples
    data, array-like
        array of the ones and zeros indicating votes for candidates
        1 and 2, respectively
    nsim, integer
        number of simulations to perform
    alpha, positive real
        confidence level
    title, string
        title for the produced plot
    num_proc, integer
        Number of CPU processes to spawn. This should be less than or
        equal to the number of CPU cores on the user's machine.
    '''
    N = np.sum(len(data))

    # Get things ready for the plot
    t = np.ones(N).cumsum()

    # For each martingale, do a power simulation
    for mart_name in martingale_dict:
        mart_closure = martingale_dict[mart_name]

        stopping_times = np.repeat(N, nsim)
        def get_stopping_time(i):
            np.random.shuffle(data)
            mart_value = mart_closure(data)
            mart_value[-1] = math.inf
            return np.where(mart_value > 1/alpha)[0][0]
        #stopping_times = np.array(list(map(get_stopping_time, range(nsim))))
        with Pool(processes=num_proc) as pool:
            stopping_times = pool.map(get_stopping_time, range(nsim))

        sns.kdeplot(stopping_times, label=mart_name, bw='silverman')
        plt.xlabel("Stopping time")
        plt.legend(loc='best')


def plot_2party_election(x, n, N, BB_alpha, BB_beta, Tb_true, Tb_null,
                         alpha=0.05, running_intersection=False,times=None,search_step=1):
    
    if running_intersection is False:
        alpha_false = 0.4
        alpha_true = 0.2
    else:
        alpha_false = 0.2
        alpha_true = 0.5
    
    t = np.cumsum(n)
    threshold = np.repeat(alpha, len(t))
    # put x into overparameterized form
    DM_x = np.vstack((x, n - x))
    pval = pval_from_martingale(DMHG_martingale(x=DM_x,
                                                DM_alpha=[BB_alpha, BB_beta],
                                                T_null=[Tb_null, N - Tb_null]))

    # Create p-value plots
    fig1 = plt.figure(0)
    plt.plot(t, pval, label="BB-HG")
    plt.plot(t, threshold, linestyle="--",
             label="$\\alpha = " + str(alpha) + "$")
    plt.title("Anytime-Valid P-Value During Election Audit")
    plt.xlabel("Number of Ballots Counted")
    plt.ylabel("any-time $p$-value")
    plt.legend(loc="best")

    # MLEs for Tb
    T_hats = N * np.cumsum(x) / t
    if running_intersection is True:
        CIs_lower_true, CIs_upper_true = \
            BBHG_confseq(x, n, N, 
                         BB_alpha, BB_beta, 
                         alpha=alpha,
                         running_intersection=True,search_step = search_step,times=times)
    
    CIs_lower_false, CIs_upper_false = \
        BBHG_confseq(x, n, N, 
                     BB_alpha, BB_beta, 
                     alpha=alpha,
                     running_intersection=False,search_step = search_step, times=times)
    # Create confidence sequence plots
    fig2 = plt.figure(1)
    plt.plot(t, T_hats,label = 'BB-HG MLE')

    plt.title("Confidence Sequence with MLE for True Number of Ballots")
    plt.xlabel('Number of Ballots Counted')
    plt.ylabel('Estimate of Winning Candidate Votes')
    plt.plot(t, np.ones(len(t)) * Tb_true, '--', color = 'green', label = 'True number of votes: {}'.format(Tb_true))
    plt.fill_between(times, CIs_lower_false, CIs_upper_false, color = 'red',alpha=alpha_false,label = 'Full Confidence Sequence')
    if running_intersection is True:
        plt.fill_between(times, CIs_lower_true, CIs_upper_true, color = 'orange',alpha=alpha_true,label = 'Running Intersection')
    plt.legend()
    return fig1, fig2

class Confseq3D:
    def __init__(self, N, DM_alpha, alpha, fineness=1):
        assert len(DM_alpha) == 3
        # Total number of values in category 1 can be between
        # 0 and N.
        # and the last party's total is determined by
        # the first two.
        self.N = N
        self.N_cmp = math.ceil((N + 1) * fineness)
        self.DM_alpha = DM_alpha
        self.alpha = alpha

        self.df = pd.DataFrame(np.ones((self.N_cmp,) * (3 - 1)))
        self.df.columns = np.minimum(self.N,
                                     np.floor(self.df.columns /
                                              fineness))
        self.df.index = self.df.columns

        # Set confidence interval for margin to be the entire space
        self.margin_lower = -N
        self.margin_upper = N

    def get_confidence_set(self):
        return self.df

    def update_from_S_t(self, x):
        # Reset margins
        self.margin_lower = self.N
        self.margin_upper = -self.N
        for total_1 in self.df.index:
            for total_2 in self.df.columns:
                # A set of votes is only valid if its sum
                # is less than or equal to N
                if total_1 + total_2 <= self.N:
                    # T_K_0 is the implied total votes for party K
                    total_3 = self.N - total_1 - total_2
                    pval = pval_from_martingale(
                            DMHG_martingale(x,
                                            self.DM_alpha,
                                            (total_1, total_2, total_3)))
                    self.df[total_1][total_2] = pval > self.alpha

                    # If the p-value <= alpha, update confidence
                    # interval for the margin of victory between
                    # categories 1 and 2.
                    if pval > self.alpha:
                        self.margin_lower = min(self.margin_lower,
                                                total_1 - total_2)
                        self.margin_upper = max(self.margin_upper,
                                                total_1 - total_2)
                else:
                    self.df[total_1][total_2] = 0

        return self.df.copy()

    def get_margin_confint(self):
        return (self.margin_lower, self.margin_upper)
    
def plot_3party_election(votes,
                         DM_alpha,
                         alpha, first, second,
                         N, sample,
                         fineness=0.1,
                         is_joint=False):
    '''
    Plot Confidence Set of 3-Party Election at a specific point in time
    Parameters
    ----------
    votes: 3xN integer array of 1s and 0s,
        Choice of candidate for each voter, with a 1 representing which party received the vote
    DM_alpha: array of length 3,
        Dirichlet-Multinomial priors for Confidence set
    alpha, positive real
        Confidence level
    first, positive integer
        The true number of votes the winning party received
    second, positive integer
        The true number of votes the runner-up party received
    sample, positive integer
        The number of samples collected at the time of the confidence set plot
    fineness, positive real
        The percentage of points that are computed for the confidence sequence
    is_joint, boolean
        Whether we are graphing the 3 sample plots or not 
    '''
    votes_cumsum = votes.cumsum(axis = 1)
    intrinsic_time = votes_cumsum.sum(axis = 0)
    T_hats = np.transpose(N * np.divide(votes_cumsum,
                            intrinsic_time[None, :]))
    c3d = Confseq3D(N, DM_alpha, alpha, fineness = fineness)
    if is_joint is False:
        cs = c3d.update_from_S_t(votes_cumsum[:, sample - 1])
        fig = plt.figure(0)
        #plt.rc('font', size = 12)
        plt.imshow(cs, 
               interpolation = 'none', extent = [0, N, N, 0])
        plt.title(str(sample) + " samples")
        plt.ylabel('Total Number of Winning Party Votes')
        plt.xlabel('Total Number of Runner-Up Party Votes')
        plt.scatter(first, second, label = "(" + str(first) + ", " + str(second) + ")",
                s = 50, color = "green")

        return fig
    else:
        samples = [10,N//2,N]
        cs1 = c3d.update_from_S_t(votes_cumsum[:, samples[0]-1])
        cs2 = c3d.update_from_S_t(votes_cumsum[:, samples[1]-1])
        cs3 = c3d.update_from_S_t(votes_cumsum[:, samples[2]-1])
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (9.5, 3.5))
        #plt.rc('font', size = 12)
        ax1.imshow(cs1, 
           interpolation = 'none', extent = [0, N, N, 0])
        ax1.set_title(str(samples[0]) + " samples")
        ax2.imshow(cs2, 
           interpolation = 'none', extent = [0, N, N, 0])
        ax2.set_title(str(samples[1]) + " samples")
        ax2.axes.get_yaxis().set_visible(False)
        ax3.imshow(cs3, 
           interpolation = 'none', extent = [0, N, N, 0])
        ax3.set_title(str(samples[2]) + " samples")
        ax3.axes.get_yaxis().set_visible(False)
        ax1.set(ylabel = 'Total Number of Winning Party Votes')
        ax2.set(xlabel = 'Total Number of Runner-Up Party Votes')
        s = 50
        ax1.scatter(first, second, label = "(" + str(first) + ", " + str(second) + ")",
            s = s, color = "green")
        ax2.scatter(first, second, label = "(" + str(first) + ", " + str(second) + ")",
            s = s, color = "green")
        ax3.scatter(first, second, label = "(" + str(first) + ", " + str(second) + ")",
            s = s, color = "green")
        ax3.legend(loc = "best")
        plt.setp((ax1, ax2, ax3), xlim=(0, N+1), ylim=(0, N+1))
        return fig

# This probably doesn't work anymore :(
def animate_3party_election(x, N, DM_alpha, alpha=0.05,
                            party_names=["Party 1", "Party 2", "Party 3"],
                            running_intersection=False,
                            plot_title=""):
    K = 3
    assert np.shape(x)[0] == K

    num_samples = np.shape(x)[1]
    x_cumsum = x.cumsum(axis=1)

    # MLE at each time: N * x_cumsum / intrinsic_time
    intrinsic_time = x_cumsum.sum(axis=0)
    T_hats = np.transpose(N *
                          np.divide(x_cumsum,
                                    intrinsic_time[None, :]))
    # Create confidence set sequence plots
    cm = ConfseqMatrixSlow(N, K, DM_alpha, alpha)
    fig = plt.figure(1)
    plt.title(plot_title)
    camera = Camera(fig)

    i = 0
    plt.axhline(y=T_hats[i, 0], color='red', label="MLE")
    # If x has no missing votes
    # (i.e. we've seen the whole election)
    final_count = x_cumsum[:, num_samples - 1]
    if final_count.sum() == N:
        # Add a dot to indicate the true
        # totals for each party
        plt.scatter(final_count[1],
                    final_count[0],
                    color='green',
                    label="True vote total")

    while i < num_samples:
        confidence_set = cm.update_from_S_t(x_cumsum[:, i])

        # same as above: if this is an entire election,
        # then plot the true total. Here we must omit
        # the label to prevent matplotlib from plotting
        # multiple labels for the true total.
        if final_count.sum() == N:
            # Add a dot to indicate the true
            # totals for each party
            plt.scatter(final_count[1],
                        final_count[0],
                        color='green')

        plt.imshow(confidence_set, origin='lower')
        plt.ylabel(party_names[0] + " votes")
        plt.xlabel(party_names[1] + " votes")
        # plot a dot for the MLE of
        # the first and second categories
        plt.scatter(T_hats[i, 1], T_hats[i, 0], color='red')
        plt.legend(loc="best")
        # Take a snapshot of this plot
        camera.snap()
        i += 1

    animation = camera.animate()
    return(animation)

def hoeffding_lambda_from_t(t, alpha, lower_bd, upper_bd):
    '''
    Get optimal tuning parameter, lambda from a time t.
    This is for use in conjunction with exponential
    supermartingale-based confidence sequences for bounded random variables
    
    Parameters
    ----------
    t, array-like
        array of times for which the confidence sequence should be optimized.
    alpha, real
        desired controlled error rate between 0 and 1
    lower_bd, real
        known lower bound for the random variables of interest
    upper_bd, real
        known upper bound for the random variables of interest

    Returns
    -------
    lambda_opt, list of real numbers
        optimal lambdas for supermartingale-based confidence 
        sequence for bounded random variables.
    '''

    t = np.array(t)
    lambda_list = np.sqrt(8 * np.log(2/alpha)/
                          (np.array(t)*np.power(upper_bd - lower_bd, 2)))
    return(lambda_list)

def hoeffding_wor_lc(x, lower_bd, upper_bd,
                     t_opt, N,
                     alpha=0.05):
    '''
    Line-crossing confidence sequence for bounded
    random variables when sampling without replacement,
    based on an exponential supermartingale. This is mainly
    a helper function for hoeffding_wor, which is the
    function which should be used in practice

    Parameters
    ----------
    x, array-like
        array of observations 
    lower_bd, real
        lower bound on the random variable
    upper_bd, real
        upper bound on the random variable
    t_opt, integer
        time for which the confidence sequence should be optimized.
    N, integer
        population size
    alpha, real between 0 and 1
        confidence level
    running_intersection, boolean
        should the running intersection of
        the confidence sequence be taken?

    Returns
    -------
    CIs_lower, list of positive reals
        upper part of CIs
    CIs_upper, list of positive reals
        upper part of CIs
    '''
    x = np.array(x)
    t = np.arange(1, len(x) + 1)
    lambda_par = hoeffding_lambda_from_t(np.array(t_opt), alpha,
                                    lower_bd = lower_bd,
                                    upper_bd = upper_bd)

    S_t = np.cumsum(x)
    Zstar = np.append(0, S_t/(N-t))[0:(len(S_t))]
    Wstar = np.append(0, t/(N-t))[0:(len(S_t))]

    psi = np.power(lambda_par, 2)*np.power(upper_bd-lower_bd, 2)/8
    
    mu_hat = (np.cumsum(Zstar) + S_t) / (t + np.cumsum(Wstar))
    mu_hat_classic = S_t/t
    margin = (t*psi + np.log(2/alpha)) / (lambda_par * (t + np.cumsum(Wstar)))
    lower_CI = mu_hat - margin
    upper_CI = mu_hat + margin

    return(lower_CI, upper_CI)

def hoeffding_wor(x, lower_bd, upper_bd,
                  t_opts, N, alpha=0.05,
                  running_intersection=False):
    '''
    Confidence sequence for bounded random variables when sampling
    without replacement, based on a Hoeffding exponential
    supermartingale and using the stitching technique

    Parameters
    ----------
    x, array-like, array of observations indicating votes
        for party 1. A single value is also acceptable
    lower_bd, real
        lower bound on the random variable
    upper_bd, real
        upper bound on the random variable
    t_opts, array-like
        array of times for which the confidence
        sequence should be optimized. Can also supply
        a single number which will be
        converted to a numpy array of length 1.
    N, integer
        Population size
    alpha, positive real
        confidence level
    running_intersection, boolean
        should the running intersection
        of the confidence sequence be taken?

    Returns
    -------
    CIs_lower, list of positive reals
        upper part of CIs
    CIs_upper, list of positive reals
        upper part of CIs
    '''
    # If just a number is presented, change it to a numpy array
    if not isinstance(t_opts, np.ndarray):
        t_opts = np.array([t_opts])
    num_t_opt = len(t_opts)

    lower_CI_max = np.repeat(lower_bd, len(x))
    upper_CI_min = np.repeat(upper_bd, len(x))
    for t_opt in t_opts:
        lower_CI_i, upper_CI_i = hoeffding_wor_lc(x, lower_bd,
                                                  upper_bd, N=N,
                                                  t_opt=t_opt,
                                                  alpha=alpha/num_t_opt)
        lower_CI_max = np.maximum(lower_CI_max, lower_CI_i)
        upper_CI_min = np.minimum(upper_CI_min, upper_CI_i)

    if running_intersection:
        lower_CI_max = np.maximum.accumulate(lower_CI_max)
        upper_CI_min = np.minimum.accumulate(upper_CI_min)

    return(lower_CI_max, upper_CI_min)

hoeffding_wor_lc_stitch = hoeffding_wor

# With-replacement Hoeffding. For simulations and comparison.
def hoeffding_wr_lc(x, lower_bd, upper_bd,
                    t_opt, N, alpha=0.05,
                    running_intersect=False):
    x = np.array(x)
    t = np.arange(1, len(x) + 1)
    lambda_par = hoeffding_lambda_from_t(np.array(t_opt), alpha,
                                    lower_bd = lower_bd,
                                    upper_bd = upper_bd)

    S_t = np.cumsum(x)

    psi = np.power(lambda_par, 2)*np.power(upper_bd-lower_bd, 2)/8
    
    mu_hat = S_t / t
    margin = (t*psi + np.log(2/alpha)) / (lambda_par * t)
    lower_CI = mu_hat - margin
    upper_CI = mu_hat + margin

    return(lower_CI, upper_CI)

# With-replacement Hoeffding. For simulations and comparison.
def hoeffding_wr_lc_stitch(x, lower_bd, upper_bd,
                          t_opts, N, alpha=0.05,
                          running_intersect=False):
    num_t_opt = len(t_opts)

    lower_CI_max = np.repeat(lower_bd, len(x))
    upper_CI_min = np.repeat(upper_bd, len(x))
    for t_opt in t_opts:
        lower_CI_i, upper_CI_i = hoeffding_wr_lc(x, lower_bd,
                                                  upper_bd, N=N,
                                                  t_opt=t_opt,
                                                  alpha=alpha/num_t_opt)
        lower_CI_max = np.maximum(lower_CI_max, lower_CI_i)
        upper_CI_min = np.minimum(upper_CI_min, upper_CI_i)

        lower_CI_max = np.maximum.accumulate(lower_CI_max)\
            if running_intersect else lower_CI_max
        upper_CI_min = np.minimum.accumulate(upper_CI_min)\
            if running_intersect else upper_CI_min

    return(lower_CI_max, upper_CI_min)

def emp_bern_lambda_from_V_t(V_t, alpha, lower_bd, upper_bd):
    '''
    Get optimal tuning parameter, lambda from a time t. This is for use
    in conjunction with the empirical Bernstein exponential 
    supermartingale-based confidence sequences for bounded random 
    variables
    
    Parameters
    ----------
    V_t, array-like
        array of variance times for which the confidence
        sequence should be optimized.
    alpha, real
        confidence level
    lower_bd, real
        Lower bound on the random variables
    upper_bd, real
        Upper bound on the random variables

    Returns
    -------
    lambda_opt, list of real numbers
        optimal lambdas for supermartingale-based confidence sequence
        for bounded random variables.
    '''

    V_t = np.array(V_t)
    c = upper_bd-lower_bd

    def fn(lambda_par):
        val = lambda_par*V_t*c/4 +\
              V_t*(1-c*lambda_par)*np.log(1-c*lambda_par)/4-\
              (1-c*lambda_par)*np.log(2/alpha)
        val = V_t/4 if lambda_par == 1/c else val
        return val
       
    lambda_par = optimize.ridder(fn, 0, 1/c)
        
    return(lambda_par)

def emp_bern_wor_lc(x, lower_bd, upper_bd,
                     V_t_opt, N,
                     alpha=0.05,
                     two_sided=True,
                     upperCS=True):
    '''
    Line-crossing confidence sequence for bounded
    random variables when sampling without replacement,
    based on an exponential supermartingale

    Parameters
    ----------
    x, array-like
        array of observations
    lower_bd, real
        lower bound on the random variable
    upper_bd, real
        upper bound on the random variable
    V_t_opt, integer
        time for which the confidence sequence should be optimized.
    N, integer
        population size
    alpha, positive real
        confidence level
    running_intersection, boolean
        should the running intersection of the
        confidence sequence be taken?
    two_sided, boolean
        Should a two-sided CS be returned
    upperCS, boolean
        should the upper CS be returned

    Returns
    -------
    CIs_lower, list of positive reals
        lower part of CIs
    CIs_upper, list of positive reals
        upper part of CIs
    '''
    x = np.array(x)
    t = np.arange(1, len(x) + 1)
    lambda_par = emp_bern_lambda_from_V_t(np.array(V_t_opt), alpha,
                                          lower_bd = lower_bd,
                                          upper_bd = upper_bd)
    
    S_t = np.cumsum(x)
    Zstar = np.append([0], S_t)[0:len(S_t)] / (N-(t-1))
    Wstar = np.append([0], t)[0:len(S_t)] / (N-(t-1))
    
    c = upper_bd-lower_bd
    psi = (-np.log(1-c*lambda_par) - c*lambda_par) / 4
    
    mu_hat = (np.cumsum(Zstar) + S_t) / (t + np.cumsum(Wstar))
    mu_hat_tminus1 = np.append([0], mu_hat[0:(len(t)-1)])
    
    V_t = 4*np.cumsum(np.power(x - mu_hat_tminus1, 2)) \
            / np.power(c,2)
    mu_hat_classic = S_t/t

    deterministic_lower_CI = ((N-t)*lower_bd + S_t)/N
    deterministic_upper_CI = ((N-t)*upper_bd + S_t)/N
    if two_sided:
        margin = (V_t*psi + np.log(2/alpha)) /\
                 (lambda_par * (t + np.cumsum(Wstar)))
        lower_CI = mu_hat - margin
        upper_CI = mu_hat + margin
    else:
        margin = (V_t*psi + np.log(1/alpha)) /\
                 (lambda_par * (t + np.cumsum(Wstar)))
        lower_CI = deterministic_lower_CI if upperCS else mu_hat - margin
        upper_CI = mu_hat + margin if upperCS else deterministic_upper_CI
    
    # Since sampling without replacement, we have both a probabilistic
    # and a deterministic upper/lower bound 
    lower_CI = np.maximum(lower_CI, deterministic_lower_CI)
    upper_CI = np.minimum(upper_CI, deterministic_upper_CI)
    
    return(lower_CI, upper_CI)
    
def emp_bern_wor(x, lower_bd, upper_bd,
                 V_t_opts, N, alpha=0.05,
                 running_intersection=False,
                 two_sided=True,
                 upperCS=True):
    '''
    Confidence sequence for bounded random variables when sampling
    without replacement, based on an exponential supermartingale and
    using the stitching technique

    Parameters
    ----------
    x, array-like
        array of observations
    lower_bd, real
        lower bound on the random variable
    upper_bd, real
        upper bound on the random variable
    V_t_opts, array-like
        array of variance processes for which the confidence
        sequence should be optimized.
    N, integer
        number of total votes in election
    alpha, positive real
        confidence level
    running_intersection, boolean
        should the running intersection of the confidence
        sequence be taken?
    two_sided, boolean
        Should a two-sided CS be returned
    upperCS, boolean
        should the upper CS be returned

    Returns
    ------- 
    CIs_lower, list of positive reals
        lower part of CIs
    CIs_upper, list of positive reals
        upper part of CIs
    '''
    num_V_t_opt = len(V_t_opts)

    lower_CI_max = np.repeat(lower_bd, len(x))
    upper_CI_min = np.repeat(upper_bd, len(x))
    for V_t_opt in V_t_opts:
        lower_CI_i, upper_CI_i = emp_bern_wor_lc(x, lower_bd,
                                                 upper_bd, N=N,
                                                 V_t_opt=V_t_opt,
                                                 alpha=alpha/num_V_t_opt,
                                                 two_sided=two_sided,
                                                 upperCS=upperCS)
        lower_CI_max = np.maximum(lower_CI_max, lower_CI_i)
        upper_CI_min = np.minimum(upper_CI_min, upper_CI_i)

    if running_intersection:
        lower_CI_max = np.maximum.accumulate(lower_CI_max)
        upper_CI_min = np.minimum.accumulate(upper_CI_min)

    return(lower_CI_max, upper_CI_min)

emp_bern_wor_lc_stitch = emp_bern_wor

def predmix_empbern_wor(x, N, lambdas=None, 
                        alpha = 0.05, 
                        lower_bd=0, upper_bd=1,
                        running_intersection=False):
    '''
    Predictable mixture empirical Bernstein confidence sequence

    Parameters
    ----------
    x, array-like of reals
        Observations of numbers between 0 and 1
    N, integer
        Population size
    lambdas, array-like of reals
        lambda values for online mixture
    alpha, positive real
        Confidence level in (0, 1)
    lower_bd, real
        A-priori known lower bound for the observations
    upper_bd, real
        A-priori known upper bound for the observations
    running_intersection, boolean
        Should the running intersection be taken?

    Returns
    -------
    l, array-like of reals
        Lower confidence sequence for the mean
    u, array-like of reals
        Upper confidence sequence for the mean
    '''
    c = upper_bd - lower_bd

    x = np.array(x)
    t = np.arange(1, len(x)+1)
    S_t = np.cumsum(x)
    S_tminus1 = np.append(0, S_t[0:(len(S_t)-1)])
    mu_hat_tminus1 = (1/2 + S_tminus1)/t

    Zstar = S_tminus1 / (N-t+1)
    Wstar = (t-1)/(N-t+1)
    
    V_t = np.cumsum(1/4 + np.power(x - mu_hat_tminus1, 2))*\
          np.power(c/2, -2)
    V_tminus1 = np.append(4*np.power(c/2, -2)/4, V_t[0:(len(x)-1)])
    
    # If the user doesn't supply a sequence of lambdas,
    # use a sensible default.
    if lambdas is None:
        lambdas = np.sqrt(8*np.log(2/alpha)/\
                  (V_tminus1*np.log(1+t)*np.power(c,2)))
        lambdas[np.logical_or(np.isnan(lambdas), lambdas == math.inf)] = 0
        lambdas = np.minimum(1/(2*c), lambdas)
    
    weighted_mu_hat_t = np.cumsum(lambdas*(x + Zstar)) /\
                        np.cumsum(lambdas*(1+Wstar))

    psi = (-np.log(1-c*lambdas) - c*lambdas)/4

    margin = (np.cumsum(np.power(c/2, -2) *
             np.power(x - mu_hat_tminus1, 2) * psi) +
             np.log(2/alpha)) / np.cumsum(lambdas*(1+Wstar))

    l, u = weighted_mu_hat_t - margin, weighted_mu_hat_t + margin
    l = np.maximum(l, lower_bd)
    u = np.minimum(u, upper_bd)

    if running_intersection:
        l = np.maximum.accumulate(l)
        u = np.minimum.accumulate(u)

    return l, u

def predmix_hoeffding_wor(x, N, lambdas=None, 
                          alpha=0.05, 
                          lower_bd=0, upper_bd=1,
                          running_intersection=False):
    '''
    Predictable mixture Hoeffding confidence sequence

    Parameters
    ----------
    x, array-like of reals
        Observations of numbers between 0 and 1
    N, integer
        Population size
    lambdas, array-like of reals
        lambda values for online mixture
    alpha, positive real
        Confidence level in (0, 1)
    lower_bd, real
        A-priori known lower bound for the observations
    upper_bd, real
        A-priori known upper bound for the observations
    running_intersection, boolean
        Should the running intersection be taken?

    Returns
    -------
    l, array-like of reals
        Lower confidence sequence for the mean
    u, array-like of reals
        Upper confidence sequence for the mean
    '''
    x = np.array(x)
    t = np.arange(1, len(x)+1)
    S_t = np.cumsum(x)
    S_tminus1 = np.append(0, S_t[0:(len(x)-1)])

    Zstar = S_tminus1 / (N-t+1)
    Wstar = (t-1)/(N-t+1)
    # remove later
    Wstar = np.append(0, t/(N-t))[0:(len(S_t))]

    if lambdas is None:
        lambdas = np.sqrt(8*np.log(2/alpha)/
                          (t*np.log(1+t)*np.power(upper_bd-lower_bd, 2)))
        lambdas[np.logical_or(np.isnan(lambdas), lambdas == math.inf)] = 0
        lambdas = np.minimum(1/np.sqrt(np.power(upper_bd-lower_bd, 2)), lambdas)
    
    psi = np.cumsum(np.power(lambdas, 2))*np.power(upper_bd-lower_bd, 2)/8

    weighted_mu_hat_t = np.cumsum(lambdas*(x + Zstar)) /\
                        np.cumsum(lambdas*(1+Wstar))

    margin = (psi + np.log(2/alpha)) /\
              np.cumsum(lambdas*(1+Wstar))
    lower_CI = weighted_mu_hat_t - margin
    upper_CI = weighted_mu_hat_t + margin

    lower_CI = np.maximum(lower_CI, lower_bd)
    upper_CI = np.minimum(upper_CI, upper_bd)

    if running_intersection:
        lower_CI = np.maximum.accumulate(lower_CI)
        upper_CI = np.minimum.accumulate(upper_CI)

    return lower_CI, upper_CI 

def hoeffding_wor_ci_seq(x, N, times=None, 
                         lower_bd=0, upper_bd=1,
                         alpha=0.05):
    '''
    Sequence of fixed-time Hoeffding confidence intervals

    Parameters
    ----------
    x, array-like of reals
        Observations of numbers between 0 and 1
    N, integer
        Population size
    times, array-like
        Times at which to return the CI. If left as None,
        this function will just assume times 1
        through len(x)
    alpha, positive real
        Confidence level in (0, 1)
    lower_bd, real
        A-priori known lower bound for the observations
    upper_bd, real
        A-priori known upper bound for the observations

    Returns
    -------
    l, array-like of reals
        Lower confidence intervals for the mean
    u, array-like of reals
        Upper confidence intervals for the mean
    '''
    if times is None:
        times = np.arange(1, len(x) + 1)
    
    x = np.array(x)
    t = np.arange(1, len(x)+1)
    S_t = np.cumsum(x)
    S_tminus1 = np.append(0, S_t[0:(len(x)-1)])

    Zstar = S_tminus1 / (N-t+1)
    Wstar = (t-1)/(N-t+1)
    
    margin = np.sqrt(1/2*\
                     np.power(upper_bd-lower_bd,2)*\
                     np.log(2/alpha))/\
             (np.sqrt(t) + np.cumsum(Wstar)/np.sqrt(t))

    mu_hat_t = (S_t + np.cumsum(Zstar)) / (t + np.cumsum(Wstar))
    lower_CI = mu_hat_t - margin
    upper_CI = mu_hat_t + margin

    l_logical, u_logical = logical_cs(x, N)
    lower_CI = np.maximum(lower_CI, l_logical)
    upper_CI = np.minimum(upper_CI, u_logical)

    return lower_CI[times-1], upper_CI[times-1]

# TODO: incorporate lower/upper bounds into function.
# For now, take the random variable, subtract off \ell, 
# divide by u-\ell, and the rv is now in [0, 1].
def lambda_adaptive(x, N, truncation=0.5, alpha=0.05, fixed_n=None):
    '''
    Get adaptive lambda sequence.

    Parameters
    ----------
    x, array-like of reals
        Observations of numbers between 0 and 1
    N, integer
        Population size
    truncation, real
        Trunction, less than 1
    alpha, positive real
        Confidence level in (0, 1)
    fixed_n, integer
        Fixed sample size to optimize the bound for.

    Returns
    -------
    lambdas, array-like of positive reals
        lambda values for the predictably-mixed
        empirical Bernstein CS/CI
    '''
    assert(truncation < 1)
    x = np.array(x)
    t = np.arange(1, len(x)+1)
    S_t = np.cumsum(x)
    mu_hat_t = S_t/t

    S_tminus1 = np.append(0, S_t[0:(len(S_t)-1)])
    V_t = (1/4 + np.cumsum(np.power(x - mu_hat_t, 2))) / t
    V_tminus1 = np.append(0, V_t[0:(len(x)-1)])

    if fixed_n is None:
        lambdas = np.sqrt(2*np.log(1/alpha)/(t*np.log(1+t)*V_tminus1))
    else:
        lambdas = np.sqrt(2*np.log(1/alpha)/(fixed_n*V_tminus1))
    lambdas[np.isnan(lambdas)] = 0

    lambdas = np.minimum(lambdas, truncation)

    return lambdas

def empbern_wor_ci(x, N, lower_bd=0, upper_bd=1, alpha=0.05):
    '''
    Fixed-time empirical Bernstein confidence interval

    Parameters
    ----------
    x, array-like of reals
        Observations of numbers between 0 and 1
    N, integer
        Population size
    lower_bd, real
        A-priori known lower bound for the observations
    upper_bd, real
        A-priori known upper bound for the observations
    alpha, positive real
        Confidence level in (0, 1)

    Returns
    -------
    l, array-like of reals
        Lower confidence intervals for the mean
    u, array-like of reals
        Upper confidence intervals for the mean
    '''
    np.random.shuffle(x)
    t = np.arange(1, len(x)+1)
    n = len(x)
    c = upper_bd - lower_bd
    mu_tilde_t = np.cumsum(x) / t

    lambdas = lambda_adaptive(x, N, alpha=alpha/2, fixed_n=n)
    
    l, u = predmix_empbern_wor(x, N, lambdas=lambdas,
                               alpha=alpha, lower_bd=lower_bd,
                               upper_bd=upper_bd,
                               running_intersection=False)
    return l[-1], u[-1]

def get_ci_seq(x, ci_fn, times, parallel=False):
    '''
    Get a sequence of confidence intervals (usually for
    plotting purposes) given data, a function for constructing
    confidence intervals, and times to compute them.

    Parameters
    ----------
    x, array-like of reals
        Observations between 0 and 1
    ci_fn, ([Real] -> (Real, Real))
        Function which takes an array-like of observations
        and produces a confidence interval (as a tuple)
    times, array-like of integers
        Times for which the CI should be computed
    parallel, boolean
        Should the computation be parallelized?
    
    Returns
    -------
    l, array-like of reals
        Sequence of lower confidence intervals
    u, array-like of reals
        Sequence of upper confidence intervals
    '''
    x = np.array(x)

    l = np.repeat(0.0, len(times))
    u = np.repeat(1.0, len(times))

    if parallel:
        n_cores = multiprocess.cpu_count()
        print('Using ' + str(n_cores) + ' cores')
        with Pool(n_cores) as p:
            result =\
                np.array(
                    p.map(
                        lambda time: ci_fn(x[0:time]), times))
        l, u = result[:, 0], result[:, 1]
    else:
        for i in np.arange(0, len(times)):
            time = times[i]
            x_t = x[0:time]
            l[i], u[i] = ci_fn(x_t)

    return l, u


def empbern_wor_ci_seq(x, N, times, lower_bd=0,
                       upper_bd=1, alpha=0.05,
                       parallel=False):
    '''
    Sequence of empirical Bernstein confidence intervals

    Parameters
    ----------
    x, array-like of reals
        Observations between 0 and 1
    N, integer
        Population size
    times, array-like of integers
        Times for which the CI should be computed
    lower_bd, real
        Lower bound on the data
    upper_bd, real
        Upper bound on the data
    alpha, real in (0, 1)
        Confidence level
    parallel, boolean
        Should the computation be parallelized?
    
    Returns
    -------
    l, array-like of reals
        Sequence of lower confidence intervals
    u, array-like of reals
        Sequence of upper confidence intervals
    '''
    cs_fn = lambda x: empbern_wor_ci(x, N,
                                     lower_bd,
                                     upper_bd,
                                     alpha)
    return get_ci_seq(x, cs_fn, times, parallel=parallel)

