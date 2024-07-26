from cswor import BBHG_confseq

import math
import random


def tight_chernoff_bound(tau, N):
    return (math.exp(tau*2/N-1) / ((tau*2/N)**(tau*2/N)))**(N/2)


def find_tau(p, N):
    tau_a = N // 2
    tau_b = N

    while tau_b - tau_a > 1:
        if tight_chernoff_bound((tau_a+tau_b)//2, N) > p:
            tau_a = (tau_a+tau_b)//2
        elif tight_chernoff_bound((tau_a+tau_b)//2, N) < p:
            tau_b = (tau_a+tau_b)//2
        else:
            tau_b = (tau_a+tau_b)//2
            break
    assert tight_chernoff_bound(tau_b, N) <= p
    return tau_b


def data_use_detection(target_model, score_function, published_data, unpublished_data, args):

    assert len(published_data) == len(unpublished_data)
    data_pairs = [[published_data[i], unpublished_data[i]] for i in range(len(published_data))]

    # random shuffle the order of pairs
    random.shuffle(data_pairs)

    sequences = []
    detected = False
    alpha1 = args.p / 2
    alpha2 = args.p / 2
    tau =  find_tau(alpha2, len(data_pairs))
    for j in range(len(data_pairs)):

        score_published = score_function(target_model, data_pairs[j][0])
        score_unpublished = score_function(target_model, data_pairs[j][1])
        
        if score_published > score_unpublished:
            success = 1
        elif score_published == score_unpublished:  # if equal, toss a coin
            if random.sample([True, False], k=1)[0]:
                success = 1
            else:
                success = 0
        else:
            success = 0
        sequences.append(success)
        lower_bounds, _ = BBHG_confseq(sequences, len(data_pairs), BB_alpha=1, BB_beta=1, alpha=alpha1)  # apply the PPRM method
        assert len(lower_bounds) == len(sequences)
        if lower_bounds[-1] >= tau:  # stop if the lower_bound is larger than the preselected threshold
            detected = True
    
        if detected:
            break
    
    return detected