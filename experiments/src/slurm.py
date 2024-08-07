# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
# from logging import getLogger
import os
import sys
import socket
import signal


# logger = getLogger()


def sig_handler(signum, frame):
    print("Signal handler called with signal " + str(signum))
    prod_id = int(os.environ['SLURM_PROCID'])
    print("Host: %s - Global rank: %i" % (socket.gethostname(), prod_id))
    if prod_id == 0:
        print("Requeuing job " + os.environ['SLURM_JOB_ID'])
        os.system('scontrol requeue ' + os.environ['SLURM_JOB_ID'])
    else:
        print("Not the master process, no need to requeue.")
    sys.exit(-1)


def term_handler(signum, frame):
    print("Signal handler called with signal " + str(signum))
    print("Bypassing SIGTERM.")


def init_signal_handler():
    """
    Handle signals sent by SLURM for time limit / pre-emption.
    """
    signal.signal(signal.SIGUSR1, sig_handler)
    signal.signal(signal.SIGTERM, term_handler)
    print("Signal handler installed.")
