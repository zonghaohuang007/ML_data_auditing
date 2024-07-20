"""Base victim class."""

import torch
from torch import nn
from torch import optim

from .models import get_model
from .training import get_optimizers, run_step, run_self_step
from .optimization_strategy import training_strategy
from ..utils import average_dicts
from ..consts import BENCHMARK, SHARING_STRATEGY
torch.backends.cudnn.benchmark = BENCHMARK
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


class DefenseModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.num_classes = in_features
        self.layer = torch.nn.Sequential(
                torch.nn.Linear(in_features * 2, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1),
                )

    def forward(self, inputs, labels):
        return torch.nn.Sigmoid()(self.layer(torch.cat((inputs, labels), 1)))
    

class _VictimBase:
    """Implement model-specific code and behavior.

    Expose:
    Attributes:
     - model
     - optimizer
     - scheduler
     - criterion

     Methods:
     - initialize
     - train
     - retrain
     - validate
     - iterate

     - compute
     - gradient
     - eval

     Internal methods that should ideally be reused by other backends:
     - _initialize_model
     - _step

    """

    def __init__(self, args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize empty victim."""
        self.args, self.setup = args, setup
        if self.args.ensemble < len(self.args.net):
            raise ValueError(f'More models requested than ensemble size.'
                             f'Increase ensemble size or reduce models.')
        self.initialize()

    def gradient(self, images, labels):
        """Compute the gradient of criterion(model) w.r.t to given data."""
        raise NotImplementedError()
        return grad, grad_norm

    def compute(self, function):
        """Compute function on all models.

        Function has arguments: model, criterion
        """
        raise NotImplementedError()

    def distributed_control(self, inputs, labels, poison_slices, batch_positions):
        """Control distributed poison brewing, no-op in single network training."""
        randgen = None
        return inputs, labels, poison_slices, batch_positions, randgen

    def sync_gradients(self, input):
        """Sync gradients of given variable. No-op for single network training."""
        return input

    def reset_learning_rate(self):
        """Reset scheduler object to initial state."""
        raise NotImplementedError()


    """ Methods to initialize a model."""

    def initialize(self, seed=None):
        raise NotImplementedError()

    """ METHODS FOR (CLEAN) TRAINING AND TESTING OF BREWED POISONS"""

    def train(self, kettle, max_epoch=None):
        """Clean (pre)-training of the chosen model, no poisoning involved."""
        print('Starting clean training ...')
        return self._iterate(kettle, poison_delta=None, max_epoch=max_epoch)

    def retrain(self, kettle, poison_delta):
        """Check poison on the initialization it was brewed on."""
        self.initialize(seed=self.model_init_seed)
        print('Model re-initialized to initial seed.')
        return self._self_iterate(kettle, poison_delta=poison_delta)

    def validate(self, kettle, poison_delta):
        if poison_delta == 'advreg':
            print('adversarial regularization...')
            kettle.inference_model = DefenseModel(len(kettle.trainset.classes)).cuda()
            kettle.adv_solver = optim.Adam(kettle.inference_model.parameters(), lr=0.001)
        """Check poison on a new initialization(s)."""
        run_stats = list()
        for runs in range(self.args.vruns):
            self.initialize()
            print('Model reinitialized to random seed.')
            run_stats.append(self._iterate(kettle, poison_delta=poison_delta))

        return average_dicts(run_stats)

    def selftrain(self, kettle):
        """Self-training"""
        self.initialize()
        print('Model reinitialized to random seed.')

        return self._self_iterate(kettle)

    def eval(self, dropout=True):
        """Switch everything into evaluation mode."""
        raise NotImplementedError()

    def _iterate(self, kettle, poison_delta):
        """Validate a given poison by training the model and checking target accuracy."""
        raise NotImplementedError()

    def _self_iterate(self, kettle, utkettle):
        """Validate a given poison by training the model and checking target accuracy."""
        raise NotImplementedError()

    def _adversarial_step(self, kettle, poison_delta, step, poison_targets, true_classes):
        """Step through a model epoch to in turn minimize target loss."""
        raise NotImplementedError()

    def _initialize_model(self, model_name):

        model = get_model(model_name, self.args.dataset, pretrained=self.args.pretrained)
        # Define training routine
        defs = training_strategy(model_name, self.args)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer, scheduler = get_optimizers(model, self.args, defs)

        return model, defs, criterion, optimizer, scheduler


    def _step(self, kettle, poison_delta, loss_fn, epoch, stats, model, defs, criterion, optimizer, scheduler):
        """Single epoch. Can't say I'm a fan of this interface, but ..."""
        run_step(kettle, poison_delta, loss_fn, epoch, stats, model, defs, criterion, optimizer, scheduler)


    def _self_step(self, kettle, loss_fn, epoch, stats, model, defs, criterion, optimizer, scheduler):
        run_self_step(kettle, loss_fn, epoch, stats, model, defs, criterion, optimizer, scheduler)
