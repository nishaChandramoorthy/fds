###############################################################################
#                                                                              #
#   sa2d_decomp_value.py copyright(c) Qiqi Wang 2015 (qiqi.wang@gmail.com)     #
#                                                                              #
################################################################################

import os
import sys
import time
import collections
import copy as copymodule
from subprocess import Popen, PIPE
from io import BytesIO

import numpy as np

def _is_like_sa_value(a):
    '''
    Check attributes of symbolic array value
    '''
    if hasattr(a, 'owner'):
        return a.owner is None or hasattr(a.owner, 'access_neighbor')
    else:
        return False

# ============================================================================ #
#                             symbolic array value                             #
# ============================================================================ #

class symbolic_array_value(object):
    def __init__(self, shape=(), owner=None, field=None):
        self.shape = np.empty(shape).shape
        self.owner = owner
        self.field = field

    def __repr__(self):
        if self.owner:
            return 'Dependent value of shape {0} generated by {1}'.format(
                    self.shape, self.owner)
        else:
            return 'Independent value of shape {0}'.format(self.shape)

    # --------------------------- properties ------------------------------ #

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    def __len__(self):
        return 1 if not self.shape else self.shape[0]


class builtin:
    ZERO = symbolic_array_value()
    RANDOM = []

def random_value(shape=()):
    new_random_value = symbolic_array_value(shape)
    builtin.RANDOM.append(new_random_value)
    return new_random_value


# ============================================================================ #
#                             computational graph                              #
# ============================================================================ #

def discover_values(output_values):
    discovered_values = []
    discovered_input_values = []
    def discover_values_from(v):
        if not hasattr(v, 'owner'):
            return
        if v.owner is None:
            if v not in discovered_input_values:
                discovered_input_values.append(v)
        elif v not in discovered_values:
            discovered_values.append(v)
            for v_inp in v.owner.inputs:
                discover_values_from(v_inp)
    for v in output_values:
        discover_values_from(v)
    return discovered_values, discovered_input_values

def sort_values(unsorted_values):
    sorted_values = []
    def is_computable(v):
        return (not _is_like_sa_value(v) or
                v in sorted_values or
                v.owner is None)
    while len(unsorted_values):
        removed_any = False
        for v in unsorted_values:
            if all([is_computable(v_inp) for v_inp in v.owner.inputs]):
                unsorted_values.remove(v)
                sorted_values.append(v)
                removed_any = True
        assert removed_any
    return sorted_values

class ComputationalGraph(object):
    '''
    Immutable compact stage
    '''
    def __init__(self, output_values):
        unsorted_values, self.input_values = discover_values(output_values)
        self.sorted_values = sort_values(unsorted_values)
        assert(all(v in self.sorted_values for v in output_values))
        assert unsorted_values == []
        self.output_values = copymodule.copy(output_values)

    def __call__(self, inputs):
        if hasattr(inputs, '__call__'):
            actual_inputs = [inputs(v) for v in self.input_values]
        elif hasattr(inputs, '__getitem__'):
            actual_inputs = [inputs[v] for v in self.input_values]
        # _act attributes are assigned to inputs
        for v, v_act in zip(self.input_values, actual_inputs):
            assert not hasattr(v, '_act')
            v._act = v_act
        # _act attributes are computed to each value
        def _act(v):
            if _is_like_sa_value(v):
                return v._act
            elif isinstance(v, np.ndarray):
                return v.reshape(v.shape + (1,))
            else:
                return v
        for v in self.sorted_values:
            assert not hasattr(v, '_act')
            inputs_act = [_act(v_inp) for v_inp in v.owner.inputs]
            v._act = v.owner.perform(inputs_act)
        # _act attributes are extracted from outputs then deleted from all
        actual_outputs = tuple(v._act for v in self.output_values)
        for v in self.input_values + self.sorted_values:
            del v._act
        return actual_outputs

################################################################################
################################################################################
################################################################################
