# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# List is needed for mypy, but not used in the code, only in special comments
from typing import Optional, List

import mxnet as mx

from sockeye.config import Config
from sockeye.layers import LayerNormalization
from . import constants as C


class RNNConfig(Config):
    """
    RNN configuration.

    :param cell_type: RNN cell type.
    :param num_hidden: Number of RNN hidden units.
    :param num_layers: Number of RNN layers.
    :param dropout: Dropout probability on RNN outputs.
    :param residual: Whether to add residual connections between multi-layered RNNs.
    :param forget_bias: Initial value of forget biases.
    """
    def __init__(self,
                 cell_type: str,
                 num_hidden: int,
                 num_layers: int,
                 dropout: float,
                 residual: bool = False,
                 forget_bias: float = 0.0) -> None:
        super().__init__()
        self.cell_type = cell_type
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.forget_bias = forget_bias


def get_stacked_rnn(config: RNNConfig, prefix: str) -> mx.rnn.SequentialRNNCell:
    """
    Returns (stacked) RNN cell given parameters.

    :param config: rnn configuration.
    :param prefix: Symbol prefix for RNN.
    :return: RNN cell.
    """

    rnn = mx.rnn.SequentialRNNCell()
    for layer in range(config.num_layers):
        # fhieber: the 'l' in the prefix does NOT stand for 'layer' but for the direction 'l' as in mx.rnn.rnn_cell::517
        # this ensures parameter name compatibility of training w/ FusedRNN and decoding with 'unfused' RNN.
        cell_prefix = "%sl%d_" % (prefix, layer)
        if config.cell_type == C.LSTM_TYPE:
            cell = mx.rnn.LSTMCell(num_hidden=config.num_hidden, prefix=cell_prefix, forget_bias=config.forget_bias)
        elif config.cell_type == C.LNLSTM_TYPE:
            cell = LayerNormLSTMCell(num_hidden=config.num_hidden, prefix=cell_prefix, forget_bias=config.forget_bias)
        elif config.cell_type == C.LNGLSTM_TYPE:
            cell = LayerNormPerGateLSTMCell(num_hidden=config.num_hidden, prefix=cell_prefix,
                                            forget_bias=config.forget_bias)
        elif config.cell_type == C.GRU_TYPE:
            cell = mx.rnn.GRUCell(num_hidden=config.num_hidden, prefix=cell_prefix)
        elif config.cell_type == C.LNGRU_TYPE:
            cell = LayerNormGRUCell(num_hidden=config.num_hidden, prefix=cell_prefix)
        elif config.cell_type == C.LNGGRU_TYPE:
            cell = LayerNormPerGateGRUCell(num_hidden=config.num_hidden, prefix=cell_prefix)
        else:
            raise NotImplementedError()
        if config.residual and layer > 0:
            cell = mx.rnn.ResidualCell(cell)
        rnn.add(cell)

        if config.dropout > 0.:
            # TODO(fhieber): add pervasive dropout?
            rnn.add(mx.rnn.DropoutCell(config.dropout, prefix=cell_prefix + "_dropout"))
    return rnn


class LayerNormLSTMCell(mx.rnn.LSTMCell):
    """
    Long-Short Term Memory (LSTM) network cell with layer normalization across gates.
    Based on Jimmy Lei Ba et al: Layer Normalization (https://arxiv.org/pdf/1607.06450.pdf)

    :param num_hidden: number of RNN hidden units. Number of units in output symbol.
    :param prefix: prefix for name of layers (and name of weight if params is None).
    :param params: RNNParams or None. Container for weight sharing between cells. Created if None.
    :param forget_bias: bias added to forget gate, default 1.0. Jozefowicz et al. 2015 recommends setting this to 1.0.
    :param norm_scale: scale/gain for layer normalization.
    :param norm_shift: shift/bias after layer normalization.
    """

    def __init__(self,
                 num_hidden: int,
                 prefix: str = 'lnlstm_',
                 params: Optional[mx.rnn.RNNParams] = None,
                 forget_bias: float = 1.0,
                 norm_scale: float = 1.0,
                 norm_shift: float = 0.0) -> None:
        super(LayerNormLSTMCell, self).__init__(num_hidden, prefix, params, forget_bias)
        self._iN = LayerNormalization(num_hidden=num_hidden * 4,
                                      prefix="%si2h" % self._prefix,
                                      scale=self.params.get('i2h_scale', shape=(num_hidden * 4,),
                                                            init=mx.init.Constant(value=norm_scale)),
                                      shift=self.params.get('i2h_shift', shape=(num_hidden * 4,),
                                                            init=mx.init.Constant(value=norm_shift)))
        self._hN = LayerNormalization(num_hidden=num_hidden * 4,
                                      prefix="%sh2h" % self._prefix,
                                      scale=self.params.get('h2h_scale', shape=(num_hidden * 4,),
                                                            init=mx.init.Constant(value=norm_scale)),
                                      shift=self.params.get('h2h_shift', shape=(num_hidden * 4,),
                                                            init=mx.init.Constant(value=norm_shift)))
        self._cN = LayerNormalization(num_hidden=num_hidden,
                                      prefix="%sc" % self._prefix,
                                      scale=self.params.get('c_scale', shape=(num_hidden,),
                                                            init=mx.init.Constant(value=norm_scale)),
                                      shift=self.params.get('c_shift', shape=(num_hidden,),
                                                            init=mx.init.Constant(value=norm_shift)))
        self._shape_fix = None

    def __call__(self, inputs, states):
        self._counter += 1
        name = '%st%d_' % (self._prefix, self._counter)
        i2h = mx.sym.FullyConnected(data=inputs, weight=self._iW, bias=self._iB,
                                    num_hidden=self._num_hidden * 4,
                                    name='%si2h' % name)
        if self._counter == 0:
            self._shape_fix = mx.sym.zeros_like(i2h)
        else:
            assert self._shape_fix is not None
        h2h = mx.sym.FullyConnected(data=states[0], weight=self._hW, bias=self._hB,
                                    num_hidden=self._num_hidden * 4,
                                    name='%sh2h' % name)
        gates = self._iN.normalize(i2h) + self._hN.normalize(self._shape_fix + h2h)
        in_gate, forget_gate, in_transform, out_gate = mx.sym.split(gates,
                                                                    num_outputs=4,
                                                                    axis=1,
                                                                    name="%sslice" % name)
        in_gate = mx.sym.Activation(in_gate, act_type="sigmoid",
                                    name='%si' % name)
        forget_gate = mx.sym.Activation(forget_gate, act_type="sigmoid",
                                        name='%sf' % name)
        in_transform = mx.sym.Activation(in_transform, act_type="tanh",
                                         name='%sc' % name)
        out_gate = mx.sym.Activation(out_gate, act_type="sigmoid",
                                     name='%so' % name)
        next_c = mx.sym._internal._plus(forget_gate * states[1], in_gate * in_transform,
                                        name='%sstate' % name)
        next_h = mx.sym._internal._mul(out_gate,
                                       mx.sym.Activation(self._cN.normalize(next_c),
                                                         act_type="tanh"),
                                       name='%sout' % name)
        return next_h, [next_h, next_c]


class LayerNormPerGateLSTMCell(mx.rnn.LSTMCell):
    """
    Long-Short Term Memory (LSTM) network cell with layer normalization per gate.
    Based on Jimmy Lei Ba et al: Layer Normalization (https://arxiv.org/pdf/1607.06450.pdf)

    :param num_hidden: number of RNN hidden units. Number of units in output symbol.
    :param prefix: prefix for name of layers (and name of weight if params is None).
    :param params: RNNParams or None. Container for weight sharing between cells. Created if None.
    :param forget_bias: bias added to forget gate, default 1.0. Jozefowicz et al. 2015 recommends setting this to 1.0.
    :param norm_scale: scale/gain for layer normalization.
    :param norm_shift: shift/bias after layer normalization.
    """

    def __init__(self,
                 num_hidden: int,
                 prefix: str = 'lnglstm_',
                 params: Optional[mx.rnn.RNNParams] = None,
                 forget_bias: float = 1.0,
                 norm_scale: float = 1.0,
                 norm_shift: float = 0.0) -> None:
        super(LayerNormPerGateLSTMCell, self).__init__(num_hidden, prefix, params, forget_bias)
        self._norm_layers = list()  # type: List[LayerNormalization]
        for name in ['i', 'f', 'c', 'o', 's']:
            scale = self.params.get('%s_shift' % name, shape=(num_hidden,),
                                    init=mx.init.Constant(value=norm_shift))
            shift = self.params.get('%s_scale' % name, shape=(num_hidden,),
                                    init=mx.init.Constant(value=norm_scale if name != "f" else forget_bias))
            self._norm_layers.append(
                LayerNormalization(num_hidden, prefix="%s%s" % (self._prefix, name), scale=scale, shift=shift))

    def __call__(self, inputs, states):
        self._counter += 1
        name = '%st%d_' % (self._prefix, self._counter)
        i2h = mx.sym.FullyConnected(data=inputs, weight=self._iW, bias=self._iB,
                                    num_hidden=self._num_hidden * 4,
                                    name='%si2h' % name)
        h2h = mx.sym.FullyConnected(data=states[0], weight=self._hW, bias=self._hB,
                                    num_hidden=self._num_hidden * 4,
                                    name='%sh2h' % name)
        gates = i2h + h2h
        in_gate, forget_gate, in_transform, out_gate = mx.sym.split(
            gates, num_outputs=4, name="%sslice" % name)

        in_gate = self._norm_layers[0].normalize(in_gate)
        forget_gate = self._norm_layers[1].normalize(forget_gate)
        in_transform = self._norm_layers[2].normalize(in_transform)
        out_gate = self._norm_layers[3].normalize(out_gate)

        in_gate = mx.sym.Activation(in_gate, act_type="sigmoid",
                                    name='%si' % name)
        forget_gate = mx.sym.Activation(forget_gate, act_type="sigmoid",
                                        name='%sf' % name)
        in_transform = mx.sym.Activation(in_transform, act_type="tanh",
                                         name='%sc' % name)
        out_gate = mx.sym.Activation(out_gate, act_type="sigmoid",
                                     name='%so' % name)
        next_c = mx.sym._internal._plus(forget_gate * states[1], in_gate * in_transform,
                                        name='%sstate' % name)
        next_h = mx.sym._internal._mul(out_gate,
                                       mx.sym.Activation(self._norm_layers[4].normalize(next_c), act_type="tanh"),
                                       name='%sout' % name)
        return next_h, [next_h, next_c]


class LayerNormGRUCell(mx.rnn.GRUCell):
    """
    Gated Recurrent Unit (GRU) network cell with layer normalization across gates.
    Based on Jimmy Lei Ba et al: Layer Normalization (https://arxiv.org/pdf/1607.06450.pdf)

    :param num_hidden: number of RNN hidden units. Number of units in output symbol.
    :param prefix: prefix for name of layers (and name of weight if params is None).
    :param params: RNNParams or None. Container for weight sharing between cells. Created if None.
    :param norm_scale: scale/gain for layer normalization.
    :param norm_shift: shift/bias after layer normalization.
    """

    def __init__(self,
                 num_hidden: int,
                 prefix: str = 'lngru_',
                 params: Optional[mx.rnn.RNNParams] = None,
                 norm_scale: float = 1.0,
                 norm_shift: float = 0.0) -> None:
        super(LayerNormGRUCell, self).__init__(num_hidden, prefix, params)
        self._iN = LayerNormalization(num_hidden=num_hidden * 3,
                                      prefix="%si2h" % self._prefix,
                                      scale=self.params.get('i2h_scale', shape=(num_hidden * 3,),
                                                            init=mx.init.Constant(value=norm_scale)),
                                      shift=self.params.get('i2h_shift', shape=(num_hidden * 3,),
                                                            init=mx.init.Constant(value=norm_shift)))
        self._hN = LayerNormalization(num_hidden=num_hidden * 3,
                                      prefix="%sh2h" % self._prefix,
                                      scale=self.params.get('h2h_scale', shape=(num_hidden * 3,),
                                                            init=mx.init.Constant(value=norm_scale)),
                                      shift=self.params.get('h2h_shift', shape=(num_hidden * 3,),
                                                            init=mx.init.Constant(value=norm_shift)))
        self._shape_fix = None

    def __call__(self, inputs, states):
        self._counter += 1

        seq_idx = self._counter
        name = '%st%d_' % (self._prefix, seq_idx)
        prev_state_h = states[0]

        i2h = mx.sym.FullyConnected(data=inputs,
                                    weight=self._iW,
                                    bias=self._iB,
                                    num_hidden=self._num_hidden * 3,
                                    name="%s_i2h" % name)
        h2h = mx.sym.FullyConnected(data=prev_state_h,
                                    weight=self._hW,
                                    bias=self._hB,
                                    num_hidden=self._num_hidden * 3,
                                    name="%s_h2h" % name)
        if self._counter == 0:
            self._shape_fix = mx.sym.zeros_like(i2h)
        else:
            assert self._shape_fix is not None

        i2h = self._iN.normalize(i2h)
        h2h = self._hN.normalize(self._shape_fix + h2h)

        i2h_r, i2h_z, i2h = mx.sym.split(i2h, num_outputs=3, name="%s_i2h_slice" % name)
        h2h_r, h2h_z, h2h = mx.sym.split(h2h, num_outputs=3, name="%s_h2h_slice" % name)

        reset_gate = mx.sym.Activation(i2h_r + h2h_r, act_type="sigmoid",
                                       name="%s_r_act" % name)
        update_gate = mx.sym.Activation(i2h_z + h2h_z, act_type="sigmoid",
                                        name="%s_z_act" % name)

        next_h_tmp = mx.sym.Activation(i2h + reset_gate * h2h, act_type="tanh",
                                       name="%s_h_act" % name)

        next_h = mx.sym._internal._plus((1. - update_gate) * next_h_tmp, update_gate * prev_state_h,
                                        name='%sout' % name)

        return next_h, [next_h]


class LayerNormPerGateGRUCell(mx.rnn.GRUCell):
    """
    Gated Recurrent Unit (GRU) network cell with layer normalization per gate.
    Based on Jimmy Lei Ba et al: Layer Normalization (https://arxiv.org/pdf/1607.06450.pdf)

    :param num_hidden: number of RNN hidden units. Number of units in output symbol.
    :param prefix: prefix for name of layers (and name of weight if params is None).
    :param params: RNNParams or None. Container for weight sharing between cells. Created if None.
    :param norm_scale: scale/gain for layer normalization.
    :param norm_shift: shift/bias after layer normalization.
    """

    def __init__(self,
                 num_hidden: int,
                 prefix: str = 'lnggru_',
                 params: Optional[mx.rnn.RNNParams] = None,
                 norm_scale: float = 1.0,
                 norm_shift: float = 0.0) -> None:
        super(LayerNormPerGateGRUCell, self).__init__(num_hidden, prefix, params)
        self._norm_layers = list()  # type: List[LayerNormalization]
        for name in ['r', 'z', 'o']:
            scale = self.params.get('%s_shift' % name, shape=(num_hidden,), init=mx.init.Constant(value=norm_shift))
            shift = self.params.get('%s_scale' % name, shape=(num_hidden,), init=mx.init.Constant(value=norm_scale))
            self._norm_layers.append(
                LayerNormalization(num_hidden, prefix="%s%s" % (self._prefix, name), scale=scale, shift=shift))

    def __call__(self, inputs, states):
        self._counter += 1

        seq_idx = self._counter
        name = '%st%d_' % (self._prefix, seq_idx)
        prev_state_h = states[0]

        i2h = mx.sym.FullyConnected(data=inputs,
                                    weight=self._iW,
                                    bias=self._iB,
                                    num_hidden=self._num_hidden * 3,
                                    name="%s_i2h" % name)
        h2h = mx.sym.FullyConnected(data=prev_state_h,
                                    weight=self._hW,
                                    bias=self._hB,
                                    num_hidden=self._num_hidden * 3,
                                    name="%s_h2h" % name)

        i2h_r, i2h_z, i2h = mx.sym.split(i2h, num_outputs=3, name="%s_i2h_slice" % name)
        h2h_r, h2h_z, h2h = mx.sym.split(h2h, num_outputs=3, name="%s_h2h_slice" % name)

        reset_gate = mx.sym.Activation(self._norm_layers[0].normalize(i2h_r + h2h_r),
                                       act_type="sigmoid", name="%s_r_act" % name)
        update_gate = mx.sym.Activation(self._norm_layers[1].normalize(i2h_z + h2h_z),
                                        act_type="sigmoid", name="%s_z_act" % name)

        next_h_tmp = mx.sym.Activation(self._norm_layers[2].normalize(i2h + reset_gate * h2h),
                                       act_type="tanh", name="%s_h_act" % name)

        next_h = mx.sym._internal._plus((1. - update_gate) * next_h_tmp, update_gate * prev_state_h,
                                        name='%sout' % name)

        return next_h, [next_h]
