"""
Based on the pylego: https://github.com/ankitkv/pylego/blob/master/pylego
"""

import torch
import torch.nn as nn


class MultilayerLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, top_down_transitions: bool = False):
        """

        :param input_size: size of the input in each cell
        :param hidden_size: size of the hidden state
        :param num_layers: number of layers of cells stacked on top of each other
        :param top_down_transitions: whether the hidden state of a cell in a higher layer must be concatenated to
        the input of a cell in the next time step of a immediate lower layer.
        """
        super(MultilayerLSTMCell, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._top_down_transitions = top_down_transitions

        self._input_sizes = None
        self._lstm_cells = None

        self._build_cell()

    def _build_cell(self):
        # The input of cells in higher layers are the hidden states from lower layers.
        # Only the bottom layer receives an external input (observations)
        self._input_sizes = [self._input_size] + [self._hidden_size for _ in range(1, self._num_layers)]

        if self._top_down_transitions:
            for l in range(self._num_layers - 1):
                # The input of a cell in a lower layer will be concatenated with the hidden state of a previous
                # (in time) cell from the immediate upper layer
                self._input_sizes[l] += self._hidden_size

        self._lstm_cells = nn.ModuleList(
            [nn.LSTMCell(self._input_sizes[l], self._hidden_size, bias=True) for l in range(self._num_layers)])

    def forward(self, x: torch.tensor, hx: torch.tensor = None) -> torch.tensor:
        """
        :param x: inputs
        :param hx: [(h_0, c_0), ..., (h_L, c_L)]
        :return:
        """

        if hx is None:
            hx = [None] * self._num_layers
        outputs = []
        input_ = x

        for l in range(self._num_layers):
            if l < self._num_layers - 1 and self._top_down_transitions:
                if hx[l + 1] is None:
                    # In the first time step, use a zero vector per sample
                    previous_hidden = input_.new_zeros([input_.size(0), self._hidden_size])
                else:
                    previous_hidden = hx[l + 1][0]

                input_ = torch.cat([input_, previous_hidden], dim=1)

            # hx will be filled with values from the previous time step when this cell is called by the MultilayerLSTM
            output = self._lstm_cells[l](input_, hx[l])

            # The output of a lower layer will be passed to the upper layer as input
            input_ = output[0]
            outputs.append(output)

        return outputs


class MultilayerLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, top_down_transitions: bool = False):
        super(MultilayerLSTM, self).__init__()
        self._lstm_cell = MultilayerLSTMCell(input_size, hidden_size, num_layers, top_down_transitions)

    def forward(self, x: torch.tensor) -> torch.tensor:
        hx = None
        outputs = []
        for t in range(x.size(1)):
            hx = self._lstm_cell(x[:, t], hx)
            outputs.append(torch.cat([h[:, None, None, :] for (h, c) in hx], dim=2))

        return torch.cat(outputs, dim=1)  # size: batch_size, length, layers, hidden_size

