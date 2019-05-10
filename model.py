import torch
import torch.nn as nn
from torch.autograd import Variable

sigmoid = nn.Sigmoid()
tanh = nn.Tanh()


class ConvLSTMCell(nn.Module):

    def __init__(self, shape, in_dim, hidden_dim, kernel_size, **kwargs):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        shape: (int, int, int)
            Channel height and width of in tensor as (channel, height, width).
        in_dim: int
            Number of dims of in tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.channel, self.height, self.width = shape
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(in_channels=(self.in_dim + self.hidden_dim) * self.channel,
                              out_channels=self.hidden_dim * self.channel * 4,
                              kernel_size=self.kernel_size,
                              padding=(kernel_size[0]//2, kernel_size[1]//2),
                              **kwargs)

    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state

        # concatenate along channel axis
        combined = torch.cat([x, h_cur], dim=1)
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim * self.channel, dim=1)

        i = sigmoid(cc_i)
        f = sigmoid(cc_f)
        o = sigmoid(cc_o)
        g = tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim * self.channel, self.height, self.width)),
                Variable(torch.zeros(batch_size, self.hidden_dim * self.channel, self.height, self.width)))


class ConvLSTM(nn.Module):

    def __init__(self, shape, in_dim, hidden_dim, out_dim, kernel_size, batch_first=True, **kwargs):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)
        self.shape = shape
        self.num_layers = len(hidden_dim)+1

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, self.num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, self.num_layers-1)
        if not len(kernel_size) == len(hidden_dim)+1 == self.num_layers:
            raise ValueError('Inconsistent list length.')

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.layer_dim = [in_dim] + hidden_dim + [out_dim]

        self.kernel_size = kernel_size
        self.batch_first = batch_first

        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            lstmcell = ConvLSTMCell(shape=self.shape, in_dim=self.layer_dim[i], hidden_dim=self.layer_dim[i+1],
                                    kernel_size=self.kernel_size[i], **kwargs)
            self.cell_list.append(lstmcell)

    def forward(self, x, hidden_state=None):
        """
        Parameters
        ----------
        x: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_out
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            x = x.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=x.size(0))

        layer_out_list = []

        seq_len = x.size(1)
        cur_layer_in = x

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]

            out_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](x=cur_layer_in[:, t, :, :, :], cur_state=[h, c])
                out_inner.append(h)

            layer_out = torch.stack(out_inner, dim=1)
            cur_layer_in = layer_out

            layer_out_list.append(layer_out)

        return torch.split(layer_out_list[-1], split_size_or_sections=1, dim=1)[0]

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
