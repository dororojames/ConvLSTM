import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell  
    Args:
        in_dim (int): Number of dims of in tensor.  
        out_shape (int, int, int): Channel height and width of in tensor as (channel, height, width).  
        kernel_size (int, int): Size of the convolutional kernel.  
        scale (nn.Module): Layer for scaling before Convolution
    """

    def __init__(self, in_dim, out_shape, kernel_size, scale=None, **kwargs):
        super(ConvLSTMCell, self).__init__()
        hidden_dim, h, w = out_shape

        self.scale = scale
        self.conv = nn.Conv2d(in_channels=(in_dim + hidden_dim), out_channels=hidden_dim * 4,
                              kernel_size=kernel_size, padding=kernel_size//2, **kwargs)
        self.norm = nn.InstanceNorm2d(hidden_dim)
        self.activation = nn.ReLU(inplace=True)

        self.h_init = torch.zeros(1, hidden_dim, h, w)
        self.c_init = torch.zeros(1, hidden_dim, h, w)

    def forward(self, X, cur_state):
        if self.scale is not None:
            X = self.scale(X)
        h_cur, c_cur = cur_state

        combined_conv = self.conv(torch.cat([X, h_cur], dim=1))
        It, Ft, Ot, Gt = combined_conv.chunk(4, dim=1)

        c_next = Ft.sigmoid() * c_cur + It.sigmoid() * Gt.tanh()
        h_next = Ot.sigmoid() * c_next.tanh()
        h_next = self.activation(self.norm(h_next))

        return h_next, c_next

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return (self.h_init.repeat(batch_size, 1, 1, 1).to(device),
                self.c_init.repeat(batch_size, 1, 1, 1).to(device))


class ConvLSTM(nn.Module):

    def __init__(self, size, in_dim, hidden_dim, out_dim, kernel_size=3, **kwargs):
        super(ConvLSTM, self).__init__()
        self.num_layers = len(hidden_dim)+1

        # Make sure that `kernel_size` is lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, self.num_layers)

        dims = [in_dim] + hidden_dim + [out_dim]

        self.cells = nn.ModuleList()
        for i in range(self.num_layers):
            lstmcell = ConvLSTMCell(in_dim=dims[i], out_shape=(dims[i+1], size[0], size[1]),
                                    kernel_size=kernel_size[i], **kwargs)
            self.cells.append(lstmcell)

    def forward(self, X, hidden=None):
        """
        Parameters
        ----------
        X:
            5-D Tensor (b, t, c, h, w)
        hidden: todo
            None. todo implement stateful

        Returns
        -------
        layer_out

        """

        # Implement stateful ConvLSTM
        if hidden is not None:
            raise NotImplementedError()
        else:
            hidden = self._init_hidden(batch_size=X.size(0))

        seq_len = X.size(1)
        cur_layer_in = X.transpose(0, 1)

        for l in range(self.num_layers):
            h, c = hidden[l]

            out_inner = []
            for t in range(seq_len):
                h, c = self.cells[l](X=cur_layer_in[t], cur_state=[h, c])
                out_inner.append(h)

            cur_layer_in = out_inner

        output = h
        return output

    def _init_hidden(self, batch_size):
        init_states = [cell.init_hidden(batch_size) for cell in self.cells]
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if isinstance(param, int):
            param = [param] * num_layers
        assert isinstance(param, list)
        while len(param) < num_layers:
            param.append(param[-1])
        return param
