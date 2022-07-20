import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import math


class LSTM_clf(nn.Module):
    def __init__(self, n_features, layer_num, n_hidden):
        super(LSTM_clf, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = layer_num
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=self.n_hidden,
                            num_layers=self.n_layers,
                            batch_first=True)
        self.linear = nn.Linear(self.n_hidden, 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1, :]).squeeze()


class GRU_clf(nn.Module):
    def __init__(self, n_features, layer_num, n_hidden):
        super(GRU_clf, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = layer_num
        self.gru = nn.GRU(input_size=n_features,
                          hidden_size=self.n_hidden,
                          num_layers=self.n_layers,
                          batch_first=True)
        self.linear = torch.nn.Linear(self.n_hidden, 2)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.linear(gru_out[:, -1, :]).squeeze()


class MLP_clf(torch.nn.Module):
    def __init__(self, input_size, n_hidden):
        super(MLP_clf, self).__init__()
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.act = torch.nn.LeakyReLU()
        self.linear1 = nn.Linear(self.input_size, self.n_hidden)
        self.linear2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.linear3 = nn.Linear(self.n_hidden, 2)

    def forward(self, x):
        x = x.reshape((x.shape[0], -1))
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = self.linear3(x)
        return x.squeeze()


class ALSTM_clf(nn.Module):
    def __init__(self, n_features, layer_num, n_hidden):
        super(ALSTM_clf, self).__init__()
        self.n_features = n_features
        self.layer_num = layer_num
        self.n_hidden = n_hidden

        self.net = nn.Sequential()
        self.net.add_module("fc_in", nn.Linear(in_features=self.n_features, out_features=self.n_hidden))
        self.net.add_module("act", nn.Tanh())
        self.rnn = nn.LSTM(input_size=self.n_hidden, hidden_size=self.n_hidden, num_layers=self.layer_num,
                           batch_first=True)
        self.fc_out = nn.Linear(in_features=self.n_hidden * 2, out_features=2)

        self.att_net = torch.nn.Sequential()
        self.att_net.add_module("att_fc_in", nn.Linear(in_features=self.n_hidden, out_features=int(self.n_hidden / 2)))
        self.att_net.add_module("att_act", nn.Tanh())
        self.att_net.add_module("att_fc_out", nn.Linear(in_features=int(self.n_hidden / 2), out_features=1, bias=False))
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))

    def forward(self, inputs):
        rnn_out, _ = self.rnn(self.net(inputs))
        attention_score = self.att_net(rnn_out)
        out_att = torch.mul(rnn_out, attention_score)
        out_att = torch.sum(out_att, dim=1)
        out = self.fc_out(torch.cat((rnn_out[:, -1, :], out_att), dim=1))
        return out


class LSTM_reg(nn.Module):
    def __init__(self, n_features, layer_num, n_hidden):
        super(LSTM_reg, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = layer_num
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=self.n_hidden,
                            num_layers=self.n_layers,
                            batch_first=True)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1, :]).squeeze()


class GRU_reg(nn.Module):
    def __init__(self, n_features, layer_num, n_hidden):
        super(GRU_reg, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = layer_num
        self.gru = nn.GRU(input_size=n_features,
                          hidden_size=self.n_hidden,
                          num_layers=self.n_layers,
                          batch_first=True)
        self.linear = torch.nn.Linear(self.n_hidden, 1)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.linear(gru_out[:, -1, :]).squeeze()


class MLP_reg(torch.nn.Module):
    def __init__(self, input_size, n_hidden):
        super(MLP_reg, self).__init__()
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.act = torch.nn.LeakyReLU()
        self.linear1 = nn.Linear(self.input_size, self.n_hidden)
        self.linear2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.linear3 = nn.Linear(self.n_hidden, 1)

    def forward(self, x):
        x = x.reshape((x.shape[0], -1))
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = self.linear3(x)
        return x.squeeze()


class ALSTM_reg(nn.Module):
    def __init__(self, n_features, layer_num, n_hidden):
        super(ALSTM_reg, self).__init__()
        self.n_features = n_features
        self.layer_num = layer_num
        self.n_hidden = n_hidden

        self.net = nn.Sequential()
        self.net.add_module("fc_in", nn.Linear(in_features=self.n_features, out_features=self.n_hidden))
        self.net.add_module("act", nn.Tanh())
        self.rnn = nn.LSTM(input_size=self.n_hidden, hidden_size=self.n_hidden, num_layers=self.layer_num,
                           batch_first=True)
        self.fc_out = nn.Linear(in_features=self.n_hidden * 2, out_features=1)

        self.att_net = torch.nn.Sequential()
        self.att_net.add_module("att_fc_in", nn.Linear(in_features=self.n_hidden, out_features=int(self.n_hidden / 2)))
        self.att_net.add_module("att_act", nn.Tanh())
        self.att_net.add_module("att_fc_out", nn.Linear(in_features=int(self.n_hidden / 2), out_features=1, bias=False))
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))

    def forward(self, inputs):
        rnn_out, _ = self.rnn(self.net(inputs))
        attention_score = self.att_net(rnn_out)
        out_att = torch.mul(rnn_out, attention_score)
        out_att = torch.sum(out_att, dim=1)
        out = self.fc_out(torch.cat((rnn_out[:, -1, :], out_att), dim=1))
        return out


class SFM_clf(nn.Module):
    def __init__(self, d_feat=6, output_dim=2, freq_dim=3, hidden_size=8, dropout_W=0.0, dropout_U=0.0, device='cpu'):
        super().__init__()
        self.input_dim = d_feat
        self.output_dim = output_dim
        self.freq_dim = freq_dim
        self.hidden_dim = hidden_size
        self.device = device

        self.W_i = nn.Parameter(init.xavier_uniform_(torch.empty((self.input_dim, self.hidden_dim))))
        self.U_i = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_i = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_ste = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim)))
        self.U_ste = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_ste = nn.Parameter(torch.ones(self.hidden_dim))

        self.W_fre = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.freq_dim)))
        self.U_fre = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.freq_dim)))
        self.b_fre = nn.Parameter(torch.ones(self.freq_dim))

        self.W_c = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim)))
        self.U_c = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_c = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_o = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim)))
        self.U_o = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_o = nn.Parameter(torch.zeros(self.hidden_dim))

        self.U_a = nn.Parameter(init.orthogonal_(torch.empty(self.freq_dim, 1)))
        self.b_a = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_p = nn.Parameter(init.xavier_uniform_(torch.empty(self.hidden_dim, self.output_dim)))
        self.b_p = nn.Parameter(torch.zeros(self.output_dim))

        self.activation = nn.Tanh()
        self.inner_activation = nn.Hardsigmoid()
        self.dropout_W, self.dropout_U = (dropout_W, dropout_U)
        self.fc_out = nn.Linear(self.output_dim, 2)

        self.states = []

    def forward(self, input):
        time_step = input.shape[1]

        for ts in range(time_step):
            x = input[:, ts, :]
            if len(self.states) == 0:  # hasn't initialized yet
                self.init_states(x)
            self.get_constants(x)
            p_tm1 = self.states[0]
            h_tm1 = self.states[1]
            S_re_tm1 = self.states[2]
            S_im_tm1 = self.states[3]
            time_tm1 = self.states[4]
            B_U = self.states[5]
            B_W = self.states[6]
            frequency = self.states[7]

            x_i = torch.matmul(x * B_W[0], self.W_i) + self.b_i
            x_ste = torch.matmul(x * B_W[0], self.W_ste) + self.b_ste
            x_fre = torch.matmul(x * B_W[0], self.W_fre) + self.b_fre
            x_c = torch.matmul(x * B_W[0], self.W_c) + self.b_c
            x_o = torch.matmul(x * B_W[0], self.W_o) + self.b_o

            i = self.inner_activation(x_i + torch.matmul(h_tm1 * B_U[0], self.U_i))
            ste = self.inner_activation(x_ste + torch.matmul(h_tm1 * B_U[0], self.U_ste))
            fre = self.inner_activation(x_fre + torch.matmul(h_tm1 * B_U[0], self.U_fre))

            ste = torch.reshape(ste, (-1, self.hidden_dim, 1))
            fre = torch.reshape(fre, (-1, 1, self.freq_dim))

            f = ste * fre

            c = i * self.activation(x_c + torch.matmul(h_tm1 * B_U[0], self.U_c))

            time = time_tm1 + 1

            omega = torch.tensor(2 * np.pi) * time * frequency

            re = torch.cos(omega)
            im = torch.sin(omega)

            c = torch.reshape(c, (-1, self.hidden_dim, 1))

            S_re = f * S_re_tm1 + c * re
            S_im = f * S_im_tm1 + c * im

            A = torch.square(S_re) + torch.square(S_im)

            A = torch.reshape(A, (-1, self.freq_dim)).float()
            A_a = torch.matmul(A * B_U[0], self.U_a)
            A_a = torch.reshape(A_a, (-1, self.hidden_dim))
            a = self.activation(A_a + self.b_a)

            o = self.inner_activation(x_o + torch.matmul(h_tm1 * B_U[0], self.U_o))

            h = o * a
            p = torch.matmul(h, self.W_p) + self.b_p

            self.states = [p, h, S_re, S_im, time, None, None, None]
        self.states = []
        return self.fc_out(p).squeeze()

    def init_states(self, x):
        reducer_f = torch.zeros((self.hidden_dim, self.freq_dim)).to(self.device)
        reducer_p = torch.zeros((self.hidden_dim, self.output_dim)).to(self.device)

        init_state_h = torch.zeros(self.hidden_dim).to(self.device)
        init_state_p = torch.matmul(init_state_h, reducer_p)

        init_state = torch.zeros_like(init_state_h).to(self.device)
        init_freq = torch.matmul(init_state_h, reducer_f)

        init_state = torch.reshape(init_state, (-1, self.hidden_dim, 1))
        init_freq = torch.reshape(init_freq, (-1, 1, self.freq_dim))

        init_state_S_re = init_state * init_freq
        init_state_S_im = init_state * init_freq

        init_state_time = torch.tensor(0).to(self.device)

        self.states = [
            init_state_p,
            init_state_h,
            init_state_S_re,
            init_state_S_im,
            init_state_time,
            None,
            None,
            None,
        ]

    def get_constants(self, x):
        constants = []
        constants.append([torch.tensor(1.0).to(self.device) for _ in range(6)])
        constants.append([torch.tensor(1.0).to(self.device) for _ in range(7)])
        array = np.array([float(ii) / self.freq_dim for ii in range(self.freq_dim)])
        constants.append(torch.tensor(array).to(self.device))

        self.states[5:] = constants


class SFM_reg(nn.Module):
    def __init__(self, d_feat=6, output_dim=1, freq_dim=3, hidden_size=16, dropout_W=0.0, dropout_U=0.0, device='cpu'):
        super().__init__()
        self.input_dim = d_feat
        self.output_dim = output_dim
        self.freq_dim = freq_dim
        self.hidden_dim = hidden_size
        self.device = device

        self.W_i = nn.Parameter(init.xavier_uniform_(torch.empty((self.input_dim, self.hidden_dim))))
        self.U_i = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_i = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_ste = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim)))
        self.U_ste = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_ste = nn.Parameter(torch.ones(self.hidden_dim))

        self.W_fre = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.freq_dim)))
        self.U_fre = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.freq_dim)))
        self.b_fre = nn.Parameter(torch.ones(self.freq_dim))

        self.W_c = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim)))
        self.U_c = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_c = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_o = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim)))
        self.U_o = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_o = nn.Parameter(torch.zeros(self.hidden_dim))

        self.U_a = nn.Parameter(init.orthogonal_(torch.empty(self.freq_dim, 1)))
        self.b_a = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_p = nn.Parameter(init.xavier_uniform_(torch.empty(self.hidden_dim, self.output_dim)))
        self.b_p = nn.Parameter(torch.zeros(self.output_dim))

        self.activation = nn.Tanh()
        self.inner_activation = nn.Hardsigmoid()
        self.dropout_W, self.dropout_U = (dropout_W, dropout_U)
        self.fc_out = nn.Linear(self.output_dim, 1)

        self.states = []

    def forward(self, input):
        time_step = input.shape[1]

        for ts in range(time_step):
            x = input[:, ts, :]
            if len(self.states) == 0:  # hasn't initialized yet
                self.init_states(x)
            self.get_constants(x)
            p_tm1 = self.states[0]
            h_tm1 = self.states[1]
            S_re_tm1 = self.states[2]
            S_im_tm1 = self.states[3]
            time_tm1 = self.states[4]
            B_U = self.states[5]
            B_W = self.states[6]
            frequency = self.states[7]

            x_i = torch.matmul(x * B_W[0], self.W_i) + self.b_i
            x_ste = torch.matmul(x * B_W[0], self.W_ste) + self.b_ste
            x_fre = torch.matmul(x * B_W[0], self.W_fre) + self.b_fre
            x_c = torch.matmul(x * B_W[0], self.W_c) + self.b_c
            x_o = torch.matmul(x * B_W[0], self.W_o) + self.b_o

            i = self.inner_activation(x_i + torch.matmul(h_tm1 * B_U[0], self.U_i))
            ste = self.inner_activation(x_ste + torch.matmul(h_tm1 * B_U[0], self.U_ste))
            fre = self.inner_activation(x_fre + torch.matmul(h_tm1 * B_U[0], self.U_fre))

            ste = torch.reshape(ste, (-1, self.hidden_dim, 1))
            fre = torch.reshape(fre, (-1, 1, self.freq_dim))

            f = ste * fre

            c = i * self.activation(x_c + torch.matmul(h_tm1 * B_U[0], self.U_c))

            time = time_tm1 + 1

            omega = torch.tensor(2 * np.pi) * time * frequency

            re = torch.cos(omega)
            im = torch.sin(omega)

            c = torch.reshape(c, (-1, self.hidden_dim, 1))

            S_re = f * S_re_tm1 + c * re
            S_im = f * S_im_tm1 + c * im

            A = torch.square(S_re) + torch.square(S_im)

            A = torch.reshape(A, (-1, self.freq_dim)).float()
            A_a = torch.matmul(A * B_U[0], self.U_a)
            A_a = torch.reshape(A_a, (-1, self.hidden_dim))
            a = self.activation(A_a + self.b_a)

            o = self.inner_activation(x_o + torch.matmul(h_tm1 * B_U[0], self.U_o))

            h = o * a
            p = torch.matmul(h, self.W_p) + self.b_p

            self.states = [p, h, S_re, S_im, time, None, None, None]
        self.states = []
        return self.fc_out(p).squeeze()

    def init_states(self, x):
        reducer_f = torch.zeros((self.hidden_dim, self.freq_dim)).to(self.device)
        reducer_p = torch.zeros((self.hidden_dim, self.output_dim)).to(self.device)

        init_state_h = torch.zeros(self.hidden_dim).to(self.device)
        init_state_p = torch.matmul(init_state_h, reducer_p)

        init_state = torch.zeros_like(init_state_h).to(self.device)
        init_freq = torch.matmul(init_state_h, reducer_f)

        init_state = torch.reshape(init_state, (-1, self.hidden_dim, 1))
        init_freq = torch.reshape(init_freq, (-1, 1, self.freq_dim))

        init_state_S_re = init_state * init_freq
        init_state_S_im = init_state * init_freq

        init_state_time = torch.tensor(0).to(self.device)

        self.states = [
            init_state_p,
            init_state_h,
            init_state_S_re,
            init_state_S_im,
            init_state_time,
            None,
            None,
            None,
        ]

    def get_constants(self, x):
        constants = []
        constants.append([torch.tensor(1.0).to(self.device) for _ in range(6)])
        constants.append([torch.tensor(1.0).to(self.device) for _ in range(7)])
        array = np.array([float(ii) / self.freq_dim for ii in range(self.freq_dim)])
        constants.append(torch.tensor(array).to(self.device))

        self.states[5:] = constants

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]


class Transformer_clf(nn.Module):
    def __init__(self, d_feat=11, d_model=4, nhead=4, num_layers=2, dropout=0.5, device=None):
        super(Transformer_clf, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.Linear(d_model, 2)
        self.device = device
        self.d_feat = d_feat

    def forward(self, src):
        # src [N, F*T] --> [N, T, F]
        #src = src.reshape(len(src), self.d_feat, -1).permute(0, 2, 1)
        src = self.feature_layer(src)

        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first

        mask = None

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)  # [60, 512, 8]

        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])  # [512, 1]

        return output.squeeze()


class Transformer_reg(nn.Module):
    def __init__(self, d_feat=11, d_model=4, nhead=4, num_layers=2, dropout=0.5, device=None):
        super(Transformer_reg, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.Linear(d_model, 1)
        self.device = device
        self.d_feat = d_feat

    def forward(self, src):
        # src [N, F*T] --> [N, T, F]
        #src = src.reshape(len(src), self.d_feat, -1).permute(0, 2, 1)
        src = self.feature_layer(src)

        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first

        mask = None

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)  # [60, 512, 8]

        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])  # [512, 1]

        return output.squeeze()

class MLP_clf_mix(nn.Module):
    def __init__(self, input_size, n_hidden, expert_num):
        super(MLP_clf_mix, self).__init__()
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.expert_num = expert_num
        self.share_linear = nn.Sequential(
            nn.Linear(self.input_size, self.n_hidden),
            nn.LeakyReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.LeakyReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.LeakyReLU())
        self.experts_out_label = nn.ModuleList([nn.Linear(self.n_hidden, 2) for _ in range(expert_num)])

    def forward(self, x):
        x = self.share_linear(x)
        out_label_lst = []
        for ex_out_label in self.experts_out_label:
            out_label_lst.append(ex_out_label(x))

        return out_label_lst


class GRU_clf_mix(nn.Module):
    def __init__(self, n_features, layer_num, n_hidden, expert_num):
        super(GRU_clf_mix, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = layer_num
        self.expert_num = expert_num
        self.share_gru = nn.GRU(input_size=self.n_features, hidden_size=self.n_hidden, num_layers=self.n_layers,
                                batch_first=True)
        self.experts_out_label = nn.ModuleList([nn.Linear(self.n_hidden, 2) for _ in range(expert_num)])

    def forward(self, x):
        gru_out = self.share_gru(x)[0][:, -1, :]
        out_label_lst = []
        for ex_out_label in self.experts_out_label:
            out_label_lst.append(ex_out_label(gru_out))
        return out_label_lst


class LSTM_clf_mix(nn.Module):
    def __init__(self, n_features, layer_num, n_hidden, expert_num):
        super(LSTM_clf_mix, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = layer_num
        self.expert_num = expert_num
        self.share_lstm = nn.LSTM(input_size=self.n_features, hidden_size=self.n_hidden, num_layers=self.n_layers,
                                  batch_first=True)
        self.experts_out_label = nn.ModuleList([nn.Linear(self.n_hidden, 2) for _ in range(expert_num)])

    def forward(self, x):
        gru_out = self.share_lstm(x)[0][:, -1, :]
        out_label_lst = []
        for ex_out_label in self.experts_out_label:
            out_label_lst.append(ex_out_label(gru_out))
        return out_label_lst


class MLP_mtl(torch.nn.Module):
    def __init__(self, input_size, n_hidden):
        super(MLP_mtl, self).__init__()
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.act = torch.nn.LeakyReLU()
        self.linear1 = nn.Linear(self.input_size, self.n_hidden)
        self.linear2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.out_clf = nn.Linear(self.n_hidden, 2)
        self.out_reg = nn.Linear(self.n_hidden, 1)

    def forward(self, x):
        x = x.reshape((x.shape[0], -1))
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        return self.out_clf(x).squeeze(), self.out_reg(x).squeeze()


class MLP_mtl_mix(nn.Module):
    def __init__(self, input_size, n_hidden, expert_num):
        super(MLP_mtl_mix, self).__init__()
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.expert_num = expert_num
        self.share_linear = nn.Sequential(
            nn.Linear(self.input_size, self.n_hidden),
            nn.LeakyReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.LeakyReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.LeakyReLU())
        self.experts_out_label = nn.ModuleList([nn.Linear(self.n_hidden, 2) for _ in range(expert_num)])
        self.experts_out_target = nn.ModuleList([nn.Linear(self.n_hidden, 1) for _ in range(expert_num)])

    def forward(self, x):
        x = self.share_linear(x)
        out_label_lst = []
        out_target_lst = []
        for ex_out_label, ex_out_target in zip(self.experts_out_label, self.experts_out_target):
            out_label_lst.append(ex_out_label(x))
            out_target_lst.append(ex_out_target(x))
        return out_label_lst, out_target_lst


class GRU_mtl_mix(nn.Module):
    def __init__(self, n_features, layer_num, n_hidden, expert_num):
        super(GRU_mtl_mix, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = layer_num
        self.expert_num = expert_num
        self.share_gru = nn.GRU(input_size=self.n_features, hidden_size=self.n_hidden, num_layers=self.n_layers,
                                batch_first=True)
        self.experts_out_label = nn.ModuleList([nn.Linear(self.n_hidden, 2) for _ in range(expert_num)])
        self.experts_out_target = nn.ModuleList([nn.Linear(self.n_hidden, 1) for _ in range(expert_num)])

    def forward(self, x):
        x = self.share_gru(x)[0][:, -1, :]
        out_label_lst = []
        out_target_lst = []
        for ex_out_label, ex_out_target in zip(self.experts_out_label, self.experts_out_target):
            out_label_lst.append(ex_out_label(x))
            out_target_lst.append(ex_out_target(x))
        return out_label_lst, out_target_lst


class LSTM_mtl_mix(nn.Module):
    def __init__(self, n_features, layer_num, n_hidden, expert_num):
        super(LSTM_mtl_mix, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = layer_num
        self.expert_num = expert_num
        self.share_lstm = nn.LSTM(input_size=self.n_features, hidden_size=self.n_hidden, num_layers=self.n_layers,
                                batch_first=True)
        self.experts_out_label = nn.ModuleList([nn.Linear(self.n_hidden, 2) for _ in range(expert_num)])
        self.experts_out_target = nn.ModuleList([nn.Linear(self.n_hidden, 1) for _ in range(expert_num)])

    def forward(self, x):
        x = self.share_lstm(x)[0][:, -1, :]
        out_label_lst = []
        out_target_lst = []
        for ex_out_label, ex_out_target in zip(self.experts_out_label, self.experts_out_target):
            out_label_lst.append(ex_out_label(x))
            out_target_lst.append(ex_out_target(x))
        return out_label_lst, out_target_lst

class expert_router(torch.nn.Module):
    def __init__(self, input_size, n_hidden):
        super(expert_router, self).__init__()
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.act = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(self.input_size, self.n_hidden)
        self.linear2 = torch.nn.Linear(self.n_hidden, 1)
        # self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.linear1(x))

        x = self.linear2(x)
        # x = self.sig(x)
        return x.squeeze()