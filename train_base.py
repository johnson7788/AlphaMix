from nn import *
from train import fit_base_model, set_random, get_dataset


if __name__ == '__main__':
    '''train LSTM,GRU,MLP,ALSTM clf on acl18'''
    set_random(0)
    data_dir = './data/acl18/split/'
    seq_length = 10
    feature_num = 11
    dataset = 'acl18'
    mode = 'clf'
    data = get_dataset(dataset, data_dir, seq_length, feature_num, mode)
    device = torch.device('cpu')
    criterion = nn.CrossEntropyLoss()
    batch_size = 128
    train_episodes = 4
    hidden = 64
    lr = 1e-4

    # net = LSTM_clf(feature_num, 2, hidden).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # for seed in range(10):
    #     fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'LSTM_clf', seed,
    #                    'clf', device)
    #
    # net = GRU_clf(feature_num, 2, hidden).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #
    # for seed in range(10):
    #     fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'GRU_clf', seed,
    #                    'clf', device)
    #
    # net = MLP_clf(seq_length * feature_num, 128).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #
    # for seed in range(10):
    #     fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'MLP_clf', seed,
    #                    'clf', device)
    #
    # net = ALSTM_clf(feature_num, 2, hidden).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #
    # for seed in range(10):
    #     fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'ALSTM_clf', seed,
    #                    'clf', device)
    #
    # net = SFM_clf(d_feat=feature_num).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # for seed in range(10):
    #     fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'SFM_clf', seed,
    #                    'clf', device)

    train_episodes = 10
    net = Transformer_clf(d_feat=11).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for seed in range(10):
        fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'Transformer_clf', seed,
                       'clf', device)

    '''train LSTM,GRU,MLP,ALSTM clf on sz_50'''
    set_random(0)
    data_dir = './data/sz_50_data/'
    seq_length = 25
    feature_num = 6
    mode = 'clf'
    dataset = 'sz_50'
    data = get_dataset(dataset, data_dir, seq_length, feature_num, mode)
    device = torch.device('cpu')
    criterion = nn.CrossEntropyLoss()
    batch_size = 128
    train_episodes = 4
    hidden = 64
    lr = 1e-4

    # net = LSTM_clf(feature_num, 2, hidden).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # for seed in range(10):
    #     fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'LSTM_clf', seed,
    #                    'clf', device)
    #
    # net = GRU_clf(feature_num, 2, hidden).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #
    # for seed in range(10):
    #     fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'GRU_clf', seed,
    #                    'clf', device)
    #
    # net = MLP_clf(seq_length * feature_num, 128).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #
    # for seed in range(10):
    #     fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'MLP_clf', seed,
    #                    'clf', device)
    #
    # net = ALSTM_clf(feature_num, 2, hidden).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #
    # for seed in range(10):
    #     fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'ALSTM_clf', seed,
    #                    'clf', device)
    #
    # net = SFM_clf(d_feat=feature_num).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # for seed in range(10):
    #     fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'SFM_clf', seed,
    #                    'clf', device)

    train_episodes = 10
    net = Transformer_clf(d_feat=6).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for seed in range(10):
        fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'Transformer_clf', seed,
                       'clf', device)

    '''train LSTM,GRU,MLP,ALSTM reg on acl18'''
    set_random(0)
    data_dir = './data/acl18/split/'
    seq_length = 10
    feature_num = 11
    dataset = 'acl18'
    mode = 'reg'
    data = get_dataset(dataset, data_dir, seq_length, feature_num, mode)

    device = torch.device('cpu')
    criterion = nn.MSELoss()
    batch_size = 128
    train_episodes = 4
    hidden = 64
    lr = 1e-4

    # net = LSTM_reg(feature_num, 2, hidden).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # for seed in range(10):
    #     fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'LSTM_reg', seed,
    #                    'reg', device)
    #
    # net = GRU_reg(feature_num, 2, hidden).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #
    # for seed in range(10):
    #     fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'GRU_reg', seed,
    #                    'reg', device)
    #
    # net = MLP_reg(seq_length * feature_num, 128).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #
    # for seed in range(10):
    #     fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'MLP_reg', seed,
    #                    'reg', device)
    #
    # net = ALSTM_reg(feature_num, 2, hidden).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #
    # for seed in range(10):
    #     fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'ALSTM_reg', seed,
    #                    'reg', device)
    #
    # net = SFM_reg(d_feat=feature_num).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # for seed in range(10):
    #     fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'SFM_reg', seed,
    #                    'reg', device)

    train_episodes = 10
    net = Transformer_reg(d_feat=11).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for seed in range(10):
        fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'Transformer_reg', seed,
                       'reg', device)

    '''train LSTM,GRU,MLP,ALSTM reg on sz_50'''
    set_random(0)
    data_dir = './data/sz_50_data/'
    seq_length = 25
    feature_num = 6
    dataset = 'sz_50'
    mode = 'reg'
    data = get_dataset(dataset, data_dir, seq_length, feature_num, mode)
    device = torch.device('cpu')
    criterion = nn.MSELoss()
    batch_size = 128
    train_episodes = 4
    hidden = 64
    lr = 1e-4

    # net = LSTM_reg(feature_num, 2, hidden).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # for seed in range(10):
    #     fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'LSTM_reg', seed,
    #                    'reg', device)
    #
    # net = GRU_reg(feature_num, 2, hidden).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #
    # for seed in range(10):
    #     fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'GRU_reg', seed,
    #                    'reg', device)
    #
    # net = MLP_reg(seq_length * feature_num, 128).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #
    # for seed in range(10):
    #     fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'MLP_reg', seed,
    #                    'reg', device)
    #
    # net = ALSTM_reg(feature_num, 2, hidden).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #
    # for seed in range(10):
    #     fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'ALSTM_reg', seed,
    #                    'reg', device)
    #
    # net = SFM_reg(d_feat=feature_num).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # for seed in range(10):
    #     fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'SFM_reg', seed,
    #                    'reg', device)

    train_episodes = 10
    net = Transformer_reg(d_feat=6).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for seed in range(10):
        fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'Transformer_reg', seed,
                       'reg', device)