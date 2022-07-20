from nn import *
from train import fit_mix_model, set_random, get_dataset


if __name__ == '__main__':
    '''train LSTM,GRU,MLP mix clf on acl18'''
    # set_random(0)
    # data_dir = './data/acl18/split/'
    # seq_length = 10
    # feature_num = 11
    # dataset = 'acl18'
    # mode = 'clf'
    # data = get_dataset(dataset, data_dir, seq_length, feature_num, mode)
    # device = torch.device('cpu')
    # criterion = nn.CrossEntropyLoss()
    # batch_size = 128
    # train_episodes = 4
    # hidden = 64
    # lr = 1e-4

    # for expert_num in [2, 3, 4, 5, 6, 7, 8]:
    #     net = LSTM_clf_mix(feature_num, 2, hidden, expert_num).to(device)
    #     optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #     for seed in range(10):
    #         fit_mix_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'LSTM_clf_mix_{}'.format(expert_num), seed,
    #                       'clf', device)
    # for expert_num in [2, 3, 4, 5, 6, 7, 8]:
    #     net = GRU_clf_mix(feature_num, 2, hidden, expert_num).to(device)
    #     optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #
    #     for seed in range(10):
    #         fit_mix_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'GRU_clf_mix_{}'.format(expert_num), seed,
    #                    'clf', device)

    # for expert_num in [2, 3, 4, 5, 6, 7, 8]:
    #     net = MLP_clf_mix(seq_length * feature_num, hidden, expert_num).to(device)
    #     optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #
    #     for seed in range(10):
    #         fit_mix_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'MLP_clf_mix_{}'.format(expert_num), seed,
    #                    'clf', device)


    '''train LSTM,GRU,MLP mix clf on sz_50'''
    set_random(0)
    data_dir = './data/sz_50_data/'
    seq_length = 25
    feature_num = 6
    mode = 'clf'
    dataset = 'sz_50'
    data = get_dataset(dataset, data_dir, seq_length, feature_num, mode)
    device = torch.device('cpu')
    criterion = nn.CrossEntropyLoss()
    batch_size = 64
    train_episodes = 4
    hidden = 128
    lr = 5e-4

    # for expert_num in [2, 3, 4, 5, 6, 7, 8]:
    #     net = LSTM_clf_mix(feature_num, 2, hidden, expert_num).to(device)
    #     optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #     for seed in range(10):
    #         fit_mix_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset,
    #                       'LSTM_clf_mix_{}'.format(expert_num), seed,
    #                       'clf', device)
    # for expert_num in [2, 3, 4, 5, 6, 7, 8]:
    #     net = GRU_clf_mix(feature_num, 2, hidden, expert_num).to(device)
    #     optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #
    #     for seed in range(10):
    #         fit_mix_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset,
    #                       'GRU_clf_mix_{}'.format(expert_num), seed,
    #                       'clf', device)

    for expert_num in [2, 3, 4, 5, 6, 7, 8]:
        net = MLP_clf_mix(seq_length * feature_num, hidden, expert_num).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        for seed in range(10):
            fit_mix_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset,
                          'MLP_clf_mix_{}'.format(expert_num), seed,
                          'clf', device)

