from nn import *
from train import fit_mtl_mix_model, fit_mtl_mix_certainty_model, set_random, get_dataset


if __name__ == '__main__':
    '''train LSTM,GRU,MLP mix clf on acl18'''
    set_random(0)
    data_dir = './data/acl18/split/'
    seq_length = 10
    feature_num = 11
    dataset = 'acl18'
    mode = 'mtl'
    data = get_dataset(dataset, data_dir, seq_length, feature_num, mode)
    device = torch.device('cpu')
    criterion = {'clf': nn.CrossEntropyLoss(), 'reg': nn.MSELoss()}
    batch_size = 64
    train_episodes = 4
    hidden = 32
    lr = 5e-4

    for expert_num in [3, 4, 5, 6, 7, 8]:
        net = LSTM_mtl_mix(feature_num, 2, hidden, expert_num).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        for weight in [0.1, 1, 5, 10, 50]:
            for certainty_weight in [1]:
                for seed in range(10):
                    fit_mtl_mix_certainty_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset,
                                      'LSTM_mtl_mix_vratio_{}_{}_{}'.format(expert_num, weight, certainty_weight), seed, 'mtl', device, weight, certainty_weight)

    '''train LSTM,GRU,MLP mix clf on sz_50'''
    set_random(0)
    data_dir = './data/sz_50_data/'
    seq_length = 25
    feature_num = 6
    mode = 'mtl'
    dataset = 'sz_50'
    data = get_dataset(dataset, data_dir, seq_length, feature_num, mode)
    device = torch.device('cpu')
    criterion = {'clf': nn.CrossEntropyLoss(), 'reg': nn.MSELoss()}
    batch_size = 64
    train_episodes = 4
    hidden = 32
    lr = 5e-4

    for expert_num in [3, 4, 5, 6, 7, 8]:
        net = LSTM_mtl_mix(feature_num, 2, hidden, expert_num).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        for weight in [0.1, 1, 5, 10, 50]:
            for certainty_weight in [1]:
                for seed in range(10):
                    fit_mtl_mix_certainty_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset,
                                                'LSTM_mtl_mix_vratio_{}_{}_{}'.format(expert_num, weight,
                                                                                     certainty_weight), seed, 'mtl',
                                                device, weight, certainty_weight)