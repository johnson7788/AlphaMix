from nn import *
from train import fit_mtl_model, set_random, get_dataset


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
    hidden = 128
    lr = 5e-4

    # for weight in [0.01, 0.1, 0.5, 1, 5, 10, 50]:
    for weight in [0.05, 0.2, 0.8, 2, 20, 30, 40, 60, 70, 100]:
        net = MLP_mtl(seq_length * feature_num, hidden).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        for seed in range(10):
            fit_mtl_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'MLP_mtl_{}'.format(weight), seed,
                          'mtl', device, weight)


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
    hidden = 128
    lr = 5e-4

    # for weight in [0.01, 0.1, 0.5, 1, 5, 10, 50]:
    for weight in [0.05, 0.2, 0.8, 2, 20, 30, 40, 60, 70, 100]:
        net = MLP_mtl(seq_length * feature_num, hidden).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        for seed in range(10):
            fit_mtl_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset,
                          'MLP_mtl_{}'.format(weight), seed,
                          'mtl', device, weight)

