import numpy as np
import torch
import random
import os
import joblib


def set_random(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_batch_data(start, batch_size, model, X, y, mode, device):
    if model.split('_')[0] in ['LSTM','ALSTM','GRU','SFM','Transformer']:
        batch_X = X[start:start+batch_size, :, :]
    if model.split('_')[0] in ['MLP']:
        batch_X = X[start:start+batch_size, :]
        batch_X = batch_X.reshape((batch_X.shape[0], batch_X.shape[1]*batch_X.shape[2]))
    batch_X = torch.tensor(batch_X, dtype=torch.float32).to(device)
    batch_y = y[start:start + batch_size]
    if mode == 'clf':
        batch_y = torch.tensor(batch_y, dtype=torch.long).to(device)
    elif mode == 'reg':
        batch_y = torch.tensor(batch_y, dtype=torch.float32).to(device)
    return batch_X, batch_y


def get_dataset(dataset, data_dir, seq_length, feature_num, mode):
    if dataset == 'acl18':
        x_train = np.loadtxt(data_dir + 'train_x.txt').reshape(-1, seq_length, feature_num)
        x_val = np.loadtxt(data_dir + 'val_x.txt').reshape(-1, seq_length, feature_num)
        x_test = np.loadtxt(data_dir + 'test_x.txt').reshape(-1, seq_length, feature_num)
        y_train = np.loadtxt(data_dir + 'train_y.txt')
        y_train_label = [1 if item >= 0 else 0 for item in y_train]
        y_val = np.loadtxt(data_dir + 'val_y.txt')
        y_val_label = [1 if item >= 0 else 0 for item in y_val]
        y_test = np.loadtxt(data_dir + 'test_y.txt')
        y_test_label = [1 if item >= 0 else 0 for item in y_test]
    elif dataset == 'sz_50':
        x_train = np.loadtxt(data_dir + 'x_train_60.txt').reshape(-1, seq_length, feature_num)
        x_val = np.loadtxt(data_dir + 'x_val_60.txt').reshape(-1, seq_length, feature_num)
        x_test = np.loadtxt(data_dir + 'x_test_60.txt').reshape(-1, seq_length, feature_num)
        y_train = np.loadtxt(data_dir + 'y_train_60.txt')
        y_train_label = [1 if item >= 0 else 0 for item in y_train]
        y_val = np.loadtxt(data_dir + 'y_val_60.txt')
        y_val_label = [1 if item >= 0 else 0 for item in y_val]
        y_test = np.loadtxt(data_dir + 'y_test_60.txt')
        y_test_label = [1 if item >= 0 else 0 for item in y_test]
    if mode == 'clf':
        data = {'X_train': x_train, 'X_val': x_val, 'X_test': x_test, 'y_train': y_train_label,
                'y_val': y_val_label, 'y_test': y_test_label}
    elif mode == 'reg':
        data = {'X_train': x_train, 'X_val': x_val, 'X_test': x_test, 'y_train': y_train,
                'y_val': y_val, 'y_test': y_test}
    elif mode == 'mtl':
        data = {'X_train': x_train, 'X_val': x_val, 'X_test': x_test, 'y_train': {'target': y_train, 'label': y_train_label},
                'y_val': {'target': y_val, 'label': y_val_label}, 'y_test': {'target': y_test, 'label': y_test_label}}
    return data


def fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, model, seed, mode, device):
    set_random(seed)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    net.train()
    best_acc, best_loss = 0, 1000
    for i in range(train_episodes):
        print('episodes: {}'.format(i))
        for b in range(0, len(y_train), batch_size):
            train_batch_X, train_batch_y = get_batch_data(b, batch_size, model, X_train, y_train, mode, device)
            output = net(train_batch_X)
            loss = criterion(output, train_batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if int(b / batch_size) % 100 == 0:
                with torch.no_grad():
                    val_loss = []
                    count = 0
                    for m in range(0, len(y_val), batch_size):
                        val_batch_X, val_batch_y = get_batch_data(m, batch_size, model, X_val, y_val, mode, device)
                        val_output = net(val_batch_X)
                        if mode == 'clf':
                            count += torch.sum(torch.argmax(val_output, dim=1) == val_batch_y)
                        loss = criterion(val_output, val_batch_y)
                        val_loss.append(loss.item())
                    print('batch:{} val_loss:{}'.format(int(b / batch_size), sum(val_loss) / len(val_loss)))
                    if not os.path.exists('./model/{}/{}/'.format(dataset, model)):
                        os.makedirs('./model/{}/{}/'.format(dataset, model))
                    if mode == 'clf':
                        print('val_acc:{}'.format(count / len(y_val)))
                        if count / len(y_val) > best_acc:
                            print('update')
                            torch.save(net, './model/{}/{}/{}_{}.pth'.format(dataset, model, model, seed))
                            best_acc = count / len(y_val)
                    elif mode == 'reg':
                        if sum(val_loss) / len(val_loss) < best_loss:
                            print('update')
                            torch.save(net, './model/{}/{}/{}_{}.pth'.format(dataset, model, model, seed))
                            best_loss = sum(val_loss) / len(val_loss)




def fit_mix_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, model, seed, mode, device):
    set_random(seed)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    net.train()
    best_acc, best_loss = 0, 100
    for i in range(train_episodes):
        print('episodes: {}'.format(i))
        for b in range(0, len(y_train), batch_size):
            train_batch_X, train_batch_y = get_batch_data(b, batch_size, model, X_train, y_train, mode, device)
            output = net(train_batch_X)
            loss_lst = []
            for out in output:
                loss_lst.append(criterion(out, train_batch_y))
            loss = sum(loss_lst)/len(loss_lst)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if int(b / batch_size) % 100 == 0:
                with torch.no_grad():
                    val_loss = []
                    count = 0
                    for m in range(0, len(y_val), batch_size):
                        val_batch_X, val_batch_y = get_batch_data(m, batch_size, model, X_val, y_val, mode, device)
                        val_output = net(val_batch_X)
                        val_output = torch.mean(torch.stack(val_output), dim=0).squeeze()
                        if mode == 'clf':
                            count += torch.sum(torch.argmax(val_output, dim=1) == val_batch_y)
                        loss = criterion(val_output, val_batch_y)
                        val_loss.append(loss.item())
                    print('batch:{} val_loss:{}'.format(int(b / batch_size), sum(val_loss) / len(val_loss)))
                    if not os.path.exists('./model/{}/{}/'.format(dataset, model)):
                        os.makedirs('./model/{}/{}/'.format(dataset, model))
                    if mode == 'clf':
                        print('val_acc:{}'.format(count / len(y_val)))
                        if count / len(y_val) > best_acc:
                            print('update')
                            torch.save(net, './model/{}/{}/{}_{}.pth'.format(dataset, model, model, seed))
                            best_acc = count / len(y_val)
                    elif mode == 'reg':
                        if sum(val_loss) / len(val_loss) < best_loss:
                            print('update')
                            torch.save(net, './model/{}/{}/{}_{}.pth'.format(dataset, model, model, seed))
                            best_loss = sum(val_loss) / len(val_loss)


def fit_mtl_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, model, seed, mode, device, weight):
    set_random(seed)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    net.train()
    best_acc, best_loss = 0, 100
    for i in range(train_episodes):
        print('episodes: {}'.format(i))
        for b in range(0, len(y_train['label']), batch_size):
            if mode == 'mtl':
                train_batch_X, train_batch_y_label = get_batch_data(b, batch_size, model, X_train, y_train['label'], 'clf', device)
                _, train_batch_y_target = get_batch_data(b, batch_size, model, X_train, y_train['target'], 'reg', device)
            output_label, output_target = net(train_batch_X)
            clf_loss = criterion['clf'](output_label, train_batch_y_label)
            reg_loss = criterion['reg'](output_target, train_batch_y_target)
            loss = clf_loss + weight * reg_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if int(b / batch_size) % 100 == 0:
                with torch.no_grad():
                    val_loss = []
                    count = 0
                    for m in range(0, len(y_val), batch_size):
                        if mode == 'mtl':
                            val_batch_X, val_batch_y_label = get_batch_data(m, batch_size, model, X_val, y_val['label'], 'clf', device)
                            _, val_batch_y_target = get_batch_data(m, batch_size, model, X_val, y_val['target'], 'reg', device)
                        val_output_label, val_output_target = net(val_batch_X)

                        clf_loss = criterion['clf'](val_output_label, val_batch_y_label)
                        reg_loss = criterion['reg'](val_output_target, val_batch_y_target)
                        loss = clf_loss + weight * reg_loss
                        val_loss.append(loss.item())
                    print('batch:{} val_loss:{}'.format(int(b / batch_size), sum(val_loss) / len(val_loss)))
                    if not os.path.exists('./model/{}/{}/'.format(dataset, model)):
                        os.makedirs('./model/{}/{}/'.format(dataset, model))
                    if mode == 'mtl':
                        if sum(val_loss) / len(val_loss) < best_loss:
                            print('update')
                            torch.save(net, './model/{}/{}/{}_{}.pth'.format(dataset, model, model, seed))
                            best_loss = sum(val_loss) / len(val_loss)


def fit_mtl_mix_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, model, seed, mode, device, weight):
    set_random(seed)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    net.train()
    best_acc, best_loss = 0, 100
    for i in range(train_episodes):
        print('episodes: {}'.format(i))
        for b in range(0, len(y_train['label']), batch_size):
            if mode == 'mtl':
                train_batch_X, train_batch_y_label = get_batch_data(b, batch_size, model, X_train, y_train['label'], 'clf', device)
                _, train_batch_y_target = get_batch_data(b, batch_size, model, X_train, y_train['target'], 'reg', device)
            output_label, output_target = net(train_batch_X)
            loss_lst = []
            for ol, ot in zip(output_label, output_target):
                clf_loss = criterion['clf'](ol, train_batch_y_label)
                reg_loss = criterion['reg'](ot, train_batch_y_target)
                loss_ex = clf_loss + weight * reg_loss
                loss_lst.append(loss_ex)
            loss = sum(loss_lst) / len(loss_lst)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if int(b / batch_size) % 100 == 0:
                with torch.no_grad():
                    val_loss = []
                    count = 0
                    for m in range(0, len(y_val), batch_size):
                        if mode == 'mtl':
                            val_batch_X, val_batch_y_label = get_batch_data(m, batch_size, model, X_val, y_val['label'], 'clf', device)
                            _, val_batch_y_target = get_batch_data(m, batch_size, model, X_val, y_val['target'], 'reg', device)
                        val_output_label, val_output_target = net(val_batch_X)
                        val_output_label = torch.mean(torch.stack(val_output_label), dim=0).squeeze()
                        val_output_target = torch.mean(torch.stack(val_output_target), dim=0).squeeze()
                        if mode == 'clf':
                            count += torch.sum(torch.argmax(val_output_label, dim=1) == val_batch_y_label)
                        clf_loss = criterion['clf'](val_output_label, val_batch_y_label)
                        reg_loss = criterion['reg'](val_output_target, val_batch_y_target)
                        loss = clf_loss + weight * reg_loss
                        val_loss.append(loss.item())
                    print('batch:{} val_loss:{}'.format(int(b / batch_size), sum(val_loss) / len(val_loss)))
                    if not os.path.exists('./model/{}/{}/'.format(dataset, model)):
                        os.makedirs('./model/{}/{}/'.format(dataset, model))

                    elif mode == 'mtl':
                        if sum(val_loss) / len(val_loss) < best_loss:
                            print('update')
                            torch.save(net, './model/{}/{}/{}_{}.pth'.format(dataset, model, model, seed))
                            best_loss = sum(val_loss) / len(val_loss)


def fit_mtl_mix_certainty_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, model, seed, mode, device, weight, certainty_weight):
    set_random(seed)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    net.train()
    best_acc, best_loss = 0, 100
    for i in range(train_episodes):
        print('episodes: {}'.format(i))
        for b in range(0, len(y_train['label']), batch_size):
            if mode == 'mtl':
                train_batch_X, train_batch_y_label = get_batch_data(b, batch_size, model, X_train, y_train['label'], 'clf', device)
                _, train_batch_y_target = get_batch_data(b, batch_size, model, X_train, y_train['target'], 'reg', device)
            output_label, output_target = net(train_batch_X)

            '''用到了loss的variance作为certainty，不合理，效果不好'''
            # clf_loss_lst = []
            # reg_loss_lst = []
            # for ol, ot in zip(output_label, output_target):
            #     clf_loss = criterion['clf'](ol, train_batch_y_label)
            #     reg_loss = criterion['reg'](ot, train_batch_y_target)
            #     clf_loss_lst.append(clf_loss)
            #     reg_loss_lst.append(reg_loss)
            # loss_clf = sum(clf_loss_lst) / len(clf_loss_lst) + torch.var(torch.stack(clf_loss_lst))
            # loss_reg = sum(reg_loss_lst) / len(reg_loss_lst) + torch.var(torch.stack(reg_loss_lst))
            # loss = loss_clf + weight * loss_reg

            clf_loss_lst = []
            reg_loss_lst = []
            clf_lst = []
            reg_lst = []
            for ol, ot in zip(output_label, output_target):
                clf_loss = criterion['clf'](ol, train_batch_y_label)
                reg_loss = criterion['reg'](ot, train_batch_y_target)
                clf_loss_lst.append(clf_loss)
                reg_loss_lst.append(reg_loss)
                clf_lst.append(ol)
                reg_lst.append(ot)
            concat = torch.mean(torch.tensor((torch.argmax(torch.stack(clf_lst), dim=2)),dtype=torch.float32), dim=0).unsqueeze(0)
            concat2 = (1 - concat)
            clf_certainty = torch.mean(torch.min(torch.concat((concat, concat2), dim=0), dim=0)[0])
            reg_certainty = torch.mean(torch.var(torch.stack(reg_lst).squeeze(), dim=0))
            loss_clf = sum(clf_loss_lst) / len(clf_loss_lst) + certainty_weight * clf_certainty
            loss_reg = sum(reg_loss_lst) / len(reg_loss_lst) + certainty_weight * reg_certainty
            loss = loss_clf + weight * loss_reg

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if int(b / batch_size) % 100 == 0:
                with torch.no_grad():
                    val_loss = []
                    count = 0
                    for m in range(0, len(y_val), batch_size):
                        if mode == 'mtl':
                            val_batch_X, val_batch_y_label = get_batch_data(m, batch_size, model, X_val, y_val['label'], 'clf', device)
                            _, val_batch_y_target = get_batch_data(m, batch_size, model, X_val, y_val['target'], 'reg', device)
                        val_output_label, val_output_target = net(val_batch_X)
                        val_output_label = torch.mean(torch.stack(val_output_label), dim=0).squeeze()
                        val_output_target = torch.mean(torch.stack(val_output_target), dim=0).squeeze()
                        if mode == 'clf':
                            count += torch.sum(torch.argmax(val_output_label, dim=1) == val_batch_y_label)
                        clf_loss = criterion['clf'](val_output_label, val_batch_y_label)
                        reg_loss = criterion['reg'](val_output_target, val_batch_y_target)
                        loss = clf_loss + weight * reg_loss
                        val_loss.append(loss.item())
                    print('batch:{} val_loss:{}'.format(int(b / batch_size), sum(val_loss) / len(val_loss)))
                    if not os.path.exists('./model/{}/{}/'.format(dataset, model)):
                        os.makedirs('./model/{}/{}/'.format(dataset, model))

                    elif mode == 'mtl':
                        if sum(val_loss) / len(val_loss) < best_loss:
                            print('update')
                            torch.save(net, './model/{}/{}/{}_{}.pth'.format(dataset, model, model, seed))
                            best_loss = sum(val_loss) / len(val_loss)


def my_bce_with_logits_loss(x, y,weight):
    max_val = (-x).clamp_min_(0)
    loss = (1 - y) * x + weight*max_val + weight*torch.log(torch.exp(-max_val) + torch.exp(-x - max_val))
#     loss = (1 - y) * x + max_val + torch.log(torch.exp(-max_val) + torch.exp(-x - max_val))
    loss = loss.mean()
    return loss


def train_router(data, net, batch_size, model_name, dataset, device, episodes, optimizer, router_net, bce_weight, seed):
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']

    for epi in range(episodes):
        train_loss = []
        for b in range(0, len(y_train['label']), batch_size):
            batch_X = X_train[b:b + batch_size, :]
            batch_X = batch_X.reshape((batch_X.shape[0], batch_X.shape[1] * batch_X.shape[2]))
            batch_X = torch.tensor(batch_X, dtype=torch.float32).to(device)
            batch_y_label = y_train['label'][b:b + batch_size]
            batch_y_label = torch.tensor(batch_y_label, dtype=torch.long).to(device)
            batch_y_target = y_train['target'][b:b + batch_size]
            batch_y_target = torch.tensor(batch_y_target,dtype=torch.float32).to(device)
            output_label, output_target = net(batch_X)

            # for i in range(len(output_label)):
            #     batch_router_x = torch.cat((output_label[i], output_target[i]), dim=1).to(device)
            #     batch_router_x = torch.cat((batch_X[:, -10:], batch_router_x), dim=1).to(device)
            #     batch_router_y = (torch.argmax(output_label[i], dim=1) == batch_y_label).float().to(device)
            #     # batch_router_y = torch.tensor((torch.argmax(output_label[i], dim=1) == batch_y_label).long(), dtype=torch.long)
            #
            #     batch_out = router_net(batch_router_x)
            #     # batch_loss = my_bce_loss(batch_out,batch_router_y)
            #     # criterion = nn.BCELoss()
            #     # batch_loss = criterion(batch_out, batch_router_y)
            #     batch_loss = my_bce_with_logits_loss(batch_out, batch_router_y)
            #     train_loss.append(batch_loss)
            #     # batch_loss.backward()
            #     batch_loss.backward(retain_graph=True)
            #     optimizer.step()
            #     optimizer.zero_grad()

            for i in range(len(output_label)):
                batch_router_x = torch.cat((torch.mean(torch.stack(output_label[:i+1]),dim=0)*20, torch.mean(torch.stack(output_target[:i+1]),dim=0)*1e4), dim=1).to(device)
                batch_router_x = torch.cat((batch_X[:, -10:], batch_router_x), dim=1).to(device)
                batch_router_y = (torch.argmax(torch.mean(torch.stack(output_label[:i+1]),dim=0), dim=1) != batch_y_label).float().to(device)
                # batch_router_y = (torch.argmax(output_label[i], dim=1) == batch_y_label).float().to(device)
                # batch_router_y = torch.tensor((torch.argmax(output_label[i], dim=1) == batch_y_label).long(), dtype=torch.long)

                batch_out = router_net(batch_router_x)
                # batch_loss = my_bce_loss(batch_out,batch_router_y)
                # criterion = nn.BCELoss()
                # batch_loss = criterion(batch_out, batch_router_y)
                batch_loss = my_bce_with_logits_loss(batch_out, batch_router_y,bce_weight)
                train_loss.append(batch_loss)
                # batch_loss.backward()
                batch_loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()
        print(sum(train_loss) / len(train_loss))
        if not os.path.exists('./router/{}/{}/seed{}/{}/'.format(dataset,model_name,seed,bce_weight)):
            os.makedirs('./router/{}/{}/seed{}/{}/'.format(dataset,model_name,seed,bce_weight))
        torch.save(router_net, 'router/{}/{}/seed{}/{}/router_epoch{}.pth'.format(dataset,model_name,seed, bce_weight, epi))