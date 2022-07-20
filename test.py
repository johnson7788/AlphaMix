import numpy as np
import torch
import pandas as pd
import statistics
import os
import joblib


def max_drawdown(ret):
    X = [1 + sum(ret[:i]) for i in range(1,len(ret)+1)]
    X = [1] + X
    mdd = 0
    peak = X[0]
    for x in X:
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        if dd > mdd:
            mdd = dd
    return mdd


def save_res(ret_o, sr_o, vol_o, dd_o, mdd_o, cr_o, sor_o,dataset,model):
    ret = [a for a, b, c, d, e, f, g in sorted(zip(ret_o, sr_o, vol_o, dd_o, mdd_o, cr_o, sor_o))]
    sr = [b for a, b, c, d, e, f, g in sorted(zip(ret_o, sr_o, vol_o, dd_o, mdd_o, cr_o, sor_o))]
    vol = [c for a, b, c, d, e, f, g in sorted(zip(ret_o, sr_o, vol_o, dd_o, mdd_o, cr_o, sor_o))]
    dd = [d for a, b, c, d, e, f, g in sorted(zip(ret_o, sr_o, vol_o, dd_o, mdd_o, cr_o, sor_o))]
    mdd = [e for a, b, c, d, e, f, g in sorted(zip(ret_o, sr_o, vol_o, dd_o, mdd_o, cr_o, sor_o))]
    cr = [f for a, b, c, d, e, f, g in sorted(zip(ret_o, sr_o, vol_o, dd_o, mdd_o, cr_o, sor_o))]
    sor = [g for a, b, c, d, e, f, g in sorted(zip(ret_o, sr_o, vol_o, dd_o, mdd_o, cr_o, sor_o))]
    if not os.path.exists('./res/{}/'.format(dataset)):
        os.makedirs('./res/{}/'.format(dataset))
    with open('./res/{}/{}.txt'.format(dataset, model), 'a') as file:
        file.write(str(ret_o) + '\n')
        file.write(str(sr_o) + '\n')
        file.write(str(ret) + '\n')
        file.write(str(sr) + '\n')
        file.write('all 10 seed average:\n')
        file.write('return rate:{}% std:{}%\n'.format(sum(ret) / len(ret) * 100, statistics.stdev(ret) * 100))
        file.write('sharpe ratio:{} std:{}\n'.format(sum(sr) / len(sr), statistics.stdev(sr)))
        file.write('vol:{}% std:{}%\n'.format(sum(vol) / len(vol) * 100, statistics.stdev(vol) * 100))
        file.write('downside deviation:{}% std:{}%\n'.format(sum(dd) / len(dd) * 100, statistics.stdev(dd) * 100))
        file.write('mdd:{}% std:{}%\n'.format(sum(mdd) / len(mdd) * 100, statistics.stdev(mdd) * 100))
        file.write('calmar ratio:{} std:{}\n'.format(sum(cr) / len(cr), statistics.stdev(cr)))
        file.write('sortino ratio:{} std:{}\n'.format(sum(sor) / len(sor), statistics.stdev(sor)))
    file.close()


def get_test_setting(dataset, model, seed):
    if dataset == 'sz_50':
        net = torch.load('./model/{}/{}/{}_{}.pth'.format(dataset, model, model, seed))
        net.eval()
        stock_lst = [601088, 600837, 601628, 601012, 600104, 600588, 600438, 600893,
                     600009, 600016, 600703, 600309, 600547, 600570,
                     601899, 601288, 601318, 601601, 601818, 601888, 600690,
                     600809, 600028, 600030, 600048, 600036, 601398,
                     600050, 600276, 600519, 601166, 603288, 600196, 600031, 600585,
                     600887, 601857, 601211, 600745, 601336, 601668, 601688, 600000]
        feature_lst = ['Norm_Open', 'Norm_High', 'Norm_Low', 'Norm_Close', 'Norm_Vwap', 'Norm_Volume']
        target = 'Day_Ret'
        df = pd.read_csv('./data/{}/{}min/{}_feature.csv'.format(dataset, 60, 600837), index_col=0)
        date_lst = df['Date'].unique()
        test_date = date_lst[794:]
    elif dataset == 'acl18':
        net = torch.load('./model/{}/{}/{}_{}.pth'.format(dataset, model, model, seed))
        net.eval()
        stock_lst = ['AAPL', 'ABB', 'AEP', 'AMGN', 'AMZN', 'BA', 'BAC', 'BBL', 'BCH', 'BHP', 'BP', 'BRK-A', 'BSAC',
                     'BUD', 'C', 'CAT', 'CELG', 'CHL', 'CHTR', 'CMCSA', 'CODI', 'CSCO', 'CVX', 'D', 'DHR', 'DIS', 'DUK',
                     'EXC', 'FB', 'GD', 'GE', 'GOOG', 'HD', 'HON', 'HRG', 'HSBC', 'IEP', 'INTC', 'JNJ', 'JPM', 'KO',
                     'LMT', 'MA', 'MCD', 'MDT', 'MMM', 'MO', 'MRK', 'MSFT', 'NEE', 'NGG', 'NVS', 'ORCL', 'PCG', 'PCLN',
                     'PEP', 'PFE', 'PG', 'PICO', 'PM', 'PPL', 'PTR', 'RDS-B', 'REX', 'SLB', 'SNP', 'SNY', 'SO', 'SPLP',
                     'SRE', 'T', 'TM', 'TOT', 'TSM', 'UL', 'UN', 'UNH', 'UPS', 'UTX', 'V', 'VZ', 'WFC', 'WMT', 'XOM']
        feature_lst = ['zopen', 'zhigh', 'zlow', 'zclose', 'zadj_close', 'zd5', 'zd10', 'zd15', 'zd20', 'zd25', 'zd30']
        target = 'ret'
        df = pd.read_csv('./data/{}/feature/{}.csv'.format(dataset, 'AAPL'), index_col=0)
        date_lst = df['Date'].unique()
        test_date = date_lst[975:]

    if model.split('_')[1] == 'reg':
        criterion = torch.nn.MSELoss()
    elif model.split('_')[1] == 'clf':
        criterion = torch.nn.CrossEntropyLoss()
    elif model.split('_')[1] == 'mtl':
        criterion = {'clf': torch.nn.CrossEntropyLoss(), 'reg': torch.nn.MSELoss()}

    return net, stock_lst, feature_lst, target, test_date, criterion

def get_gbt_test_setting(dataset, model, seed):
    if dataset == 'sz_50':
        net = joblib.load('./model/{}/{}/{}_{}.pkl'.format(dataset, model, model, seed))
        stock_lst = [601088, 600837, 601628, 601012, 600104, 600588, 600438, 600893,
                     600009, 600016, 600703, 600309, 600547, 600570,
                     601899, 601288, 601318, 601601, 601818, 601888, 600690,
                     600809, 600028, 600030, 600048, 600036, 601398,
                     600050, 600276, 600519, 601166, 603288, 600196, 600031, 600585,
                     600887, 601857, 601211, 600745, 601336, 601668, 601688, 600000]
        feature_lst = ['Norm_Open', 'Norm_High', 'Norm_Low', 'Norm_Close', 'Norm_Vwap', 'Norm_Volume']
        target = 'Day_Ret'
        df = pd.read_csv('./data/{}/{}min/{}_feature.csv'.format(dataset, 60, 600837), index_col=0)
        date_lst = df['Date'].unique()
        test_date = date_lst[794:]
    elif dataset == 'acl18':
        net = joblib.load('./model/{}/{}/{}_{}.pkl'.format(dataset, model, model, seed))
        stock_lst = ['AAPL', 'ABB', 'AEP', 'AMGN', 'AMZN', 'BA', 'BAC', 'BBL', 'BCH', 'BHP', 'BP', 'BRK-A', 'BSAC',
                     'BUD', 'C', 'CAT', 'CELG', 'CHL', 'CHTR', 'CMCSA', 'CODI', 'CSCO', 'CVX', 'D', 'DHR', 'DIS', 'DUK',
                     'EXC', 'FB', 'GD', 'GE', 'GOOG', 'HD', 'HON', 'HRG', 'HSBC', 'IEP', 'INTC', 'JNJ', 'JPM', 'KO',
                     'LMT', 'MA', 'MCD', 'MDT', 'MMM', 'MO', 'MRK', 'MSFT', 'NEE', 'NGG', 'NVS', 'ORCL', 'PCG', 'PCLN',
                     'PEP', 'PFE', 'PG', 'PICO', 'PM', 'PPL', 'PTR', 'RDS-B', 'REX', 'SLB', 'SNP', 'SNY', 'SO', 'SPLP',
                     'SRE', 'T', 'TM', 'TOT', 'TSM', 'UL', 'UN', 'UNH', 'UPS', 'UTX', 'V', 'VZ', 'WFC', 'WMT', 'XOM']
        feature_lst = ['zopen', 'zhigh', 'zlow', 'zclose', 'zadj_close', 'zd5', 'zd10', 'zd15', 'zd20', 'zd25', 'zd30']
        target = 'ret'
        df = pd.read_csv('./data/{}/feature/{}.csv'.format(dataset, 'AAPL'), index_col=0)
        date_lst = df['Date'].unique()
        test_date = date_lst[975:]

    if model.split('_')[1] == 'reg':
        criterion = torch.nn.MSELoss()
    elif model.split('_')[1] == 'clf':
        criterion = torch.nn.CrossEntropyLoss()
    elif model.split('_')[1] == 'mtl':
        criterion = {'clf': torch.nn.CrossEntropyLoss(), 'reg': torch.nn.MSELoss()}

    return net, stock_lst, feature_lst, target, test_date, criterion

def test_base_by_date(model_name, net, date_lst, dataset, stock_lst, feature_lst, target, n_step, top_k, mode, criterion):
    stock_df_lst, acc_lst, ret_lst, loss_lst = [], [], [], []
    logit_dic = {'<40':0,'40-45':0,'45-50':0,'50-55':0,'55-60':0,'>60':0}
    for stock in stock_lst:
        if dataset == 'acl18':
            df = pd.read_csv('./data/{}/feature/{}.csv'.format(dataset, stock), index_col=0).reset_index()
        elif dataset == 'sz_50':
            df = pd.read_csv('./data/{}/60min/{}_feature.csv'.format(dataset, stock), index_col=0).reset_index()
        stock_df_lst.append(df)
    for date in date_lst:
        data_x, data_y = [], []
        for stock_df in stock_df_lst:
            if stock_df[stock_df['Date'] == date].shape[0] > 0:
                end_idx = stock_df[stock_df['Date'] == date].index[-1]
                if (end_idx - n_step + 1) >= 0:
                    seq_x = stock_df.iloc[end_idx - n_step + 1:end_idx + 1][feature_lst]
                    seq_y = stock_df.iloc[end_idx][target]
                    if model_name.split('_')[0] in ['LSTM', 'ALSTM', 'GRU', 'SFM','Transformer']:
                        data_x.append(seq_x)
                    elif model_name.split('_')[0] in ['MLP']:
                        data_x.append(seq_x.values.reshape(1, -1))
                    data_y.append(seq_y)
        data_x_date = torch.tensor(np.array(data_x), dtype=torch.float32).to(torch.device('cpu'))
        if mode == 'clf':
            data_y_date = torch.tensor(np.array([1 if item >= 0 else 0 for item in data_y]), dtype=torch.long).to(
                torch.device('cpu'))
        elif mode == 'reg':
            data_y_date = torch.tensor(np.array(data_y), dtype=torch.float32).to(torch.device('cpu'))
        output = net(data_x_date)
        loss = criterion(output, data_y_date)
        if mode == 'clf':
            # certainty = torch.max(torch.softmax(output,dim=1),dim=1)
            certainty = (torch.softmax(output, dim=1)[:,-1])
            for cer in certainty:
                if cer<=0.4:
                    logit_dic['<40']+=1
                elif cer > 0.4 and cer <=0.45:
                    logit_dic['40-45'] += 1
                elif cer > 0.45 and cer <= 0.5:
                    logit_dic['45-50'] += 1
                elif cer > 0.5 and cer <= 0.55:
                    logit_dic['50-55'] += 1
                elif cer > 0.55 and cer <= 0.6:
                    logit_dic['55-60'] += 1
                elif cer > 0.6:
                    logit_dic['>60'] += 1
            correct = torch.sum(torch.argmax(output, dim=1) == data_y_date)
            acc_lst.append((correct / len(data_y_date)))
            logit_lst = list(output[:, -1].cpu().data.numpy())

            # if len(logit_lst) != len(set(logit_lst)):
            #     print('evaluation not correct!!!!!')
            ret = [x for _, x in sorted(zip(logit_lst, data_y))]
            logit = [x for x, _ in sorted(zip(logit_lst, data_y))]
            idx = top_k
            while logit[-1 * idx] == logit[-1 * (idx + 1)]:
                print('dup')
                idx += 1
            ret_lst.append(sum(ret[-1 * idx:]) / idx)
        elif mode == 'reg':
            loss_lst.append(loss)
            tar_lst = list(output.cpu().data.numpy())
            # if len(tar_lst) != len(set(tar_lst)):
            #     print('evaluation not correct!!!!!')
            ret = [x for _, x in sorted(zip(tar_lst, data_y))]
            tar = [x for x, _ in sorted(zip(tar_lst, data_y))]
            idx = top_k
            while tar[-1 * idx] == tar[-1 * (idx + 1)]:
                print('dup')
                idx += 1
            ret_lst.append(sum(ret[-1 * idx:]) / idx)
    return acc_lst, ret_lst, loss_lst


def test_ens_by_date(model_name, net_lst, date_lst, dataset, stock_lst, feature_lst, target, n_step, top_k, mode, criterion):
    stock_df_lst, acc_lst, ret_lst, loss_lst = [], [], [], []
    for stock in stock_lst:
        if dataset == 'acl18':
            df = pd.read_csv('./data/{}/feature/{}.csv'.format(dataset, stock), index_col=0).reset_index()
        elif dataset == 'sz_50':
            df = pd.read_csv('./data/{}/60min/{}_feature.csv'.format(dataset, stock), index_col=0).reset_index()
        stock_df_lst.append(df)
    for date in date_lst:
        data_x, data_y = [], []
        for stock_df in stock_df_lst:
            if stock_df[stock_df['Date'] == date].shape[0] > 0:
                end_idx = stock_df[stock_df['Date'] == date].index[-1]
                if (end_idx - n_step + 1) >= 0:
                    seq_x = stock_df.iloc[end_idx - n_step + 1:end_idx + 1][feature_lst]
                    seq_y = stock_df.iloc[end_idx][target]
                    if model_name.split('_')[0] in ['LSTM', 'ALSTM', 'GRU', 'SFM','Transformer']:
                        data_x.append(seq_x)
                    elif model_name.split('_')[0] in ['MLP']:
                        data_x.append(seq_x.values.reshape(1, -1))
                    data_y.append(seq_y)
        data_x_date = torch.tensor(np.array(data_x), dtype=torch.float32).to(torch.device('cpu'))
        if mode == 'clf':
            data_y_date = torch.tensor(np.array([1 if item >= 0 else 0 for item in data_y]), dtype=torch.long).to(
                torch.device('cpu'))
        elif mode == 'reg':
            data_y_date = torch.tensor(np.array(data_y), dtype=torch.float32).to(torch.device('cpu'))
        output = []
        for net in net_lst:
            output.append(net(data_x_date))
        output = torch.mean(torch.stack(output), dim=0).squeeze()
        loss = criterion(output, data_y_date)
        if mode == 'clf':
            correct = torch.sum(torch.argmax(output, dim=1) == data_y_date)
            acc_lst.append((correct / len(data_y_date)))
            logit_lst = list(output[:, -1].cpu().data.numpy())
            ret = [x for _, x in sorted(zip(logit_lst, data_y))]
            logit = [x for x, _ in sorted(zip(logit_lst, data_y))]
            idx = top_k
            while logit[-1 * idx] == logit[-1 * (idx + 1)]:
                print('dup')
                idx += 1
            ret_lst.append(sum(ret[-1 * idx:]) / idx)
        elif mode == 'reg':
            loss_lst.append(loss)
            tar_lst = list(output.cpu().data.numpy())
            ret = [x for _, x in sorted(zip(tar_lst, data_y))]
            tar = [x for x, _ in sorted(zip(tar_lst, data_y))]
            idx = top_k
            while tar[-1 * idx] == tar[-1 * (idx + 1)]:
                print('dup')
                idx += 1
            ret_lst.append(sum(ret[-1 * idx:]) / idx)
    return acc_lst, ret_lst, loss_lst

def test_gbt_by_date(model_name, net, date_lst, dataset, stock_lst, feature_lst, target, n_step, top_k, mode, criterion):
    stock_df_lst, acc_lst, ret_lst, loss_lst = [], [], [], []
    for stock in stock_lst:
        if dataset == 'acl18':
            df = pd.read_csv('./data/{}/feature/{}.csv'.format(dataset, stock), index_col=0).reset_index()
        elif dataset == 'sz_50':
            df = pd.read_csv('./data/{}/60min/{}_feature.csv'.format(dataset, stock), index_col=0).reset_index()
        stock_df_lst.append(df)
    for date in date_lst:
        data_x, data_y = [], []
        for stock_df in stock_df_lst:
            if stock_df[stock_df['Date'] == date].shape[0] > 0:
                end_idx = stock_df[stock_df['Date'] == date].index[-1]
                if (end_idx - n_step + 1) >= 0:
                    seq_x = stock_df.iloc[end_idx - n_step + 1:end_idx + 1][feature_lst]
                    seq_y = stock_df.iloc[end_idx][target]
                    if model_name.split('_')[0] in ['LSTM', 'ALSTM', 'GRU', 'SFM','Transformer']:
                        data_x.append(seq_x)
                    elif model_name.split('_')[0] in ['MLP', 'lgb', 'cat']:
                        data_x.append(seq_x.values.reshape(1, -1))
                    data_y.append(seq_y)
        data_x_date = np.array(data_x)
        # data_x_date = torch.tensor(np.array(data_x), dtype=torch.float32).to(torch.device('cpu'))
        if mode == 'clf':
            data_y_date = torch.tensor(np.array([1 if item >= 0 else 0 for item in data_y]), dtype=torch.long).to(
                torch.device('cpu'))
        elif mode == 'reg':
            data_y_date = np.array(data_y)

        # output = net(data_x_date)
        output = net.predict(data_x_date.squeeze())
        # loss = criterion(output, data_y_date)
        if mode == 'clf':
            correct = torch.sum(torch.argmax(output, dim=1) == data_y_date)
            acc_lst.append((correct / len(data_y_date)))
            logit_lst = list(output[:, -1].cpu().data.numpy())

            # if len(logit_lst) != len(set(logit_lst)):
            #     print('evaluation not correct!!!!!')
            ret = [x for _, x in sorted(zip(logit_lst, data_y))]
            logit = [x for x, _ in sorted(zip(logit_lst, data_y))]
            idx = top_k
            while logit[-1 * idx] == logit[-1 * (idx + 1)]:
                print('dup')
                idx += 1
            ret_lst.append(sum(ret[-1 * idx:]) / idx)
        elif mode == 'reg':
            tar_lst = list(output)
            # if len(tar_lst) != len(set(tar_lst)):
            #     print('evaluation not correct!!!!!')
            ret = [x for _, x in sorted(zip(tar_lst, data_y))]
            tar = [x for x, _ in sorted(zip(tar_lst, data_y))]
            idx = top_k
            while tar[-1 * idx] == tar[-1 * (idx + 1)]:
                print('dup')
                idx += 1
            ret_lst.append(sum(ret[-1 * idx:]) / idx)
    return acc_lst, ret_lst, loss_lst


def test_mix_by_date(model_name, net, date_lst, dataset, stock_lst, feature_lst, target, n_step, top_k, mode, criterion):
    stock_df_lst, acc_lst, ret_lst, loss_lst = [], [], [], []
    for stock in stock_lst:
        if dataset == 'acl18':
            df = pd.read_csv('./data/{}/feature/{}.csv'.format(dataset, stock), index_col=0).reset_index()
        elif dataset == 'sz_50':
            df = pd.read_csv('./data/{}/60min/{}_feature.csv'.format(dataset, stock), index_col=0).reset_index()
        stock_df_lst.append(df)
    for date in date_lst:
        data_x, data_y = [], []
        for stock_df in stock_df_lst:
            if stock_df[stock_df['Date'] == date].shape[0] > 0:
                end_idx = stock_df[stock_df['Date'] == date].index[-1]
                if (end_idx - n_step + 1) >= 0:
                    seq_x = stock_df.iloc[end_idx - n_step + 1:end_idx + 1][feature_lst]
                    seq_y = stock_df.iloc[end_idx][target]
                    if model_name.split('_')[0] in ['LSTM', 'ALSTM', 'GRU', 'SFM']:
                        data_x.append(seq_x)
                    elif model_name.split('_')[0] in ['MLP']:
                        data_x.append(seq_x.values.reshape(1, -1))
                    data_y.append(seq_y)
        data_x_date = torch.tensor(np.array(data_x), dtype=torch.float32).to(torch.device('cpu'))
        if mode == 'clf':
            data_y_date = torch.tensor(np.array([1 if item >= 0 else 0 for item in data_y]), dtype=torch.long).to(
                torch.device('cpu'))
        elif mode == 'reg':
            data_y_date = torch.tensor(np.array(data_y), dtype=torch.float32).to(torch.device('cpu'))
        output = net(data_x_date)
        output = torch.mean(torch.stack(output), dim=0).squeeze()
        loss = criterion(output, data_y_date)
        if mode == 'clf':
            correct = torch.sum(torch.argmax(output, dim=1) == data_y_date)
            acc_lst.append((correct / len(data_y_date)))
            logit_lst = list(output[:, -1].cpu().data.numpy())

            # if len(logit_lst) != len(set(logit_lst)):
            #     print('evaluation not correct!!!!!')
            ret = [x for _, x in sorted(zip(logit_lst, data_y))]
            logit = [x for x, _ in sorted(zip(logit_lst, data_y))]
            idx = top_k
            while logit[-1 * idx] == logit[-1 * (idx + 1)]:
                print('dup')
                idx += 1
            ret_lst.append(sum(ret[-1 * idx:]) / idx)
        elif mode == 'reg':
            loss_lst.append(loss)
            tar_lst = list(output.cpu().data.numpy())
            # if len(tar_lst) != len(set(tar_lst)):
            #     print('evaluation not correct!!!!!')
            ret = [x for _, x in sorted(zip(tar_lst, data_y))]
            tar = [x for x, _ in sorted(zip(tar_lst, data_y))]
            idx = top_k
            while tar[-1 * idx] == tar[-1 * (idx + 1)]:
                print('dup')
                idx += 1
            ret_lst.append(sum(ret[-1 * idx:]) / idx)
    return acc_lst, ret_lst, loss_lst


def test_mtl_by_date(model_name, net, date_lst, dataset, stock_lst, feature_lst, target, n_step, top_k, mode, criterion):
    stock_df_lst, acc_lst, ret_lst, loss_lst = [], [], [], []
    for stock in stock_lst:
        if dataset == 'acl18':
            df = pd.read_csv('./data/{}/feature/{}.csv'.format(dataset, stock), index_col=0).reset_index()
        elif dataset == 'sz_50':
            df = pd.read_csv('./data/{}/60min/{}_feature.csv'.format(dataset, stock), index_col=0).reset_index()
        stock_df_lst.append(df)
    for date in date_lst:
        data_x, data_y = [], []
        for stock_df in stock_df_lst:
            if stock_df[stock_df['Date'] == date].shape[0] > 0:
                end_idx = stock_df[stock_df['Date'] == date].index[-1]
                if (end_idx - n_step + 1) >= 0:
                    seq_x = stock_df.iloc[end_idx - n_step + 1:end_idx + 1][feature_lst]
                    seq_y = stock_df.iloc[end_idx][target]
                    if model_name.split('_')[0] in ['LSTM', 'ALSTM', 'GRU']:
                        data_x.append(seq_x)
                    elif model_name.split('_')[0] in ['MLP']:
                        data_x.append(seq_x.values.reshape(1, -1))
                    data_y.append(seq_y)
        data_x_date = torch.tensor(np.array(data_x), dtype=torch.float32).to(torch.device('cpu'))

        data_y_date_label = torch.tensor(np.array([1 if item >= 0 else 0 for item in data_y]), dtype=torch.long).to(
                torch.device('cpu'))
        data_y_date_target = torch.tensor(np.array(data_y), dtype=torch.float32).to(torch.device('cpu'))
        output_label, output_target = net(data_x_date)
        clf_loss = criterion['clf'](output_label, data_y_date_label)
        reg_loss = criterion['reg'](output_target, data_y_date_target)
        # 乘一起,  weight的参数！！！
        loss = clf_loss + 1 * reg_loss
        if mode == 'mtl':
            loss_lst.append(loss)
            tar = []
            ret = []
            # for x, y, z, k in sorted(
            #         zip(list(output_target.cpu().data.numpy()), list(output_label[:, 0].cpu().data.numpy()),
            #             list(output_label[:, -1].cpu().data.numpy()), data_y)):
            #     if z > y:
            #         tar.append(x)
            #         ret.append(k)
            for x, y, z, k in sorted(
                    zip(list(output_label[:, -1].cpu().data.numpy()), list(output_label[:, 0].cpu().data.numpy()),
                        list(output_target.cpu().data.numpy()), data_y)):
                if z > 0:
                    tar.append(x)
                    ret.append(k)

            idx = top_k
            if len(tar) > idx:
                while tar[-1 * idx] == tar[-1 * (idx + 1)] and (idx+1 <= len(tar)):
                    print('dup')
                    idx += 1
                ret_lst.append(sum(ret[-1 * idx:]) / idx)
            elif len(ret) == 0:
                ret_lst.append(0)
            else:
                ret_lst.append(sum(ret) / len(ret))

    return acc_lst, ret_lst, loss_lst


def test_mtl_mix_by_date(model_name, net, date_lst, dataset, stock_lst, feature_lst, target, n_step, top_k, mode, criterion):
    stock_df_lst, acc_lst, ret_lst, loss_lst = [], [], [], []
    for stock in stock_lst:
        if dataset == 'acl18':
            df = pd.read_csv('./data/{}/feature/{}.csv'.format(dataset, stock), index_col=0).reset_index()
        elif dataset == 'sz_50':
            df = pd.read_csv('./data/{}/60min/{}_feature.csv'.format(dataset, stock), index_col=0).reset_index()
        stock_df_lst.append(df)
    for date in date_lst:
        data_x, data_y = [], []
        for stock_df in stock_df_lst:
            if stock_df[stock_df['Date'] == date].shape[0] > 0:
                end_idx = stock_df[stock_df['Date'] == date].index[-1]
                if (end_idx - n_step + 1) >= 0:
                    seq_x = stock_df.iloc[end_idx - n_step + 1:end_idx + 1][feature_lst]
                    seq_y = stock_df.iloc[end_idx][target]
                    if model_name.split('_')[0] in ['LSTM', 'ALSTM', 'GRU']:
                        data_x.append(seq_x)
                    elif model_name.split('_')[0] in ['MLP']:
                        data_x.append(seq_x.values.reshape(1, -1))
                    data_y.append(seq_y)
        data_x_date = torch.tensor(np.array(data_x), dtype=torch.float32).to(torch.device('cpu'))

        data_y_date_label = torch.tensor(np.array([1 if item >= 0 else 0 for item in data_y]), dtype=torch.long).to(
                torch.device('cpu'))
        data_y_date_target = torch.tensor(np.array(data_y), dtype=torch.float32).to(torch.device('cpu'))
        output_label, output_target = net(data_x_date)
        output_label = torch.mean(torch.stack(output_label), dim=0).squeeze()
        output_target = torch.mean(torch.stack(output_target), dim=0).squeeze()
        clf_loss = criterion['clf'](output_label, data_y_date_label)
        reg_loss = criterion['reg'](output_target, data_y_date_target)

        loss = clf_loss + 1 * reg_loss
        if mode == 'mtl':
            loss_lst.append(loss)
            tar = []
            ret = []
            # for x, y, z, k in sorted(
            #         zip(list(output_target.cpu().data.numpy()), list(output_label[:, 0].cpu().data.numpy()),
            #             list(output_label[:, -1].cpu().data.numpy()), data_y)):
            #     if z > y:
            #         tar.append(x)
            #         ret.append(k)
            for x, y, z, k in sorted(
                    zip(list(output_label[:, -1].cpu().data.numpy()), list(output_label[:, 0].cpu().data.numpy()),
                        list(output_target.cpu().data.numpy()), data_y)):
                if z > 0:
                    tar.append(x)
                    ret.append(k)

            idx = top_k
            if len(tar) > idx:
                while tar[-1 * idx] == tar[-1 * (idx + 1)] and (idx + 1 <= len(tar)):
                    print('dup')
                    idx += 1
                ret_lst.append(sum(ret[-1 * idx:]) / idx)
            elif len(ret) == 0:
                ret_lst.append(0)
            else:
                ret_lst.append(sum(ret) / len(ret))
    return acc_lst, ret_lst, loss_lst


def test_mtl_mix_router_by_date(model_name, net, date_lst, dataset, stock_lst, feature_lst, target, freq, n_step, top_k, mode, criterion,bce_weight,seed):
    stock_df_lst, acc_lst, ret_lst, loss_lst = [], [], [], []
    for stock in stock_lst:
        if dataset == 'acl18':
            df = pd.read_csv('./data/{}/feature/{}.csv'.format(dataset, stock), index_col=0).reset_index()
        elif dataset == 'sz_50':
            df = pd.read_csv('./data/{}/{}min/{}_feature.csv'.format(dataset,freq,stock), index_col=0).reset_index()
        stock_df_lst.append(df)
    pick = {'1': 0, '2': 0, '3': 0, '4': 0}
    for date in date_lst:
        data_x, data_y = [], []
        for stock_df in stock_df_lst:
            if stock_df[stock_df['Date'] == date].shape[0] > 0:
                end_idx = stock_df[stock_df['Date'] == date].index[-1]
                if (end_idx - n_step + 1) >= 0:
                    seq_x = stock_df.iloc[end_idx - n_step + 1:end_idx + 1][feature_lst]
                    seq_y = stock_df.iloc[end_idx][target]
                    # if model_name in ['LSTM_clf', 'ALSTM_clf', 'SFM_clf', 'GRU_clf', 'LSTM_reg', 'ALSTM_reg', 'SFM_reg',
                    #                   'GRU_reg','GRU_mtl']:
                    if model_name.split('_')[0] in ['LSTM', 'ALSTM', 'GRU']:
                        data_x.append(seq_x)
                    # elif model_name in ['MLP_clf', 'MLP_reg','MLP_mtl']:
                    elif model_name.split('_')[0] in ['MLP']:
                        data_x.append(seq_x.values.reshape(1, -1))
                    data_y.append(seq_y)
        data_x_date = torch.tensor(np.array(data_x), dtype=torch.float32).to(torch.device('cpu'))

        # data_y_date_label = torch.tensor(np.array([1 if item >= 0 else 0 for item in data_y]), dtype=torch.long).to(
        #         torch.device('cpu'))
        data_y_date_label = torch.tensor(np.array([1 if item >= 0 else 0 for item in data_y]), dtype=torch.float32).to(
                torch.device('cpu'))
        data_y_date_target = torch.tensor(np.array(data_y), dtype=torch.float32).to(torch.device('cpu'))
        output_label, output_target = net(data_x_date)

        # router_net = torch.load('router_epoch{}.pth'.format(4))
        router_net = torch.load('./router/{}/{}/seed{}/{}/router_epoch{}.pth'.format(model_name,dataset,seed,bce_weight,3))
        router_net.eval()
        # output_label = torch.mean(torch.stack(output_label), dim=0).squeeze()
        # output_target = torch.mean(torch.stack(output_target), dim=0).squeeze()

        out_label_lst = []
        out_target_lst = []
        for idx in range(output_label[0].shape[0]):
            pick_i = 1
            for i in range(len(output_label)):
                a = data_x_date[:, :, -10:]
                b = torch.mean(torch.stack(output_label[:i+1]),dim=0)
                check_input = torch.cat((data_x_date[:, :, -10:], torch.mean(torch.stack(output_label[:i+1]),dim=0)*10, torch.mean(torch.stack(output_target[:i+1]), dim=0)*1e4),dim=2).squeeze()
                res = torch.sigmoid(router_net(check_input))
                if res[idx]<0.5:
                    pick_i = i+1
                    break
                pick_i = i+1
            pick[str(pick_i)]+=1
            out_label_lst.append(torch.mean(torch.stack(output_label[:pick_i+1]),dim=0)[idx])
            out_target_lst.append(torch.mean(torch.stack(output_target[:pick_i + 1]), dim=0)[idx])
        output_label = (torch.stack(out_label_lst, dim=0)).squeeze()
        output_target = torch.stack(out_target_lst, dim=0).squeeze()

        if mode == 'mtl':
            tar = []
            ret = []
            # for x, y, z, k in sorted(
            #         zip(list(output_target.cpu().data.numpy()), list(output_label[:, 0].cpu().data.numpy()),
            #             list(output_label[:, -1].cpu().data.numpy()), data_y)):
            #     if z > y:
            #         tar.append(x)
            #         ret.append(k)
            for x, y, z, k in sorted(
                    zip(list(output_label[:, -1].cpu().data.numpy()), list(output_label[:, 0].cpu().data.numpy()),
                        list(output_target.cpu().data.numpy()), data_y)):
                if z > 0:
                    tar.append(x)
                    ret.append(k)

            idx = top_k
            if len(tar) > idx:
                while tar[-1 * idx] == tar[-1 * (idx + 1)] and (idx + 1 <= len(tar)):
                    print('dup')
                    idx += 1
                ret_lst.append(sum(ret[-1 * idx:]) / idx)
            elif len(ret) == 0:
                ret_lst.append(0)
            else:
                ret_lst.append(sum(ret) / len(ret))
    print(pick)
    return acc_lst, ret_lst, loss_lst



