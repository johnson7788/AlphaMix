import numpy as np
import torch
import pandas as pd
import statistics
from test import test_mtl_mix_router_by_date, max_drawdown, save_res,test_mtl_mix_by_date


if __name__ == '__main__':
    '''eval LSTM,GRU,MLP,ALSTM clf on acl18'''
    #for model in ['LSTM_clf', 'GRU_clf', 'MLP_clf', 'ALSTM_clf']:
    # for expert_num in [2, 3, 4, 5, 6, 7, 8]:
    #for expert_num in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    for bce_weight in [1.6, 1.7, 1.8, 1.9]:
    # for bce_weight in [1.1,1.2,1.3,1.4,1.6,1.7,1.8,1.9]:
        print(bce_weight)
        for expert_num in [4]:
            # for weight in [0.01,0.1,1,5,10,50]:
            # for weight in [0.02, 0.2, 2, 20, 100]:
            for weight in [5]:
                for model in ['MLP_mtl_mix_vratio_{}_{}'.format(expert_num, weight)]:
                # for model in ['MLP_mtl_mix_{}_{}'.format(expert_num,weight),'GRU_mtl_mix_{}_{}'.format(expert_num,weight)]:
                    loss, ret, sr, vol, dd, mdd, cr, sor = [], [], [], [], [], [], [], []
                    print(model)
                    dataset = 'acl18'
                    print(dataset)
                    for seed in range(5):
                        net = torch.load('./model/{}/{}/{}_{}.pth'.format(dataset,model,model,seed))
                        net.eval()
                        stock_lst = ['AAPL','ABB','AEP','AMGN','AMZN','BA','BAC','BBL','BCH','BHP','BP','BRK-A','BSAC',
                         'BUD','C','CAT','CELG','CHL','CHTR','CMCSA','CODI','CSCO','CVX','D','DHR','DIS','DUK','EXC','FB','GD',
                        'GE','GOOG','HD','HON','HRG','HSBC','IEP','INTC','JNJ','JPM','KO','LMT','MA','MCD','MDT','MMM','MO',
                         'MRK','MSFT','NEE','NGG','NVS','ORCL','PCG','PCLN','PEP','PFE','PG','PICO','PM','PPL','PTR','RDS-B',
                         'REX','SLB','SNP','SNY','SO','SPLP','SRE','T','TM','TOT','TSM','UL','UN','UNH','UPS','UTX','V','VZ',
                         'WFC','WMT','XOM']
                        feature_lst = ['zopen','zhigh','zlow','zclose','zadj_close','zd5','zd10','zd15','zd20','zd25','zd30']
                        target = 'ret'

                        df = pd.read_csv('./data/{}/feature/{}.csv'.format(dataset, 'AAPL'), index_col=0)
                        date_lst = df['Date'].unique()
                        test_date = date_lst[975:]

                        criterion = {'clf':torch.nn.CrossEntropyLoss(),'reg':torch.nn.MSELoss()}
                        # _, ret_lst, loss_lst = test_mtl_mix_by_date(model, net, test_date, dataset, stock_lst,
                        #                                                    feature_lst, target, 60, 10, 4, 'mtl', criterion)
                        _, ret_lst, loss_lst = test_mtl_mix_router_by_date(model,net,test_date,dataset,stock_lst,feature_lst,target,60,10,4,'mtl',criterion,bce_weight, seed)
                        # loss.append((sum(loss_lst)/len(loss_lst)).item())
                        ret.append(sum(ret_lst))
                        sr.append(sum(ret_lst)/np.std(ret_lst)/np.sqrt(len(ret_lst)))
                        vol.append(np.std(ret_lst))
                        mdd.append(max_drawdown(ret_lst))
                        cr.append(sum(ret_lst)/max_drawdown(ret_lst))
                        neg_ret_lst = []
                        for day_ret in ret_lst:
                            if day_ret < 0:
                                neg_ret_lst.append(day_ret)
                        dd.append(np.std(neg_ret_lst))
                        sor.append(sum(ret_lst)/np.std(neg_ret_lst)/np.sqrt(len(ret_lst)))
                    save_res(ret, sr, vol, dd, mdd, cr, sor,dataset,model+'_{}'.format(bce_weight))

    for bce_weight in [1.4]:
    # for bce_weight in [1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 1.8, 1.9]:
    # for bce_weight in [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,3,4,5,6,7,8,9,10]:
        print(bce_weight)
        for expert_num in [4]:
            # for weight in [0.01,0.1,1,5,10,50]:
            # for weight in [0.02, 0.2, 2, 20, 100]:
            for weight in [5]:
                for model in ['MLP_mtl_mix_vratio_{}_{}'.format(expert_num, weight)]:
                # for model in ['MLP_mtl_mix_{}_{}'.format(expert_num,weight),'GRU_mtl_mix_{}_{}'.format(expert_num,weight)]:
                    loss, ret, sr, vol, dd, mdd, cr, sor = [], [], [], [], [], [], [], []
                    print(model)
                    dataset = 'sz_50'
                    print(dataset)
                    for seed in range(10):
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

                        criterion = {'clf':torch.nn.CrossEntropyLoss(),'reg':torch.nn.MSELoss()}
                        # _, ret_lst, loss_lst = test_mtl_mix_by_date(model, net, test_date, dataset, stock_lst,
                        #                                                    feature_lst, target, 60, 10, 4, 'mtl', criterion)
                        _, ret_lst, loss_lst = test_mtl_mix_router_by_date(model,net,test_date,dataset,stock_lst,feature_lst,target,60,25,4,'mtl',criterion,bce_weight,seed)
                        # loss.append((sum(loss_lst)/len(loss_lst)).item())
                        ret.append(sum(ret_lst))
                        sr.append(sum(ret_lst)/np.std(ret_lst)/np.sqrt(len(ret_lst)))
                        vol.append(np.std(ret_lst))
                        mdd.append(max_drawdown(ret_lst))
                        cr.append(sum(ret_lst)/max_drawdown(ret_lst))
                        neg_ret_lst = []
                        for day_ret in ret_lst:
                            if day_ret < 0:
                                neg_ret_lst.append(day_ret)
                        dd.append(np.std(neg_ret_lst))
                        sor.append(sum(ret_lst)/np.std(neg_ret_lst)/np.sqrt(len(ret_lst)))
                    save_res(ret, sr, vol, dd, mdd, cr, sor, dataset, model+'_{}'.format(bce_weight))