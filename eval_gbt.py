import numpy as np
import torch
import pandas as pd
import statistics
from test import test_gbt_by_date, max_drawdown, save_res, get_gbt_test_setting


if __name__ == '__main__':
    for dataset in [ 'sz_50']:
        for model in ['lgb_reg']:
        # for model in ['SFM_reg']:
        # for model in ['LSTM_reg', 'GRU_reg', 'MLP_reg', 'ALSTM_reg']:
            ret, sr, vol, dd, mdd, cr, sor = [], [], [], [], [], [], []
            for seed in range(10):
                net, stock_lst, feature_lst, target, test_date, criterion = get_gbt_test_setting(dataset, model, seed)

                if dataset == 'acl18':
                    _, ret_lst, _ = test_gbt_by_date(model, net, test_date, dataset, stock_lst, feature_lst, target,
                                                      10, 4, 'reg', criterion)
                elif dataset == 'sz_50':
                    _, ret_lst, _ = test_gbt_by_date(model, net, test_date, dataset, stock_lst, feature_lst, target,
                                                      25, 4, 'reg', criterion)
                ret.append(sum(ret_lst))
                sr.append(sum(ret_lst) / np.std(ret_lst) / np.sqrt(len(ret_lst)))
                vol.append(np.std(ret_lst))
                mdd.append(max_drawdown(ret_lst))
                cr.append(sum(ret_lst) / max_drawdown(ret_lst))
                neg_ret_lst = []
                for day_ret in ret_lst:
                    if day_ret < 0:
                        neg_ret_lst.append(day_ret)
                dd.append(np.std(neg_ret_lst))
                sor.append(sum(ret_lst) / np.std(neg_ret_lst) / np.sqrt(len(ret_lst)))
            save_res(ret, sr, vol, dd, mdd, cr, sor, dataset, model)