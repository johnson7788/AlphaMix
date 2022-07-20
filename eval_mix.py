import numpy as np
import torch
import pandas as pd
import statistics
from test import test_mix_by_date, max_drawdown, save_res, get_test_setting


if __name__ == '__main__':
    # for dataset in ['acl18', 'sz_50']:
    for dataset in ['sz_50']:
        for expert_num in [2, 3, 4, 5, 6, 7, 8]:
            for model in ['MLP_clf_mix_{}'.format(expert_num)]:
            # for model in ['LSTM_clf_mix_{}'.format(expert_num), 'GRU_clf_mix_{}'.format(expert_num), 'MLP_clf_mix_{}'.format(expert_num)]:
                ret, sr, vol, dd, mdd, cr, sor = [], [], [], [], [], [], []
                for seed in range(10):
                    net, stock_lst, feature_lst, target, test_date, criterion = get_test_setting(dataset, model, seed)

                    if dataset == 'acl18':
                        _, ret_lst, _ = test_mix_by_date(model, net, test_date, dataset, stock_lst, feature_lst, target,
                                                          10, 4, 'clf', criterion)
                    elif dataset == 'sz_50':
                        _, ret_lst, _ = test_mix_by_date(model, net, test_date, dataset, stock_lst, feature_lst, target,
                                                          25, 4, 'clf', criterion)
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

