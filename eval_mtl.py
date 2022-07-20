import numpy as np
from test import test_mtl_by_date, max_drawdown, save_res, get_test_setting


if __name__ == '__main__':
    for dataset in ['acl18']:
        for weight in [0.05]:
        # for weight in [0.05, 0.2, 0.8, 2, 20, 30, 40, 60, 70, 100]:
        # for weight in [0.01, 0.1, 0.5, 1, 5, 10, 50]:
            for model in ['MLP_mtl_{}'.format(weight)]:
                ret, sr, vol, dd, mdd, cr, sor = [], [], [], [], [], [], []
                for seed in range(10):
                    net, stock_lst, feature_lst, target, test_date, criterion = get_test_setting(dataset, model, seed)

                    if dataset == 'acl18':
                        _, ret_lst, _ = test_mtl_by_date(model, net, test_date, dataset, stock_lst, feature_lst, target,
                                                         10, 4, 'mtl', criterion)
                    elif dataset == 'sz_50':
                        _, ret_lst, _ = test_mtl_by_date(model, net, test_date, dataset, stock_lst, feature_lst, target,
                                                         25, 4, 'mtl', criterion)
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

