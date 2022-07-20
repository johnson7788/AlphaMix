import numpy as np
from test import test_mtl_mix_by_date, max_drawdown, save_res, get_test_setting


if __name__ == '__main__':
    for dataset in ['sz_50']:
        for expert_num in [4]:
        # for expert_num in [3, 4, 5, 6, 7, 8]:
        # for expert_num in [2, 3, 4, 5, 6, 7, 8]:
            # for weight in [0.01, 0.1, 0.5, 1, 5, 10, 50, 0.05, 0.2, 2, 20, 30, 100]:
            # for weight in [0.01, 0.1, 0.5, 1, 5, 10, 50]:
            # for weight in [0.1, 1, 5, 10, 50]:
            for weight in [5]:
                for certainty_weight in [1]:
                # for certainty_weight in [0.05, 0.1, 0.2, 0.5]:
                #     for model in ['MLP_mtl_mix_vratio_{}_{}_{}'.format(expert_num, weight, certainty_weight)]:
                    for model in ['MLP_mtl_mix_vratio_{}_{}'.format(expert_num, weight)]:
                        ret, sr, vol, dd, mdd, cr, sor = [], [], [], [], [], [], []
                        # for seed in range(5):
                        for seed in range(10):
                            net, stock_lst, feature_lst, target, test_date, criterion = get_test_setting(dataset, model, seed)

                            if dataset == 'acl18':
                                _, ret_lst, _ = test_mtl_mix_by_date(model, net, test_date, dataset, stock_lst, feature_lst, target,
                                                                 10, 6, 'mtl', criterion)
                            elif dataset == 'sz_50':
                                _, ret_lst, _ = test_mtl_mix_by_date(model, net, test_date, dataset, stock_lst, feature_lst, target,
                                                                 25, 5, 'mtl', criterion)
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