from train import set_random, get_dataset
import lightgbm as lgbm
import catboost as cat
import joblib

if __name__ == '__main__':
    '''train LSTM,GRU,MLP,ALSTM clf on acl18'''
    # set_random(0)
    # data_dir = './data/acl18/split/'
    # seq_length = 10
    # feature_num = 11
    # dataset = 'acl18'
    # mode = 'reg'
    # data = get_dataset(dataset, data_dir, seq_length, feature_num, mode)

    set_random(0)
    data_dir = './data/sz_50_data/'
    seq_length = 25
    feature_num = 6
    mode = 'clf'
    dataset = 'sz_50'
    data = get_dataset(dataset, data_dir, seq_length, feature_num, mode)

    X_train, y_train = data['X_train'].reshape((data['X_train'].shape[0], -1)), data['y_train']
    X_val, y_val = data['X_val'].reshape((data['X_val'].shape[0], -1)), data['y_val']
    for seed in range(10):
        reg = lgbm.LGBMRegressor(max_depth=3, reg_alpha=0.01, learning_rate=0.001, n_estimators=2000, feature_frac=0.8,
                                 random_state=seed)
        reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='mse')
        joblib.dump(reg, 'lgb_{}.pkl'.format(seed))

        # reg = cat.CatBoostRegressor(max_depth=3, learning_rate=0.001, n_estimators=2000, random_seed=seed)
        # reg.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        # joblib.dump(reg, 'cat_{}.pkl'.format(seed))

    # load model
    # gbm_pickle = joblib.load('lgb.pkl')
    # res = gbm_pickle.predict(data['X_test'].reshape((data['X_test'].shape[0], -1)))
    # print(res.shape)
