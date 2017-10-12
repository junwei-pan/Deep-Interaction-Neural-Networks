import numpy as np
from sklearn.metrics import roc_auc_score

import utils
from models import LR, FM, PNN1, PNN1_Fixed, PNN2, FNN, CCPM, Fast_CTR, Fast_CTR_Concat,XNN,XNN2


train_file = '/home/ysun1/tensorflow-starter-kit_trunk/product-nets-master/data_cretio/train1M.txt.thres20.yx.0.7'
test_file = '/home/ysun1/tensorflow-starter-kit_trunk/product-nets-master/data_cretio/train1M.txt.thres20.yx.0.3'


# fm_model_file = '../data/fm.model.txt'
print "train_file: ", train_file
print "test_file: ", test_file


input_dim = utils.INPUT_DIM

train_data = utils.read_data(train_file)
# train_data = utils.shuffle(train_data)
test_data = utils.read_data(test_file)

train_size = train_data[0].shape[0]
test_size = test_data[0].shape[0]
num_feas = len(utils.FIELD_SIZES)

min_round = 1
num_round = 1000
early_stop_round = 50
batch_size = 256

field_sizes = utils.FIELD_SIZES
field_offsets = utils.FIELD_OFFSETS


def train(model):
    history_score = []
    print 'epochs\tloss\ttrain-auc\teval-auc'
    for i in range(num_round):
        fetches = [model.optimizer, model.loss]
        if batch_size > 0:
            ls = []
            for j in range(train_size / batch_size +1):#manual batch generation and training
                X_i, y_i = utils.slice(train_data, j * batch_size, batch_size)
                _, l = model.run(fetches, X_i, y_i)
                ls.append(l)
        elif batch_size == -1:
            X_i, y_i = utils.slice(train_data)
            _, l = model.run(fetches, X_i, y_i)
            ls = [l]
        train_preds = model.run(model.y_prob, utils.slice(train_data)[0]) # use the trained model to score whole training dataset
        test_preds = model.run(model.y_prob, utils.slice(test_data)[0]) # score whole testing set
        train_score = roc_auc_score(train_data[1], train_preds) #metrics
        test_score = roc_auc_score(test_data[1], test_preds)
        print '%d\t%f\t%f\t%f' % (i, np.mean(ls), train_score, test_score)
        history_score.append(test_score)
        if i > min_round and i > early_stop_round:
            if np.argmax(history_score) == i - early_stop_round and history_score[-1] - history_score[
                        -1 * early_stop_round] < 1e-5:
                print 'early stop\nbest iteration:\n[%d]\teval-auc: %f' % (
                    np.argmax(history_score), np.max(history_score))
                break
def train2(model):
    history_score = []
    print 'epochs\tloss\ttrain-auc\teval-auc'
    for i in range(num_round):
        fetches = [model.optimizer, model.loss]
        if batch_size > 0:
            ls = []
            for j in range(train_size / batch_size +1):#manual batch generation and training
                X_i, y_i = utils.slice(train_data, j * batch_size, batch_size)
                _, l = model.run(fetches, X_i, y_i)
                ls.append(l)
        elif batch_size == -1:
            X_i, y_i = utils.slice(train_data)
            _, l = model.run(fetches, X_i, y_i)
            ls = [l]
        train_preds = []
        test_preds = []
        for j in range(train_size / batch_size +1):#manual batch generation and training
                X_train, y_train = utils.slice(train_data, j * batch_size, batch_size)
                preds = model.run(model.y_prob, X_train)
                #print(len(preds.tolist()))
                train_preds+=preds.tolist()
        for j in range(test_size / batch_size +1):#manual batch generation and training
                X_test, y_test = utils.slice(test_data, j * batch_size, batch_size)
                preds = model.run(model.y_prob, X_test)
                test_preds+=preds.tolist()
        train_score = roc_auc_score(train_data[1], train_preds) #metrics
        test_score = roc_auc_score(test_data[1], test_preds)
        print '%d\t%f\t%f\t%f' % (i, np.mean(ls), train_score, test_score)
        history_score.append(test_score)
        if i > min_round and i > early_stop_round:
            if np.argmax(history_score) == i - early_stop_round and history_score[-1] - history_score[
                        -1 * early_stop_round] < 1e-5:
                print 'early stop\nbest iteration:\n[%d]\teval-auc: %f' % (
                    np.argmax(history_score), np.max(history_score))
                break

algo = 'pnn1_fixed'
print "algo", algo

if algo == 'lr':
    lr_params = {
        'input_dim': input_dim,
        'opt_algo': 'gd',
        'learning_rate': 0.01,
        'l2_weight': 0,
        'random_seed': 0
    }

    model = LR(**lr_params)
elif algo == 'fm':
    fm_params = {
        'input_dim': input_dim,
        'factor_order': 10,
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'l2_w': 0,
        'l2_v': 0,
    }

    model = FM(**fm_params)
elif algo == 'fnn':
    fnn_params = {
        'layer_sizes': [field_sizes, 1, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'layer_l2': [0, 0],
        'random_seed': 0
    }

    model = FNN(**fnn_params)
elif algo == 'ccpm':
    ccpm_params = {
        'layer_sizes': [field_sizes, 10, 5, 3],
        'layer_acts': ['tanh', 'tanh', 'none'],
        'layer_keeps': [1, 1, 1],
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'random_seed': 0
    }

    model = CCPM(**ccpm_params)
elif algo == 'pnn1':
    pnn1_params = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0
    }

    model = PNN1(**pnn1_params)
elif algo == 'fast_ctr':
    fast_ctr_params = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0
    }

    model = Fast_CTR(**fast_ctr_params)
elif algo == 'fast_ctr_concat':
    fast_ctr_concat_params = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0
    }

    model = Fast_CTR_Concat(**fast_ctr_concat_params)
elif algo == 'pnn1_fixed':
    pnn1_fixed_params = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'layer_l2': [0, 0.1],
        'kernel_l2': 0,
        'random_seed': 0
    }

    model = PNN1_Fixed(**pnn1_fixed_params)
elif algo == 'pnn2':
    pnn2_params = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'gd',
        'learning_rate': 0.01,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0
    }

    model = PNN2(**pnn2_params)
    
elif algo == 'xnn':
    xnn_params = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'gd',
        'learning_rate': 0.01,
        'layer_l2': [0, 0],
        'interaction_l2': 0,
        'random_seed': 0,
        'batch_size':batch_size,
        'norm_type':'l2',
        'gate_type':'mul'
        
    }

    model = XNN(**xnn_params)
elif algo == 'xnn2':#tensor lists
    xnn2_params = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'gd',
        'learning_rate': 0.01,
        'layer_l2': [0, 0],
        'interaction_l2': 0,
        'random_seed': 0,
        'batch_size':batch_size,
        'norm_type':'l2',
        'gate_type':'mul'
        
    }

    model = XNN2(**xnn2_params)

if algo in {'fnn', 'ccpm', 'pnn1', 'pnn1_fixed', 'pnn2', 'fast_ctr', 'fast_ctr_concat','xnn','xnn2'}:
    train_data = utils.split_data(train_data)
    test_data = utils.split_data(test_data)

train2(model)
