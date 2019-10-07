import numpy as np
from copy import deepcopy

# MSE error
def mse(y_true:np.ndarray,y_predict:np.ndarray):
    return ((y_true - y_predict)**2).mean()

#Specificity defined in the paper ( True positive rate)
def specificity(beta_true:np.ndarray,beta_predict:np.ndarray):
    num_p = (beta_true>0).sum()
    num_correct = ((np.sign(beta_predict)==np.sign(beta_true))*(beta_true>0)).sum()
    return num_correct / num_p

#Sensitivity defined in the paper ( True negative rate)
def sensitivity(beta_true:np.ndarray,beta_predict:np.ndarray):
    num_p = (beta_true < 0).sum()
    num_correct = ((np.sign(beta_predict) == np.sign(beta_true)) * (beta_true < 0)).sum()
    return num_correct / num_p

#GridSearchCV used to fine tune the tuning parameters (similar to sklearn.model_selection.GridSearchCV)
def GridSearchCV(estimator_class,param_grid,X,Y,n_folds=10,loss_func=mse):
    '''
    Fine tune the pre-set parameters of the estimator (eg. lambda in Lasso) use k-fold evaluation
    Notes:
        If predict_X is not None, the "best" estimator will be used to do the prediction.
    '''
    def gen_params(params_combination):
        ps = [{}]
        for p, value in params_combination.items():
            if isinstance(value, tuple) or isinstance(value, list):
                for params in ps:
                    params[p] = value[0]
                expand = []
                for v in value[1:]:
                    tmp = deepcopy(ps)
                    for params in tmp:
                        params[p] = v
                    expand += tmp
                ps += expand
            else:
                for params in ps:
                    params[p] = value
        return ps

    def split_k_fold(n_folds=10,length=70):
        group_n = length // n_folds
        remain = length - group_n*n_folds
        positions = list(zip(range(length),np.random.rand(length).argsort()))
        positions.sort(key=lambda x:x[1])
        k_folds_set = []
        sep0 = 0
        for i in range(n_folds):
            sep = sep0
            sep += group_n
            sep += 1 if i<remain else 0
            val_fold = [i[0] for i in positions[sep0:sep]]
            train_fold = [i[0] for i in (positions[:sep0]+positions[sep:])]
            k_folds_set.append([train_fold,val_fold])
            sep0 = sep
        return k_folds_set

    n = X.shape[0]
    n_folds_set = split_k_fold(n_folds,n)
    params_set = gen_params(param_grid)
    best_loss = 1e20                    #loss
    best_info = {}
    info_set = []
    for params in params_set:
        val_loss = 0
        # use k-fold test to fine tune the 'outside' parameters
        for train_idxs,val_idxs in n_folds_set:
            train_x,train_y = X[train_idxs],Y[train_idxs]
            val_x,val_y = X[val_idxs],Y[val_idxs]
            estimator = estimator_class(**params)
            estimator.fit(train_x, train_y)
            pre_y = estimator.predict(val_x)
            val_loss += loss_func(val_y,pre_y) / n_folds

        #use all data to train model and calculate the train MSE error.
        estimator = estimator_class(**params)
        estimator.fit(X, Y)
        pre = estimator.predict(X)
        train_loss = loss_func(Y,pre)

        info = dict(loss=dict(val_loss=val_loss,train_loss=train_loss),estimator=estimator)
        info_set.append(info)
        if val_loss < best_loss:
            best_loss = val_loss
            best_info = info
    print("best k-fold mean loss: %.5f"%best_loss)
    return best_info,info_set


if __name__ == "__main__":
    from data_genertor import simulation_data
    from models import Lasso

    data = simulation_data(p=600, n=71, seed=100)
    mu, cov = data.gen_mu_cov(case=2)
    data.gen_beta()
    data.gen_X(mu, cov)
    data.gen_Y(std=1, form="norm")

    params_grid = dict(lambda_=[0.1,0.5],
                       max_iter=1000,
                       second_stage=True)
    best_info,info_set = GridSearchCV(Lasso,params_grid,data.X,data.Y,10,mse)

