import numpy as np
from copy import deepcopy
from scipy.optimize import minimize,minimize_scalar

def _soft_thresholding(x, alpha):
    """
        A function used when fitting Lasso & MCP model by Coordinate Descent algorithm.
    """
    if x > 0.0 and alpha < abs(x):
        return x - alpha
    elif x < 0.0 and alpha < abs(x):
        return x + alpha
    else:
        return 0.0


def second_stage_thresholding(betas,p):
    """
        a second - stage “thresholding” procedure proposed in the paper
    """
    betas_ = deepcopy(betas)
    tmp = np.abs(betas_)
    # tmp = tmp[tmp>0]
    tmp.sort()
    sigma_hat = tmp[:(len(tmp)//2)].std()
    T = sigma_hat * np.sqrt(2*np.log(p))
    betas_ = np.where(np.abs(betas_)>=T,betas_,0)
    return betas_


class Lasso():

    def __init__(self, lambda_: float = 1.0, max_iter: int = 1000,
                 second_stage: bool =True,silence: bool =True) -> None:
        self.lambda_: float = lambda_
        self.max_iter: int = max_iter
        self.second_stage: bool = second_stage
        self.coef_ = None
        self.intercept_ = None
        self.silence = silence

    def fit(self, X: np.ndarray, Y: np.ndarray):
        ######normalize data ( x&y ) -> zero mean
        #x
        n = X.shape[0]
        p = X.shape[1]
        l = np.ones((n,1))
        x_mean = X.mean(axis=0)
        x_var = X.var(axis=0)
        x = X-np.dot(l, x_mean.reshape(1, p))
        #y
        y_mean = Y.mean()
        y = Y - y_mean
        ######iteration
        beta = np.zeros(p)
        pre_beta = np.zeros(p)
        r = y.copy()   # residual
        pre_r = y.copy()
        for iteration in range(1, (1+self.max_iter)):
            for j in range(p):
                r_j = r + beta[j] * x[:, j]
                z = np.dot(x[:, j].T, r_j) / n
                #update
                beta[j] = _soft_thresholding(z, self.lambda_) / x_var[j]
                r = r - (beta[j] - pre_beta[j]) * x[:, j]

            mse = (r**2).mean()
            pre_mse = (pre_r**2).mean()
            if (pre_mse-mse)/pre_mse < 0.0001:
                if not self.silence:
                    print("Nearly stop improve in iter %d" % iteration)
                    print("previous mse:%.4f , current mse:%.4f"%(pre_mse,mse))
                if pre_mse < mse:
                    beta = pre_beta.copy()
                break
            pre_r = r
            pre_beta = beta.copy()

        ######compute the original beta s and the intercept
        beta_ = beta.copy()
        if self.second_stage:
            beta_ = second_stage_thresholding(beta_, p)
        intercept = - (x_mean * beta_).sum() + y_mean
        self.coef_ = beta_
        self.intercept_ = intercept
        return self

    def predict(self, X: np.ndarray):
        y = np.dot(X, self.coef_)
        y += self.intercept_ * np.ones(len(y))
        return y


class MCP():

    def __init__(self, lambda_: float = 0.1, gamma_:float = 1.5,
                 max_iter: int = 1000, second_stage: bool =False,silence: bool =True) -> None:
        self.lambda_: float = lambda_
        self.gamma_: float = gamma_     # gamma_ should be larger than 1
        self.max_iter: int = max_iter
        self.second_stage: bool = second_stage
        self.coef_ = None
        self.intercept_ = None
        self.silence = silence

    def fit(self, X: np.ndarray, Y: np.ndarray):
        ######normalize data ( x&y ) -> zero mean
        #x
        n = X.shape[0]
        p = X.shape[1]
        l = np.ones((n,1))
        x_mean = X.mean(axis=0)
        x_var = X.var(axis=0)
        x = X-np.dot(l, x_mean.reshape(1, p))
        #y
        y_mean = Y.mean()
        y = Y - y_mean
        ######iteration
        beta = np.zeros(p)
        pre_beta = np.zeros(p)
        r = y.copy()   # residual
        pre_r = y.copy()
        for iteration in range(1, (1+self.max_iter)):
            for j in range(p):
                r_j = r + beta[j] * x[:, j]
                z = np.dot(x[:, j].T, r_j) / n
                #update
                if np.abs(z) <= self.lambda_ * self.gamma_:
                    beta[j] = _soft_thresholding(z, self.lambda_) / (x_var[j] * (1-1/self.gamma_))
                else:
                    beta[j] = z / x_var[j]
                r = r - (beta[j] - pre_beta[j]) * x[:, j]
            mse = (r**2).mean()
            pre_mse = (pre_r**2).mean()
            if (pre_mse-mse)/pre_mse < 0.0001:
                if not self.silence:
                    print("Nearly stop improve in iter %d" % iteration)
                    print("previous mse:%.4f , current mse:%.4f"%(pre_mse,mse))
                if pre_mse < mse:
                    beta = pre_beta.copy()
                break
            pre_r = r
            pre_beta = beta.copy()

        ######compute the original beta s and the intercept
        beta_ = beta.copy()
        if self.second_stage:
            beta_ = second_stage_thresholding(beta_, p)
        intercept = - (x_mean * beta_).sum() + y_mean
        self.coef_ = beta_
        self.intercept_ = intercept
        return self

    def predict(self, X: np.ndarray):
        y = np.dot(X, self.coef_)
        y += self.intercept_ * np.ones(len(y))
        return y


class spline_Lasso():

    def __init__(self, lambda_1: float = 0.1, lambda_2: float=0.1,
                 max_iter: int = 1000, second_stage: bool =True,silence: bool =True) -> None:
        self.lambda_1: float = lambda_1
        self.lambda_2: float = lambda_2
        self.max_iter: int = max_iter
        self.second_stage: bool = second_stage
        self.coef_ = None
        self.intercept_ = None
        self.silence = silence

    def fit(self, X: np.ndarray, Y: np.ndarray):
        ######normalize data ( x&y ) -> zero mean
        #x
        n = X.shape[0]
        p = X.shape[1]
        L = np.zeros((p - 2, p))
        for i in range(p-2):
            L[i, i] = 1
            L[i, i+2] = 1
            L[i, i+1] = -2
        x_new = np.concatenate([X, np.sqrt(n * self.lambda_2) * L], axis=0)
        y_new = np.concatenate([Y, np.zeros(p - 2)])
        l = np.ones((n+p-2,1))
        x_mean = x_new.mean(axis=0)
        x_var = x_new.var(axis=0)
        x = x_new-np.dot(l, x_mean.reshape(1, p))
        #y
        y_mean = y_new.mean()
        y = y_new - y_mean

        ######iteration
        beta = np.zeros(p)
        pre_beta = np.zeros(p)
        r = y.copy()   # residual
        pre_r = y.copy()
        for iteration in range(1, (1+self.max_iter)):
            for j in range(p):
                r_j = r + beta[j] * x[:, j]
                z = np.dot(x[:, j].T, r_j) / (n+p-2)
                #update
                beta[j] = _soft_thresholding(z, self.lambda_1) / x_var[j]
                r = r - (beta[j] - pre_beta[j]) * x[:, j]

            mse = (r[:n]**2).mean()
            pre_mse = (pre_r[:n]**2).mean()
            if (pre_mse-mse)/pre_mse < 0.0001:
                if not self.silence:
                    print("Nearly stop improve in iter %d" % iteration)
                    print("previous mse:%.4f , current mse:%.4f"%(pre_mse,mse))
                if pre_mse < mse:
                    beta = pre_beta.copy()
                break
            pre_r = r
            pre_beta = beta.copy()

        ######compute the original beta s and the intercept
        beta_ = beta.copy()
        if self.second_stage:
            beta_ = second_stage_thresholding(beta_, p)
        intercept = - (x_mean * beta_).sum() + y_mean
        self.coef_ = beta_
        self.intercept_ = intercept
        return self

    def predict(self, X: np.ndarray):
        y = np.dot(X, self.coef_)
        y += self.intercept_ * np.ones(len(y))
        return y


class spline_MCP():

    def __init__(self, lambda_1: float = 0.1, gamma_:float = 1.5,lambda_2:float = 0.1,
                 max_iter: int = 1000, second_stage: bool =False,silence: bool =True) -> None:
        self.lambda_1: float = lambda_1
        self.gamma_: float = gamma_     # gamma_ should be larger than 1
        self.lambda_2: float = lambda_2
        self.max_iter: int = max_iter
        self.second_stage: bool = second_stage
        self.coef_ = None
        self.intercept_ = None
        self.silence = silence

    def fit(self, X: np.ndarray, Y: np.ndarray):
        ######normalize data ( x&y ) -> zero mean
        #x
        n = X.shape[0]
        p = X.shape[1]
        L = np.zeros((p - 2, p))
        for i in range(p - 2):
            L[i, i] = 1
            L[i, i + 2] = 1
            L[i, i + 1] = -2
        x_new = np.concatenate([X, np.sqrt(n * self.lambda_2) * L], axis=0)
        y_new = np.concatenate([Y, np.zeros(p - 2)])
        l = np.ones((n + p - 2, 1))
        x_mean = x_new.mean(axis=0)
        x_var = x_new.var(axis=0)
        x = (x_new - np.dot(l, x_mean.reshape(1, p)))
        # y
        y_mean = y_new.mean()
        y = y_new - y_mean
        ######iteration
        beta = np.zeros(p)
        pre_beta = np.zeros(p)
        r = y.copy()   # residual
        pre_r = y.copy()
        for iteration in range(1, (1+self.max_iter)):
            for j in range(p):
                r_j = r + beta[j] * x[:, j]
                z = np.dot(x[:, j].T, r_j) / (n+p-2)
                #update
                if np.abs(z) <= self.lambda_1 * self.gamma_:
                    beta[j] = _soft_thresholding(z, self.lambda_1) / (x_var[j]*(1 - 1 / self.gamma_))
                else:
                    beta[j] = z / x_var[j]
                r = r - (beta[j] - pre_beta[j]) * x[:, j]
            mse = (r[:n]**2).mean()
            pre_mse = (pre_r[:n]**2).mean()
            if (pre_mse-mse)/pre_mse < 0.0001:
                if not self.silence:
                    print("Nearly stop improve in iter %d" % iteration)
                    print("previous mse:%.4f , current mse:%.4f"%(pre_mse,mse))
                if pre_mse < mse:
                    beta = pre_beta.copy()
                break
            pre_r = r
            pre_beta = beta.copy()

        ######compute the original beta s and the intercept
        beta_ = beta.copy()
        if self.second_stage:
            beta_ = second_stage_thresholding(beta_, p)
        intercept = - (x_mean * beta_).sum() + y_mean
        self.coef_ = beta_
        self.intercept_ = intercept
        return self

    def predict(self, X: np.ndarray):
        y = np.dot(X, self.coef_)
        y += self.intercept_ * np.ones(len(y))
        return y


class smooth_Lasso():

    def __init__(self, lambda_1: float = 0.1, lambda_2: float=0.1,
                 max_iter: int = 1000, second_stage: bool =True,silence: bool =True) -> None:
        self.lambda_1: float = lambda_1
        self.lambda_2: float = lambda_2
        self.max_iter: int = max_iter
        self.second_stage: bool = second_stage
        self.coef_ = None
        self.intercept_ = None
        self.silence = silence

    def fit(self, X: np.ndarray, Y: np.ndarray):
        ######normalize data ( x&y ) -> zero mean
        #x
        n = X.shape[0]
        p = X.shape[1]
        L = np.zeros((p - 1, p))
        for i in range(p-1):
            L[i, i] = 1
            L[i, i+1] = -1

        x_new = np.concatenate([X, np.sqrt(n * self.lambda_2) * L], axis=0)
        y_new = np.concatenate([Y, np.zeros(p - 1)])
        l = np.ones((n+p-1,1))
        x_mean = x_new.mean(axis=0)
        x_var = x_new.var(axis=0)
        x = (x_new-np.dot(l, x_mean.reshape(1, p)))
        #y
        y_mean = y_new.mean()
        y = y_new - y_mean

        ######iteration
        beta = np.zeros(p)
        pre_beta = np.zeros(p)
        r = y.copy()   # residual
        pre_r = y.copy()
        for iteration in range(1, (1+self.max_iter)):
            for j in range(p):
                r_j = r + beta[j] * x[:, j]
                z = np.dot(x[:, j].T, r_j) / (n+p-1)
                #update
                beta[j] = _soft_thresholding(z, self.lambda_1) / x_var[j]
                r = r - (beta[j] - pre_beta[j]) * x[:, j]

            mse = (r[:n]**2).mean()
            pre_mse = (pre_r[:n]**2).mean()
            if (iteration>10) and ((pre_mse-mse)/pre_mse < 0.0001):
                if not self.silence:
                    print("Nearly stop improve in iter %d" % iteration)
                    print("previous mse:%.4f , current mse:%.4f"%(pre_mse,mse))
                if pre_mse < mse:
                    beta = pre_beta.copy()
                break
            pre_r = r
            pre_beta = beta.copy()

        ######compute the original beta s and the intercept
        beta_ = beta.copy()
        if self.second_stage:
            beta_ = second_stage_thresholding(beta_, p)
        intercept = - (x_mean * beta_).sum() + y_mean
        self.coef_ = beta_
        self.intercept_ = intercept
        return self

    def predict(self, X: np.ndarray):
        y = np.dot(X, self.coef_)
        y += self.intercept_ * np.ones(len(y))
        return y


class fused_Lasso():

    def __init__(self, lambda_1: float = 0.1, lambda_2: float=0.1,
                 max_iter: int = 10, second_stage: bool =True,silence: bool =True) -> None:
        self.lambda_1: float = lambda_1
        self.lambda_2: float = lambda_2
        self.max_iter: int = max_iter
        self.second_stage: bool = second_stage
        self.coef_ = None
        self.intercept_ = None
        self.silence = silence

    def fit(self, X: np.ndarray, Y: np.ndarray):
        ######normalize data ( x&y ) -> zero mean
        #x
        n = X.shape[0]
        p = X.shape[1]
        l = np.ones((n,1))
        x_mean = X.mean(axis=0)
        x_var = X.var(axis=0)
        x = X-np.dot(l, x_mean.reshape(1, p))
        #y
        y_mean = Y.mean()
        y = Y - y_mean
        L = np.zeros((p - 1, p))
        for i in range(p - 1):
            L[i, i] = 1
            L[i, i + 1] = -1

        beta = np.zeros(p)
        def loss(beta_j, y, j,pre_beta,later_beta):
            r = y - beta_j*x[:,j]
            reg_l1 = self.lambda_1 * np.abs(beta_j)
            if 0<j<p-1  :
                reg_fused = self.lambda_2 * (np.abs(beta_j-pre_beta) + np.abs(beta_j-later_beta))
            elif j==0:
                reg_fused = self.lambda_2 * np.abs(beta_j - later_beta)
            elif j==p-1:
                reg_fused = self.lambda_2 * np.abs(beta_j - pre_beta)
            loss =  (r**2).mean() + reg_l1 + reg_fused
            return loss

        pre_beta = np.zeros(p)
        r = y.copy()  # residual
        pre_r = y.copy()
        for iteration in range(1, (1 + self.max_iter)):
            for j in range(p):
                r_j = r + beta[j] * x[:, j]
                # opt_result = minimize(loss,0.5,args=(r_j,j,beta[max(j-1,0)],beta[min(j+1,p-1)]))
                opt_result = minimize_scalar(loss, args=(r_j, j, beta[max(j - 1, 0)], beta[min(j + 1, p - 1)]))
                beta[j] = opt_result.x
                r = r - (beta[j] - pre_beta[j]) * x[:, j]
            mse = (r[:n] ** 2).mean()
            pre_mse = (pre_r[:n] ** 2).mean()
            if  ((pre_mse - mse) / pre_mse < 0.01):
                if not self.silence:
                    print("Nearly stop improve in iter %d" % iteration)
                    print("previous mse:%.4f , current mse:%.4f" % (pre_mse, mse))
                if pre_mse < mse:
                    beta = pre_beta.copy()
                break
            pre_r = r
            pre_beta = beta.copy()
        ######compute the original beta s and the intercept
        beta_ = beta.copy()
        if self.second_stage:
            beta_ = second_stage_thresholding(beta_, p)
        intercept = - (x_mean * beta_).sum() + y_mean
        self.coef_ = beta_
        self.intercept_ = intercept
        return self


    def predict(self, X: np.ndarray):
        y = np.dot(X, self.coef_)
        y += self.intercept_ * np.ones(len(y))
        return y


if __name__ == "__main__":
    #some test

    ## A simple test of the Lasso compared to sklearn.linear_model.lasso
    from data_genertor import simulation_data
    from sklearn import linear_model
    import matplotlib.pyplot as plt

    # generate data
    data = simulation_data(p=600, n=71)
    mu, cov = data.gen_mu_cov()
    data.gen_beta()
    data.gen_X(mu, cov)
    data.gen_Y(std=1, form="norm")

    ############################
    reg0 = linear_model.Lasso(alpha=0.05)
    #lasso
    reg1 = Lasso(lambda_=0.01, max_iter=50, second_stage=False)
    #mcp
    reg2 = MCP(lambda_=0.01,gamma_=1.5,second_stage=False)
    #spline-lasoo
    reg3 = spline_Lasso(lambda_1=0.01,lambda_2=0.01,second_stage=False)
    #spline-MCP
    reg4 = spline_MCP(lambda_1=0.05,gamma_=2,lambda_2=0.3,second_stage=True)
    #smooth-lasso
    reg5 = smooth_Lasso(lambda_1=0.01,lambda_2=0.2,second_stage=True)
    #fused-lass
    reg6 = fused_Lasso(lambda_1=0.01,lambda_2=0.5,second_stage=True)
    #test
    # reg0.fit(data.X,data.Y)
    # reg1.fit(data.X,data.Y)
    # reg2.fit(data.X,data.Y)
    # reg3.fit(data.X, data.Y)
    reg4.fit(data.X, data.Y)
    # reg5.fit(data.X, data.Y)
    # reg6.fit(data.X, data.Y)
    plt.plot(data.beta,"bo")
    # plt.plot(reg0.coef_,"g")
    plt.plot(reg4.coef_,"r*")
    plt.show()

