import numpy as np
from numpy.random import multivariate_normal,standard_normal,standard_t


class simulation_data():
    """
        The class used to generate the simulation data for the tests
    """
    def __init__(self,p=600,n=71):
        self.p = p          # dim
        self.n = n          # num of samples
        self.X = None
        self.Y = None
        self.intercept = 0
        self.beta = np.zeros(self.p)
        self.noise = None

    def gen_beta(self,num_part1=54, num_part2=25,intercept=None):
        """
            Used to generate the beta s for stimulation tests
            num_part1 + num_part2 = the number of beta s to be non-zero

        The related sentences in the paper goes like this:
            β’s are obtained from a continuous function consisting of two parts: one part
            being a combination of sine and constant function,and the other being a
            combination of two linear functions.
        Note:
            In fact I'm not quit sure that I understand the meaning of paper correctly. And I
            just develop the program according to my understanding and the figures shown in
            the essay.
        """
        #intercept
        if intercept is not None:
            self.intercept = intercept
        #part1
        num_p1_1 = num_part1 // 8
        num_p2_2 = num_part1 - num_p1_1
        sep_point = int(num_p2_2 * 0.8)
        tmp1 = np.sin([np.pi/(num_p2_2-6) * i for i in range(1,sep_point)]) * 3
        tmp2 = np.ones(num_p1_1) * tmp1[-1]
        tmp3 = np.sin([np.pi/(num_p2_2-6) * i for i in range(sep_point,num_p2_2-3)]) * 3
        tmp4 = np.linspace(tmp3[-1]+0.1,0,4,endpoint=False)
        part1 = np.concatenate([tmp1,tmp2,tmp3,tmp4])

        #part2
        num_p2_1 = num_part2 // 2 + 1
        num_p2_2 = num_part2 - num_p2_1
        slop = 5.5 / num_p2_1
        tmp1 = np.array([slop*i for i in range(num_p2_1)]) - 0.2
        tmp2 = np.array([5-slop*i for i in range(1,num_p2_2+1)]) - 0.2
        part2 = np.concatenate([tmp1,tmp2])

        self.beta[1:(len(part1)+1)] = part1
        self.beta[(len(part1)+30):(len(part1)+30+len(part2))] = part2

        return self.beta

    def set_beta(self, beta):
        self.beta = beta

    def gen_mu_cov(self,case=1):
        assert case in [1,2],"ERROR"
        mu_ = np.zeros(self.p)
        if case==1:
            cov_ = np.eye(self.p)
        elif case==2:
            cov_ = np.eye(self.p)
            for i in range(self.p-1):
                for j in range(i+1,self.p):
                    v = 0.5**(j-i)
                    cov_[i,j] = v
                    cov_[j,i] = v
        return mu_,cov_

    def gen_X(self,mu=None,cov=None):
        if mu is None:
            mu_ = np.zeros(self.p)  # default: zero mean
        else:
            mu_ = np.array(mu)
        if cov is None:
            cov_ = np.eye(self.p)    #default: I
        else:
            cov_ = np.array(cov)
        self.X = multivariate_normal(mu_,cov_, self.n)
        return self.X

    def gen_Y(self,std=1,form="norm",df=5):
        if form == "norm":
            e = standard_normal(self.n)
        elif form == "t":
            e = standard_t(df,self.n) * np.sqrt((df-2)/df)
        self.noise = e * std
        self.Y = self.intercept + np.dot(self.X, self.beta) + self.noise
        return self.Y


if __name__ == "__main__":
    ## some test
    import matplotlib.pyplot as plt
    data = simulation_data(p=600,n=71)
    beta = data.gen_beta(58,21)
    print((beta>0).sum())
    print((beta<0).sum())
    plt.figure(figsize=(16, 8))
    plt.plot(beta, "o")
    plt.xlabel("location", fontsize=15)
    plt.ylabel("beta", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig("test.png")
    plt.show()
    mu,cov = data.gen_mu_cov(case=2)
    data.gen_X(mu,cov)
    data.gen_Y(std=3,form="t")

