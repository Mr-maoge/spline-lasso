from data_genertor import simulation_data
from models import spline_Lasso,spline_MCP,smooth_Lasso,fused_Lasso
from utiles import mse, specificity, sensitivity, GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
np.random.seed(20191005)
#####################tuning parameters
params_grid_spline_Lasso = {"lambda_1":[0.01,0.02],
                            "lambda_2":[0.2,0.3],
                            }

params_grid_spline_MCP = {"lambda_1":[0.01,0.02],
                          "gamma_":[2],
                          "lambda_2":[0.2,0.3],
                         }

params_grid_smooth_Lasso = {"lambda_1":[0.01,0.02],
                            "lambda_2":[0.2,0.4],
                            }

params_grid_fused_Lasso = {"lambda_1":[0.01],
                            "lambda_2":[0.2,0.3],
                           }

#####################some preparation
models_class = [spline_Lasso,spline_MCP,smooth_Lasso,fused_Lasso]
models_name = ["spline_Lasso","spline_MCP","smooth_Lasso","fused_Lasso"]
model_parms_grid = [params_grid_smooth_Lasso, params_grid_spline_MCP,
                    params_grid_smooth_Lasso, params_grid_fused_Lasso]
result_models_name = []
for name in models_name:
    result_models_name.append(name)
    result_models_name.append(name+"/Thresh")

#####################do the simulation
def simulate(noise_level="low",form="norm"):
    repeat_times = 20
    if noise_level == "low":
        noise_std = 1
    else:
        noise_std = 3

    # restore the simulation result (mse,beta_mse,beta_specificity,beta_sensitivity)
    result_table = pd.DataFrame(columns=["case","model","testing_mse","beta_specificity","beta_sensitivity"])
    result_table["case"] = [1]*(2*len(models_class)) + [2]*(2*len(models_class))
    result_table["model"] = result_models_name + result_models_name
    # restore the "best" estimators of each different kinds of models
    info_set = [[],[]]
    data_set = []
    for case in [1, 2]:
        print("*"*10+"case1"+"*"*10)
        testing_mse = np.zeros(len(models_class)*2)
        beta_mse = np.zeros(len(models_class)*2)
        beta_specificity = np.zeros(len(models_class)*2)
        beta_sensitivity = np.zeros(len(models_class)*2)
        for t in tqdm(range(repeat_times)):
            #generate data
            data = simulation_data(p=600, n=71)
            mu, cov = data.gen_mu_cov(case=case)
            data.gen_beta()
            data.gen_X(mu, cov)
            data.gen_Y(std=noise_std, form=form)
            if t == repeat_times-1:
                data_set.append(data)

            for i in range(len(models_class)):
                model_class = models_class[i]
                params_grid = model_parms_grid[i]
                params_grid["second_stage"] = False
                best_info1, _ = GridSearchCV(model_class, params_grid, data.X, data.Y, 10, mse)
                testing_mse1 = best_info1["loss"]["train_loss"]
                beta_mse1 = mse(data.beta,best_info1["estimator"].coef_)
                beta_specificity1 = specificity(data.beta,best_info1["estimator"].coef_)
                beta_sensitivity1 = sensitivity(data.beta,best_info1["estimator"].coef_)

                params_grid["second_stage"] = True
                best_info2, _ = GridSearchCV(model_class, params_grid, data.X, data.Y, 10, mse)
                testing_mse2 = best_info2["loss"]["train_loss"]
                beta_mse2 = mse(data.beta, best_info2["estimator"].coef_)
                beta_specificity2 = specificity(data.beta, best_info2["estimator"].coef_)
                beta_sensitivity2 = sensitivity(data.beta, best_info2["estimator"].coef_)

                #save
                testing_mse[i*2] += testing_mse1 / repeat_times
                beta_mse[i*2] += beta_mse1 / repeat_times
                beta_specificity[i*2] += beta_specificity1 / repeat_times
                beta_sensitivity[i*2] += beta_sensitivity1 / repeat_times
                if t == repeat_times-1:
                    info_set[case-1].append(best_info1)

                testing_mse[i*2+1] += testing_mse2 / repeat_times
                beta_mse[i*2+1] += beta_mse2 / repeat_times
                beta_specificity[i*2+1] += beta_specificity2 / repeat_times
                beta_sensitivity[i*2+1] += beta_sensitivity2 / repeat_times
                if t == repeat_times - 1:
                    info_set[case - 1].append(best_info2)

        result_table.loc[result_table["case"]==case,"beta_mse"] = beta_mse
        result_table.loc[result_table["case"] == case, "testing_mse"] = testing_mse
        result_table.loc[result_table["case"] == case, "beta_specificity"] = beta_specificity
        result_table.loc[result_table["case"] == case, "beta_sensitivity"] = beta_sensitivity

    return result_table, info_set, data_set


for noise_level,form in [("low","norm"),("high","norm"),("low","t")]:
    print("="*10+"noise_level: {} , form: {}".format(noise_level,form)+"="*10)
    result_table, info_set, data_set = simulate(noise_level,form)
    result_table.to_csv("simulate_result/result_table_{}_{}.csv".format(noise_level,form))
    pickle.dump(info_set,open("simulate_result/info_set_{}_{}.pkl".format(noise_level,form),"wb"))
    pickle.dump(data_set, open("simulate_result/data_set_{}_{}.pkl".format(noise_level,form), "wb"))

#####################some analysis (plot)
info_set = pickle.load(open("simulate_result/info_set_low_norm.pkl","rb"))
data_set = pickle.load(open("simulate_result/data_set_low_norm.pkl","rb"))

for case in [1,2]:
    data = data_set[case - 1]
    info = info_set[case-1]
    for i in range(len(result_models_name)):
        model = info[i]
        model_name = result_models_name[i]
        plt.figure(figsize=(16,8))
        plt.plot(data.beta,"bo")
        plt.plot(model["estimator"].coef_,"g*")
        plt.legend(["True Beta","Fitted Beta"],fontsize=15)
        plt.xlabel("location",fontsize=15)
        plt.ylabel("beta",fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title("%s Estimation"%model_name,fontsize=15)
        plt.savefig("simulate_result/beta_plots/case%d_%s.png"%(case,model_name.replace("/","_")))
        plt.show()
        plt.close()


#####################output some table used to show in the report
for noise_level, form in [("low", "norm"), ("high", "norm"), ("low", "t")]:
    tmp1 = pd.read_csv("simulate_result/result_table_{}_{}.csv".format(noise_level,form))
    order = ["fused_Lasso","spline_MCP","spline_Lasso","smooth_Lasso","fused_Lasso/Thresh",
             "spline_Lasso/Thresh","smooth_Lasso/Thresh"]
    case1 = tmp1[tmp1.case==1][['model', 'testing_mse','beta_specificity',
                                'beta_sensitivity', 'beta_mse']].set_index("model",drop=True)
    case1["beta_mse"] = np.sqrt(case1["beta_mse"] * 600)
    case1 = case1.T
    case1 = case1.loc[:, order]

    case2 = tmp1[tmp1.case == 2][['model', 'testing_mse', 'beta_specificity',
                                  'beta_sensitivity', 'beta_mse']].set_index("model", drop=True)
    case2["beta_mse"] = np.sqrt(case2["beta_mse"] * 600)
    case2 = case2.T
    case2 = case2.loc[:,order]

    case1.to_csv("{}_{}_case1.csv".format(noise_level,form))
    case2.to_csv("{}_{}_case2.csv".format(noise_level,form))
