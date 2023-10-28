import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame, concat
import seaborn as sns


def sig_controller(alpha, beta, gama):
    list_alpha = []
    list_beta = []
    list_gama = []
    list_avgAcc = []
    list_avgPre = []
    list_avgRec = []
    list_avgF1 = []

    for a in range(len(alpha)):
        for b in range(len(beta)):
            for g in range(len(gama)):
                path = "../../collusion_P_0.8/KCDN_W(废)/a_b_r/a" + str(alpha[a]) + "_b" + str(beta[b]) + "_r" + str(
                    gama[g]) + "_KCDN_Percentage_36_0.csv"

                Accuracy = list(pd.read_csv(path)["Acc"])
                Precision = list(pd.read_csv(path)["Pre"])
                Recall = list(pd.read_csv(path)["Rec"])
                F1 = list(pd.read_csv(path)["f1_score"])

                list_alpha.append(alpha[a] / alpha[a])
                list_beta.append(beta[b] / alpha[a])
                list_gama.append(gama[g] / alpha[a])
                list_avgAcc.append(sum(Accuracy) / 30)
                list_avgPre.append(sum(Precision) / 30)
                list_avgRec.append(sum(Recall) / 30)
                list_avgF1.append(sum(F1) / 30)

    pd_alpha = DataFrame({"alpha": list_alpha})
    pd_beta = DataFrame({"beta": list_beta})
    pd_gama = DataFrame({"gama": list_gama})

    avgAcc = DataFrame({"avgAcc": list_avgAcc})
    avgPre = DataFrame({"avgPre": list_avgPre})
    avgRec = DataFrame({"avgRec": list_avgRec})
    avgF1 = DataFrame({"avgF1": list_avgF1})

    write = concat([pd_alpha, pd_beta, pd_gama, avgAcc, avgPre, avgRec, avgF1], axis=1)
    write = write.drop_duplicates()
    write.to_csv("data_drop_duplicates_1.csv", float_format="%.5f", sep=',', index=0)





def pltData(target):
    data = pd.read_csv("data_drop_duplicates.csv")

    beta = list(data["beta"])

    gama = list(data["gama"])

    target_data = myTransformers(beta, gama, data, target)

    print(target_data)

    x_axis = ["0.03125", "0.0625", "0.125", "0.25", "0.5", "1", "2", "4", "8", "16", "32"]
    y_axis = ["32", "16", "8", "4", "2", "1", "0.5", "0.25", "0.125", "0.0625", "0.03125"]

    max_number = 0
    for i in range(len(target_data)):
        for j in range(len(target_data[i])):
            if target_data[i][j] > max_number:
                max_number = target_data[i][j]
    print(max_number)

    min_number = 1
    for i in range(len(target_data)):
        for j in range(len(target_data[i])):
            if target_data[i][j] < min_number:
                min_number = target_data[i][j]
    print(min_number)

    plt.figure(figsize=(14, 8))

    sns.heatmap(target_data,
                annot=True,
                fmt="0.4f",
                vmin=min_number,
                vmax=max_number,
                cmap='Blues',
                yticklabels=y_axis,
                xticklabels=x_axis,
                linewidths=0.2)

    if target == "Acc":
        plt.title(r'Accuracy under different $\beta$ and $\gamma$ ($\alpha = 1$)', size=26)
        plt.xlabel(r'$\gamma$', labelpad=20, size=26)
        plt.ylabel(r'$\beta$', labelpad=20, size=26)
        path = "hyperparameter_Accuracy.jpg"
        plt.gcf().savefig(path, dpi=1200, format='jpg')

    elif target == "Pre":
        plt.title(r'Precision under different $\beta$ and $\gamma$ ($\alpha = 1$)', size=26)
        plt.xlabel(r'$\gamma$', labelpad=20, size=26)
        plt.ylabel(r'$\beta$', labelpad=20, size=26)
        path = "hyperparameter_Precision.jpg"
        plt.gcf().savefig(path, dpi=1200, format='jpg')

    elif target == "Rec":
        plt.title(r'Recall under different $\beta$ and $\gamma$ ($\alpha = 1$)', size=26)
        plt.xlabel(r'$\gamma$', labelpad=20, size=26)
        plt.ylabel(r'$\beta$', labelpad=20, size=26)
        path = "hyperparameter_Pecall.jpg"
        plt.gcf().savefig(path, dpi=1200, format='jpg')

    elif target == "F1":
        plt.title(r'F1-score under different $\beta$ and $\gamma$ ($\alpha = 1$)', size=26)
        plt.xlabel(r'$\gamma$', labelpad=20, size=26)
        plt.ylabel(r'$\beta$', labelpad=20, size=26)
        path = "hyperparameter_F1.jpg"
        plt.gcf().savefig(path, dpi=1200, format='jpg')


    plt.show()


def myTransformers(beta, gama, data, str):
    index = 0

    if str == "Acc":
        index = 3
    elif str == "Pre":
        index = 4
    elif str == "Rec":
        index = 5
    elif str == "F1":
        index = 6

    beta = list(set(beta))
    beta.sort()
    gama = list(set(gama))
    gama.sort()

    W = np.zeros((len(beta), len(gama)))

    for i in range(len(data)):
        for b in range(len(beta)):
            for g in range(len(gama)):
                if data.iloc[i][1] == beta[b] and data.iloc[i][2] == gama[g]:
                    W[b][g] = data.iloc[i][index]

    # 上下反转
    W_tran = np.flipud(W)

    # np.savetxt(str + "_W.csv", W_tran, delimiter=",", fmt='%.05f')

    return W_tran





if __name__ == '__main__':
    # 统计数据【alpha, beta,game, avgAcc, avgPre, avgRec, avgF1】
    # 【1, beta/alpha, game/alpha, avgAcc, avgPre, avgRec, avgF1】

    alpha = [1]
    # gama = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
    # beta = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
    # sig_controller(alpha, beta, gama)

    # 画图
    pltData("Acc")
    pltData("Pre")
    pltData("Rec")
    pltData("F1")
