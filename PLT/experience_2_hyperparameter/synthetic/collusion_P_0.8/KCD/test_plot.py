import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pandas import DataFrame, concat


def Mean_Var(path):

    index = [12,24,36,48,60]

    Algorithm = path

    Mean_acc = []
    Var_acc = []

    Mean_pre = []
    Var_pre = []

    Mean_rec = []
    Var_rec = []

    Mean_f1 = []
    Var_f1 = []

    for i in range(len(index)):

        path = "../../collusion_P_0.8/KCD(废)/"+str(Algorithm)+"/KCD_Percentage_"+str(index[i])+"_0.csv"

        Accuracy = list(pd.read_csv(path)["Acc"])
        Precision = list(pd.read_csv(path)["Pre"])
        Recall = list(pd.read_csv(path)["Rec"])
        F1 = list(pd.read_csv(path)["f1_score"])

        Mean_acc.append(sum(Accuracy)/30)
        Var_acc.append(np.std(Accuracy))


        Mean_pre.append(sum(Precision)/30)
        Var_pre.append(np.std(Precision))


        Mean_rec.append(sum(Recall)/30)
        Var_rec.append(np.std(Recall))

        Mean_f1.append(sum(F1)/30)
        Var_f1.append(np.std(F1))


    Mean_acc = DataFrame({"Mean_acc": Mean_acc})
    Var_acc = DataFrame({"Var_acc": Var_acc})
    Mean_pre = DataFrame({"Mean_pre": Mean_pre})
    Var_pre = DataFrame({"Var_pre": Var_pre})
    Mean_rec = DataFrame({"Mean_rec": Mean_rec})
    Var_rec = DataFrame({"Var_rec": Var_rec})
    Mean_f1 = DataFrame({"Mean_f1": Mean_f1})
    Var_f1 = DataFrame({"Var_f1": Var_f1})



    write = concat([Mean_acc, Var_acc, Mean_pre, Var_pre, Mean_rec, Var_rec, Mean_f1, Var_f1], axis=1)
    print(write)

    save_path = "../../collusion_P_0.8/KCD(废)/"+str(Algorithm)+"/Mean_Var.csv"
    write.to_csv(save_path)





def Plt(xx):

    path = "KCD_1_0.5/Mean_Var.csv"
    KCD_mean_1 = list(pd.read_csv(path)["Mean_"+str(xx)])

    path = "KCD_1_1/Mean_Var.csv"
    KCD_mean_2 = list(pd.read_csv(path)["Mean_"+str(xx)])

    path = "KCD_1_2/Mean_Var.csv"
    KCD_mean_3 = list(pd.read_csv(path)["Mean_"+str(xx)])

    path = "KCD_1_4/Mean_Var.csv"
    KCD_mean_4 = list(pd.read_csv(path)["Mean_"+str(xx)])

    path = "KCD_1_8/Mean_Var.csv"
    KCD_mean_5 = list(pd.read_csv(path)["Mean_"+str(xx)])




    x_ticks = ['0.1', '0.2', '0.3', '0.4', '0.5']
    x = range(len(x_ticks))

    plt.plot(x, KCD_mean_1,
                 linewidth=1.5, marker='',
                 ms=7,
                 label=u'KCD_1_0.5')

    plt.plot(x, KCD_mean_2,
                 linewidth=1.5, marker='',
                 ms=7,
                 label=u'KCD_1_1')

    plt.plot(x, KCD_mean_3,
                 linewidth=1.5, marker='',
                 ms=7,
                 label=u'KCD_1_2')
    plt.plot(x, KCD_mean_4,
                 linewidth=1.5, marker='',
                 ms=7,
                 label=u'KCD_1_4')

    plt.plot(x, KCD_mean_5,
                 linewidth=1.5, marker='',
                 ms=7,
                 label=u'KCD_1_8')





    plt.legend()  # 让图例生效
    plt.xticks(x, x_ticks, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y')

    plt.xlim(-0.2, 4.2)
    plt.ylim(0, 1)

    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"Collusion Percentage", size=18)  # X轴标签
    plt.ylabel("KCD_"+xx, size=18)  # Y轴标签

    plt.tick_params(direction='in')
    plt.tick_params(top=True, bottom=True, left=True, right=True)


    path = 'experience2_synthetic_Acc.jpg'
    plt.gcf().savefig(path, dpi=1800, format='jpg')


    plt.show()






if __name__ == '__main__':


    """求平均值与标准型差"""
    # Mean_Var("KCD_1_1")
    # Mean_Var("KCD_1_0.5")
    # Mean_Var("KCD_1_2")
    # Mean_Var("KCD_1_4")
    # Mean_Var("KCD_1_8")


    """画图"""
    Plt("acc")
    Plt("rec")
    Plt("f1")
    Plt("pre")
