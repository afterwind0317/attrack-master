import pandas as pd


def mean(str, path):

    data = pd.read_csv(path, header=None)

    for i in range(len(data)):
        tmp_data = []
        for j in range(len(data.iloc[i])):
            tmp_data.append(data.iloc[i][j])

        tmp_avg = round(sum(tmp_data) / 30, 4)

    print(str, tmp_avg)






if __name__ == '__main__':

    path = "HDS_LACF_acc_collusion_0.8-32.csv"
    path_CP = "CP_HDS_acc_collusion_0.8-32.csv"
    path_FC = "FC_HDS_acc_collusion_0.8-32.csv"
    path_KCD = "KCD_HDS_acc_collusion_0.8-32.csv"
    path_KCDN = "KCDN_HDS_acc_collusion_0.8-32.csv"

    mean("原串谋数据集：", path)
    mean("CP：", path_CP)
    mean("FC：", path_FC)
    mean("KCD：", path_KCD)
    mean("KCDN_W：", path_KCDN)