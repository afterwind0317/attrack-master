import csv
import math
import random
import sys

import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 设定要计算的数据集,串谋之后的数据集
dataname = "_collaborate_D.csv"


# /////////////////////////////////

# /////////////////////////////////
def get_MV_acc(data, gt):
    """
    找出每一首歌的注释，然后取投票结果
    进行对比，如果与gt一致，那么分子+1
    返回fz/fm
    """
    # 去重任务(task = 1-807)
    task = data['question'].drop_duplicates().sort_values()
    task = DataFrame({'task': task})
    # print(task)
    fz = 0  # r=Rg个数

    for i in range(len(task)):  # 循环每一个任务
        # print("当前任务：", task.iloc[i][0])

        annotation = []
        R = 0  # 真值
        for j in range(len(data)):
            if data.iloc[j][4] == task.iloc[i][0]:
                R = data.iloc[j][7]  # 提取真值
                annotation.append(data.iloc[j][6])
        # print(annotation)
        # r = top1(annotation)  # 取众数 投票结果
        r = max(annotation, default='列表为空', key=lambda v: annotation.count(v))
        # print("mv:", r)
        # print("gt:", R)
        if int(r) == int(R):
            fz = fz + 1
        # print(fz, "/", len(gt))
        acc = fz / len(gt)
    return acc

def MV(collusion_P, list, gt_data, Alg):

    W = np.zeros((len(list), 30))
    gt = gt_data

    for p in range(len(collusion_P)):

        for i in range(len(list)):

            for epo in range(0,30):

                print(list[i], epo, "/", 30)
                datafile = "../../../experience/result/collusion_data/collusion_" + str(
                    collusion_P[p]) + "/experience_" + str(list[i]) + "/data" + str(epo) + "_collaborate_D_"+Alg+".csv"

                data = pd.read_csv(datafile)
                print("collusion_P=", collusion_P, " percentage=", list[i], "epoch=", epo)
                tmp_acc = get_MV_acc(data, gt)
                print("MV-acc=", tmp_acc)

                W[i][epo] = tmp_acc

            save_path = "../../../experience/acc_result_del/"+Alg+"_MV_acc_collusion_"+str(collusion_P[p])+"-"+str(list[i])+".csv"
            np.savetxt(save_path, W, delimiter=",", fmt='%.8f')


#==============================================================
class get_acc_HDS:
    def __init__(self, datafile, truth_file, **kwargs):
        self.datafile = datafile
        self.truth_file = truth_file
        e2wl, w2el, label_set = self.gete2wlandw2el()
        self.e2wl = e2wl  # {t0:[w0:l0], t1:[w1:l1], ...}
        self.w2el = w2el  # {w0:[t0:l0], w1:[t1:l1], ...}
        self.workers = self.w2el.keys()
        self.label_set = label_set
        self.initalquality = 0.7
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

    # E-step
    def Update_e2lpd(self):
        self.e2lpd = {}
        for example, worker_label_set in self.e2wl.items():
            lpd = {}
            total_weight = 0
            for tlabel, prob in self.l2pd.items():
                weight = prob
                for (w, label) in worker_label_set:
                    weight *= self.w2cm[w][tlabel][label]

                lpd[tlabel] = weight
                total_weight += weight
            for tlabel in lpd:
                if total_weight == 0:
                    # uniform distribution
                    lpd[tlabel] = 1.0 / len(self.label_set)
                else:
                    lpd[tlabel] = lpd[tlabel] * 1.0 / total_weight

            self.e2lpd[example] = lpd

        # print(self.e2lpd)  # 推断

    # M-step

    def Update_l2pd(self):
        for label in self.l2pd:
            self.l2pd[label] = 0
        for _, lpd in self.e2lpd.items():
            for label in lpd:
                self.l2pd[label] += lpd[label]

        for label in self.l2pd:
            self.l2pd[label] *= 1.0 / len(self.e2lpd)
        # print(self.l2pd)  # 更新先验

    def Update_w2cm(self):

        for w in self.workers:
            for tlabel in self.label_set:
                for label in self.label_set:
                    self.w2cm[w][tlabel][label] = 0

        w2lweights = {}
        for w in self.w2el:
            w2lweights[w] = {}
            for label in self.label_set:
                w2lweights[w][label] = 0
            for example, _ in self.w2el[w]:
                for label in self.label_set:
                    w2lweights[w][label] += self.e2lpd[example][label]

            for tlabel in self.label_set:

                if w2lweights[w][tlabel] == 0:
                    for label in self.label_set:
                        if tlabel == label:
                            self.w2cm[w][tlabel][label] = self.initalquality
                        else:
                            self.w2cm[w][tlabel][label] = (1 - self.initalquality) * 1.0 / (len(self.label_set) - 1)

                    continue

                for example, label in self.w2el[w]:
                    self.w2cm[w][tlabel][label] += self.e2lpd[example][tlabel] * 1.0 / w2lweights[w][tlabel]

        for w in self.workers:
            for tlabel in self.label_set:
                a = 0
                for label in self.label_set:
                    if tlabel == label:
                        a = (1 - self.w2cm[w][tlabel][label]) * 1.0 / (len(self.label_set) - 1)
                for label in self.label_set:
                    if tlabel != label:
                        self.w2cm[w][tlabel][label] = a

        return self.w2cm

    # initialization
    def Init_l2pd(self):
        # uniform probability distribution
        l2pd = {}
        for label in self.label_set:
            l2pd[label] = 1.0 / len(self.label_set)
        return l2pd

    def Init_w2cm(self):
        w2cm = {}
        for worker in self.workers:
            w2cm[worker] = {}
            for tlabel in self.label_set:
                w2cm[worker][tlabel] = {}
                for label in self.label_set:
                    if tlabel == label:
                        w2cm[worker][tlabel][label] = self.initalquality
                    else:
                        w2cm[worker][tlabel][label] = (1 - self.initalquality) / (len(self.label_set) - 1)

        return w2cm

    def run(self, iter=50):

        self.l2pd = self.Init_l2pd()  # {l0:p0, l1:p1, l2:p2, l3:p3}
        self.w2cm = self.Init_w2cm()
        # {'78': {'1': {'1': 0.7, '3': 0.1, '4': 0.1, '2': 0.1},
        #         '3': {'1': 0.1, '3': 0.7, '4': 0.1, '2': 0.1},
        #         '4': {'1': 0.1, '3': 0.1, '4': 0.7, '2': 0.1},
        #         '2': {'1': 0.1, '3': 0.1, '4': 0.1, '2': 0.7}
        #         }
        #  '2': {'1': {'1': 0.7, '3': 0.1, '4': 0.1, '2': 0.1},
        #        '3': {'1': 0.1, '3': 0.7, '4': 0.1, '2': 0.1},
        #        '4': {'1': 0.1, '3': 0.1, '4': 0.7, '2': 0.1},
        #        '2': {'1': 0.1, '3': 0.1, '4': 0.1, '2': 0.7}
        #        }
        #  }

        while iter > 0:
            # print(iter)
            # E-step
            self.Update_e2lpd()
            # M-step
            self.Update_l2pd()
            self.Update_w2cm()
            # compute the likelihood
            # print self.computelikelihood()
            iter -= 1

        return self.e2lpd, self.w2cm

    def computelikelihood(self):

        lh = 0

        for _, worker_label_set in self.e2wl.items():
            temp = 0
            for tlabel, prior in self.l2pd.items():
                inner = prior
                for worker, label in worker_label_set:
                    inner *= self.w2cm[worker][tlabel][label]
                temp += inner

            lh += math.log(temp)

        return lh

    ###################################
    # The above is the EM method (a class)
    # The following are several external functions
    ###################################

    def get_accuracy(self):
        e2truth = {}
        f = open(self.truth_file, 'r')
        reader = csv.reader(f)
        next(reader)

        for line in reader:
            # example, truth = line[1:3]
            example = line[0]
            truth = line[1]

            e2truth[example] = truth

        tcount = 0
        count = 0
        e2lpd = self.e2lpd
        for e in e2lpd:
            if e not in e2truth:
                continue
            temp = 0
            for label in e2lpd[e]:
                if temp < e2lpd[e][label]:
                    temp = e2lpd[e][label]
            candidate = []
            for label in e2lpd[e]:
                if temp == e2lpd[e][label]:
                    candidate.append(label)
            truth = random.choice(candidate)
            count += 1
            if truth == e2truth[e]:
                tcount += 1

        return tcount * 1.0 / count

    def gete2wlandw2el(self):
        e2wl = {}
        w2el = {}
        label_set = []

        f = open(self.datafile, 'r')
        reader = csv.reader(f)
        next(reader)

        for line in reader:
            # example, worker, label = line[1:4]
            example, worker = line[4:6]
            label = line[6]
            if example not in e2wl:
                e2wl[example] = []
            e2wl[example].append([worker, label])

            if worker not in w2el:
                w2el[worker] = []
            w2el[worker].append([example, label])

            if label not in label_set:
                label_set.append(label)
        return e2wl, w2el, label_set

def HDS(collusion_P, list, truth_file, Alg):
    W = np.zeros((len(list), 30))

    # 串谋机会
    for p in range(len(collusion_P)):
        # 串谋占比
        for i in range(len(list)):
            # 30次
            for epo in range(30):
                print(list[i], epo, "/", 30)
                datafile = "../../../experience/result/collusion_data/collusion_" + str(
                    collusion_P[p]) + "/experience_" + str(list[i]) + "/data" + str(epo) + "_collaborate_D_"+Alg+".csv"

                truth_file = truth_file

                model = get_acc_HDS(datafile, truth_file=truth_file)
                model.run()
                accuracy = model.get_accuracy()
                print("HDS 准确率：%f" % (accuracy))
                print(i, " ", epo)
                #
                W[i][epo] = accuracy
                #
            save_path = "../../../experience/acc_result_del/"+Alg+"_HDS_acc_collusion_"+str(collusion_P[p])+"-"+str(list[i])+".csv"
            np.savetxt(save_path, W, delimiter=",", fmt='%.8f')



#================================================================
class get_GLAD_acc:
    def __init__(self, datafile, **kwargs):
        e2wl, w2el, label_set = self.gete2wlandw2el(datafile)
        self.e2wl = e2wl
        self.w2el = w2el
        self.workers = self.w2el.keys()
        self.examples = self.e2wl.keys()
        self.label_set = label_set
        self.datafile = datafile
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

    def sigmoid(self, x):
        if (-x) > math.log(sys.float_info.max):
            return 0
        if (-x) < math.log(sys.float_info.min):
            return 1

        return 1 / (1 + math.exp(-x))

    def logsigmoid(self, x):
        # For large negative x, -log(1 + exp(-x)) = x
        if (-x) > math.log(sys.float_info.max):
            return x
        # For large positive x, -log(1 + exp(-x)) = 0
        if (-x) < math.log(sys.float_info.min):
            return 0

        value = -math.log(1 + math.exp(-x))
        # if (math.isinf(value)):
        #    return x

        return value

    def logoneminussigmoid(self, x):
        # For large positive x, -log(1 + exp(x)) = -x
        if (x) > math.log(sys.float_info.max):
            return -x
        # For large negative x, -log(1 + exp(x)) = 0
        if (x) < math.log(sys.float_info.min):
            return 0

        value = -math.log(1 + math.exp(x))
        # if (math.isinf(value)):
        #    return -x

        return value

    def kronecker_delta(self, answer, label):
        if answer == label:
            return 1
        else:
            return 0

    def expbeta(self, beta):
        if beta >= math.log(sys.float_info.max):
            return sys.float_info.max
        else:
            return math.exp(beta)

    # E step
    def Update_e2lpd(self):
        self.e2lpd = {}
        for example, worker_label_set in self.e2wl.items():
            lpd = {}
            total_weight = 0

            for tlabel, prob in self.prior.items():
                weight = math.log(prob)
                for (worker, label) in worker_label_set:
                    logsigma = self.logsigmoid(self.alpha[worker] * self.expbeta(self.beta[example]))
                    logoneminussigma = self.logoneminussigmoid(self.alpha[worker] * self.expbeta(self.beta[example]))
                    delta = self.kronecker_delta(label, tlabel)
                    weight = weight + delta * logsigma + (1 - delta) * (
                                logoneminussigma - math.log(len(self.label_set) - 1))

                if weight < math.log(sys.float_info.min):
                    lpd[tlabel] = 0
                else:
                    lpd[tlabel] = math.exp(weight)
                total_weight = total_weight + lpd[tlabel]

            for tlabel in lpd:
                if total_weight == 0:
                    lpd[tlabel] = 1.0 / len(self.label_set)
                else:
                    lpd[tlabel] = lpd[tlabel] * 1.0 / total_weight

            self.e2lpd[example] = lpd

    # M_step

    def gradientQ(self):

        self.dQalpha = {}
        self.dQbeta = {}

        for example, worker_label_set in self.e2wl.items():
            dQb = 0
            for (worker, label) in worker_label_set:
                for tlabel in self.prior.keys():
                    sigma = self.sigmoid(self.alpha[worker] * self.expbeta(self.beta[example]))
                    delta = self.kronecker_delta(label, tlabel)
                    dQb = dQb + self.e2lpd[example][tlabel] * (delta - sigma) * self.alpha[worker] * self.expbeta(
                        self.beta[example])
            self.dQbeta[example] = dQb - (self.beta[example] - self.priorbeta[example])

        for worker, example_label_set in self.w2el.items():
            dQa = 0
            for (example, label) in example_label_set:
                for tlabel in self.prior.keys():
                    sigma = self.sigmoid(self.alpha[worker] * self.expbeta(self.beta[example]))
                    delta = self.kronecker_delta(label, tlabel)
                    dQa = dQa + self.e2lpd[example][tlabel] * (delta - sigma) * self.expbeta(self.beta[example])
            self.dQalpha[worker] = dQa - (self.alpha[worker] - self.prioralpha[worker])

    def computeQ(self):

        Q = 0
        # the expectation of examples given priors, alpha and beta
        for worker, example_label_set in self.w2el.items():
            for (example, label) in example_label_set:
                logsigma = self.logsigmoid(self.alpha[worker] * self.expbeta(self.beta[example]))
                logoneminussigma = self.logoneminussigmoid(self.alpha[worker] * self.expbeta(self.beta[example]))
                for tlabel in self.prior.keys():
                    delta = self.kronecker_delta(label, tlabel)
                    Q = Q + self.e2lpd[example][tlabel] * (
                                delta * logsigma + (1 - delta) * (logoneminussigma - math.log(len(self.label_set) - 1)))

        # the expectation of the sum of priors over all examples
        for example in self.e2wl.keys():
            for tlabel, prob in self.prior.items():
                Q = Q + self.e2lpd[example][tlabel] * math.log(prob)
        # Gaussian (standard normal) prior for alpha
        for worker in self.w2el.keys():
            Q = Q + math.log(
                (pow(2 * math.pi, -0.5)) * math.exp(-pow((self.alpha[worker] - self.prioralpha[worker]), 2) / 2))
        # Gaussian (standard normal) prior for beta
        for example in self.e2wl.keys():
            Q = Q + math.log(
                (pow(2 * math.pi, -0.5)) * math.exp(-pow((self.beta[example] - self.priorbeta[example]), 2) / 2))
        return Q

    def optimize_f(self, x):
        # unpack x
        i = 0
        for worker in self.workers:
            self.alpha[worker] = x[i]
            i = i + 1
        for example in self.examples:
            self.beta[example] = x[i]
            i = i + 1

        return -self.computeQ()  # Flip the sign since we want to minimize

    def optimize_df(self, x):
        # unpack x
        i = 0
        for worker in self.workers:
            self.alpha[worker] = x[i]
            i = i + 1
        for example in self.examples:
            self.beta[example] = x[i]
            i = i + 1

        self.gradientQ()

        # pack x
        der = np.zeros_like(x)
        i = 0
        for worker in self.workers:
            der[i] = -self.dQalpha[worker]  # Flip the sign since we want to minimize
            i = i + 1
        for example in self.examples:
            der[i] = -self.dQbeta[example]  # Flip the sign since we want to minimize
            i = i + 1

        return der

    def Update_alpha_beta(self):

        x0 = []
        for worker in self.workers:
            x0.append(self.alpha[worker])
        for example in self.examples:
            x0.append(self.beta[example])

        #        res = minimize(self.optimize_f, x0, method='BFGS', jac=self.optimize_df,tol=0.01,
        #               options={'disp': True,'maxiter':100})

        res = minimize(self.optimize_f, x0, method='CG', jac=self.optimize_df, tol=0.01,
                       options={'disp': False, 'maxiter': 25})

        self.optimize_f(res.x)

    # likelihood
    def computelikelihood(self):
        L = 0

        for example, worker_label_set in self.e2wl.items():
            L_example = 0;
            for tlabel, prob in self.prior.items():
                L_label = prob
                for (worker, label) in worker_label_set:
                    sigma = self.sigmoid(self.alpha[worker] * self.expbeta(self.beta[example]))
                    delta = self.kronecker_delta(label, tlabel)
                    L_label = L_label * pow(sigma, delta) * pow((1 - sigma) / (len(self.label_set) - 1), 1 - delta)
                L_example = L_example + L_label
            L = L + math.log(L_example)

        for worker in self.w2el.keys():
            L = L + math.log(
                (1 / pow(2 * math.pi, 1 / 2)) * math.exp(-pow((self.alpha[worker] - self.prioralpha[worker]), 2) / 2))

        for example in self.e2wl.keys():
            L = L + math.log(
                (1 / pow(2 * math.pi, 1 / 2)) * math.exp(-pow((self.beta[example] - self.priorbeta[example]), 2) / 2))

    # initialization
    def Init_prior(self):
        # uniform probability distribution
        prior = {}
        for label in self.label_set:
            prior[label] = 1.0 / len(self.label_set)
        return prior

    def Init_alpha_beta(self):
        prioralpha = {}
        priorbeta = {}
        for worker in self.w2el.keys():
            prioralpha[worker] = 1
        for example in self.e2wl.keys():
            priorbeta[example] = 1
        return prioralpha, priorbeta

    def get_workerquality(self):
        sum_worker = sum(self.alpha.values())
        norm_worker_weight = dict()
        for worker in self.alpha.keys():
            norm_worker_weight[worker] = self.alpha[worker] / sum_worker
        return norm_worker_weight

    def run(self, threshold=1e-4):

        self.prior = self.Init_prior()
        self.prioralpha, self.priorbeta = self.Init_alpha_beta()

        self.alpha = self.prioralpha
        self.beta = self.priorbeta

        Q = 0
        self.Update_e2lpd()
        Q = self.computeQ()

        # print("启动了。。。。。。。。。。。。。")
        while True:
            lastQ = Q

            # E-step
            self.Update_e2lpd()
            Q = self.computeQ()
            # print(Q)

            # M-step
            self.Update_alpha_beta()
            Q = self.computeQ()
            # print(Q)

            # compute the likelihood
            # print self.computelikelihood()

            if (math.fabs((Q - lastQ) / lastQ)) < threshold:
                break

        return self.e2lpd, self.alpha

    ###################################
    # The above is the EM method (a class)
    # The following are several external functions
    ###################################

    def get_accuracy(self):
        e2truth = {}
        f = open(self.truth_file, 'r')
        reader = csv.reader(f)
        next(reader)

        for line in reader:
            # example, truth = line
            example, truth = line[0:2]
            e2truth[example] = truth

        tcount = 0
        count = 0
        e2lpd = self.e2lpd
        for e in e2lpd:

            if e not in e2truth:
                continue

            temp = 0
            for label in e2lpd[e]:
                if temp < e2lpd[e][label]:
                    temp = e2lpd[e][label]

            candidate = []

            for label in e2lpd[e]:
                if temp == e2lpd[e][label]:
                    candidate.append(label)

            truth = random.choice(candidate)

            count += 1

            if truth == e2truth[e]:
                tcount += 1

        return tcount * 1.0 / count

    def gete2wlandw2el(self, datafile):
        e2wl = {}
        w2el = {}
        label_set = []

        f = open(datafile, 'r')
        reader = csv.reader(f)
        next(reader)

        for line in reader:
            # example, worker, label = line
            example, worker, label = line[4:7]
            if example not in e2wl:
                e2wl[example] = []
            e2wl[example].append([worker, label])

            if worker not in w2el:
                w2el[worker] = []
            w2el[worker].append([example, label])

            if label not in label_set:
                label_set.append(label)

        return e2wl, w2el, label_set

def GLAD(collusion_P, list, truth_file, Alg):
    W = np.zeros((len(list), 30))
    # 串谋机会
    for p in range(len(collusion_P)):

        for i in range(len(list)):

            for epo in range(30):
                print(list[i], epo, "/", 30)
                datafile = "../../../experience/result/collusion_data/collusion_" + str(
                    collusion_P[p]) + "/experience_" + str(list[i]) + "/data" + str(epo) + "_collaborate_D_"+Alg+".csv"

                # print(path)
                print("collusion_P=", collusion_P, " percentage=", list[i], "epoch=", epo)
                truth_file = truth_file

                glad = get_GLAD_acc(datafile, truth_file=truth_file)
                glad.run()
                # print(weight)
                # print(e2lpd)
                # truthfile = sys.argv[2]
                accuracy = glad.get_accuracy()
                print('The GLAD accuracy is %f' % accuracy)

                W[i][epo] = accuracy
            save_path = "../../../experience/acc_result_del/"+Alg+"_GLAD_acc_collusion_"+str(collusion_P[p])+"-"+str(list[i])+".csv"
            np.savetxt(save_path, W, delimiter=",", fmt='%.8f')




if __name__ == '__main__':
    """
    计算串谋工人参与后，准确率情况，即：
    CFC-MV：
    CFC-GLAD：
    CFC-HDS：
    """

    truth_file = "../../../data/result_file.csv"
    gt = pd.read_csv(truth_file)


    collusion_P = [0.8]
    percentage = [36]


    """MV acc"""
    MV(collusion_P, percentage, gt, "FC")


    #
    # """HDS acc"""
    # HDS(collusion_P, percentage, truth_file=truth_file)
    #
    #
    #
    # """GLAD acc"""
    # GLAD(collusion_P, percentage, truth_file)
