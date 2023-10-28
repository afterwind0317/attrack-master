import numpy as np
from matplotlib import pyplot as plt

def func(m,n):
    a=b=result=1
    if m<n:
        print("n不能小于m 且均为整数")
    elif ((type(m)!=int)or(type(n)!=int)):
        print("n不能小于m 且均为整数")
    else:
        minNI=min(n,m-n)#使运算最简便
        for j in range(0,minNI):
        #使用变量a,b 让所用的分母相乘后除以所有的分子
            a=a*(m-j)
            b=b*(minNI-j)
            result=a//b #在此使用“/”和“//”均可，因为a除以b为整数
        return result




def weight(a1, a2):
    c = 4
    W = []
    X = np.arange(1, 51)
    f = a1*a2 + (c-1)*((1-a1)/(c-1))*((1-a2)/(c-1))
    print(f)
    for i in range(len(X)):
        x = int(X[i])
        C = func(50, x)
        # print(C,"*",f,"^",x,"*",(1-f),"^",(50-x))
        F = C * (f**x) * ((1-f)**(50-x))
        W.append(F)
    return W



if __name__ == '__main__':

    x = np.arange(1, 51)
    x = x/50
    labels = ['0', '0.2', '0.4', '0.6', '0.8', '1.0']

    y0 = weight(0.5, 0.5)
    y1 = weight(0.5, 0.9)
    y2 = weight(0.6, 0.8)
    y3 = weight(0.8, 0.8)
    y4 = weight(0.9, 0.9)

    plt.xlabel('Consistency rate of answers', fontsize=14)
    plt.ylabel('Density', fontsize=14)

    plt.plot(x, y0, '--', color='gray', linewidth=1.5, markersize=3, label=r'$\alpha_{1}=0.5,\alpha_{2}=0.5$')
    plt.plot(x, y1, '--', color='y', linewidth=1.5, markersize=3, label=r'$\alpha_{1}=0.5,\alpha_{2}=0.9$')  # 绘制折线图，添加数据点，设置点的大小
    plt.plot(x, y2, '-.', color='g', linewidth=1.5, markersize=3, label=r'$\alpha_{1}=0.6,\alpha_{2}=0.8$')
    plt.plot(x, y3, '-', color='b', linewidth=1.5, markersize=3, label=r'$\alpha_{1}=0.8,\alpha_{2}=0.8$')
    plt.plot(x, y4, '.-', color='r', linewidth=1.5, markersize=3, label=r'$\alpha_{1}=0.9,\alpha_{2}=0.9$')


    plt.legend(loc='upper left')
    plt.tick_params(top=True, bottom=True, left=True, right=True)
    plt.tick_params(direction='in')
    plt.grid(axis='y', alpha=0.3)
    plt.grid(axis='x', alpha=0.3)

    path = 'Collusion_Proof.jpg'
    plt.gcf().savefig(path, dpi=1800, format='jpg')


    plt.show()