import numpy as np
from numpy import linalg 

class PCA:

    '''
    dataset 形如array([样本1,样本2,...,样本m]),每个样本是一个n维的ndarray
    '''
    def __init__(self, dataset):
    	# 这里的参数跟上文是反着来的(每行是一个样本)，需要转置一下
        self.dataset = np.matrix(dataset, dtype='float64').T

    '''
    求主成分;
    threshold可选参数表示方差累计达到threshold后就不再取后面的特征向量.
    '''
    def principal_comps(self, threshold = 0.85):
    	# 返回满足要求的特征向量
        ret = []
        data = []

		# 标准化
        for (index, line) in enumerate(self.dataset):
            self.dataset[index] -= np.mean(line)
            # np.std(line, ddof = 1)即样本标准差(分母为n - 1)
            self.dataset[index] /= np.std(line, ddof = 1)
        # 求协方差矩阵
        Cov = np.cov(self.dataset)
		# 求特征值和特征向量
        eigs, vectors = linalg.eig(Cov)
		# 第i个特征向量是第i列，为了便于观察将其转置一下
        for i in range(len(eigs)):
            data.append((eigs[i], vectors[:, i].T))
        # 按照特征值从大到小排序
        data.sort(key = lambda x: x[0], reverse = True)

        sum = 0
        for comp in data:
            sum += comp[0] / np.sum(eigs)
            ret.append(
                tuple(map(
                	# 保留5位小数
                    lambda x: np.round(x, 5),
                    # 特征向量、方差贡献率、累计方差贡献率
                    (comp[1], comp[0] / np.sum(eigs), sum)
                ))
            )
            print('特征值:', comp[0], '特征向量:', ret[-1][0], '方差贡献率:', ret[-1][1], '累计方差贡献率:', ret[-1][2])
            if sum > threshold:
                return ret      

        return ret 

p = PCA(
[[66, 64, 65, 65, 65],
 [65, 63, 63, 65, 64],
 [57, 58, 63, 59, 66],
 [67, 69, 65, 68, 64],
 [61, 61, 62, 62, 63],
 [64, 65, 63, 63, 63],
 [64, 63, 63, 63, 64],
 [63, 63, 63, 63, 63],
 [65, 64, 65, 66, 64],
 [67, 69, 69, 68, 67],
 [62, 63, 65, 64, 64],
 [68, 67, 65, 67, 65],
 [65, 65, 66, 65, 64],
 [62, 63, 64, 62, 66],
 [64, 66, 66, 65, 67]]
)

lst = p.principal_comps()

print(lst)
