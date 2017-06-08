
# coding: utf-8

# # 1 矩阵运算
# 
# ## 1.1 创建一个 4*4 的单位矩阵

# In[97]:

# 这个项目设计来帮你熟悉 python list 和线性代数
# 你不能调用任何python库，包括NumPy，来完成作业

A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]

B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]

#TODO 创建一个 4*4 单位矩阵
I = [[1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]]

# print I,'\n',A


# ## 1.2 返回矩阵的行数和列数

# In[98]:

# TODO 返回矩阵的行数和列数
def shape(M):
    row_num = len(M)
    col_num = len(M[0])
    return row_num,col_num


# ## 1.3 每个元素四舍五入到特定小数数位

# In[99]:

# TODO 每个元素四舍五入到特定小数数位
# 直接修改参数矩阵，无返回值
def matxRound(M, decPts=4):
    for x in range(0,len(M)):
        for y in range(0,len(M[0])):
            M[x][y] = round(M[x][y],decPts)
    pass


# ## 1.4 计算矩阵的转置

# In[100]:

# TODO 计算矩阵的转置
def transpose(M):
    row_num,col_num = shape(M)
    MT = []
    
    for c in range(0,col_num):
        MTi = []
        for r in range(0,row_num):
            MTi.append(M[r][c])
        MT.append(MTi)
    return MT


# ## 1.5 计算矩阵乘法 AB

# In[101]:

# TODO 计算矩阵乘法 AB，如果无法相乘则返回None
def matxMultiply(A, B):
    def dot_product(C,D):
        n = len(C)
        sum = 0
        for i in range(0,n):
            a = C[i]*D[i]
            sum += a
        return sum
    
    # 判断是否可以相乘
    row_num_A, col_num_A = shape(A)
    row_num_B, col_num_B = shape(B)
    E = transpose(B)
    if col_num_A != row_num_B:
        return None
    else:
        C = []
        for i in range(0,row_num_A):
            Ci = []
            for j in range(0,col_num_B):
                a = dot_product(A[i],E[j])
                Ci.append(a)
            C.append(Ci)
    return C

#print matxMultiply(A,B)


# ## 1.6 测试你的函数是否实现正确

# **提示：** 你可以用`from pprint import pprint`来更漂亮的打印数据，详见[用法示例](http://cn-static.udacity.com/mlnd/images/pprint.png)和[文档说明](https://docs.python.org/2/library/pprint.html#pprint.pprint)。

# In[102]:

import pprint
pp = pprint.PrettyPrinter(indent=1,width=40)
#TODO 测试1.2 返回矩阵的行和列
pp.pprint(shape(A))
print ''

#TODO 测试1.3 每个元素四舍五入到特定小数数位
C = [[1.01928,0.398747,1.23456],[0.347758,2.446672,0.35521]]
matxRound(C)
pp.pprint(C)
print ''

#TODO 测试1.4 计算矩阵的转置
b = transpose(B)
pp.pprint(b)
bb = transpose(b)
pp.pprint(bb)
print ''

#TODO 测试1.5 计算矩阵乘法AB，AB无法相乘
C = [[0,1,2],[1,2,3]]
b = matxMultiply(A,C)
pp.pprint(b)
print ''
#TODO 测试1.5 计算矩阵乘法AB，AB可以相乘
b = matxMultiply(A,B)
pp.pprint(b)
print ''


# # 2 Gaussign Jordan 消元法
# 
# ## 2.1 构造增广矩阵
# 
# $ A = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n}\\
#     a_{21}    & a_{22} & ... & a_{2n}\\
#     a_{31}    & a_{22} & ... & a_{3n}\\
#     ...    & ... & ... & ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn}\\
# \end{bmatrix} , b = \begin{bmatrix}
#     b_{1}  \\
#     b_{2}  \\
#     b_{3}  \\
#     ...    \\
#     b_{n}  \\
# \end{bmatrix}$
# 
# 返回 $ Ab = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n} & b_{1}\\
#     a_{21}    & a_{22} & ... & a_{2n} & b_{2}\\
#     a_{31}    & a_{22} & ... & a_{3n} & b_{3}\\
#     ...    & ... & ... & ...& ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn} & b_{n} \end{bmatrix}$

# In[103]:

# TODO 构造增广矩阵，假设A，b行数相同


#from decimal import *
def augmentMatrix(A, b):
    
    Ab = [[0]*(len(A[0])+1) for i in range (len(A))]

    for i in range (len(A)):
        for j in range (len(A[0])):
            Ab[i][j] = A[i][j]
        Ab[i][len(A[0])] = b[i][0]
    return Ab 

'''                      
    num_row_A = len(A)
    num_row_b = len(b) 
    num_col_A = len(A[0])
    if  not num_row_A == num_row_b:
        return None
    else:
        C = []
        for i in range(0,num_row_A):
            Ci = [] 
            
            for j in range(0,num_col_A):
                
                Ci.append(A[i][j])
            Ci.append(b[i])
            C.append(Ci)
    return C '''
    
c = [[1],[2],[3]]

A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]
print augmentMatrix(A,c) 


# ## 2.2 初等行变换
# - 交换两行
# - 把某行乘以一个非零常数
# - 把某行加上另一行的若干倍：

# In[104]:

# TODO r1 <---> r2 
# 直接修改参数矩阵，无返回值
def swapRows(M, r1, r2):
#     pass

    M[r1],M[r2] = M[r2],M[r1]
#    return M

        
    

# TODO r1 <--- r1 * scale， scale!=0
# 直接修改参数矩阵，无返回值
def scaleRow(M, r, scale):
    pass
    
    if not scale == 0:
        M[r]=[scale*x for x in M[r]]
    else:
        raise ValueError 
        
# TODO r1 <--- r1 + r2*scale
# 直接修改参数矩阵，无返回值
def addScaledRow(M, r1, r2, scale):
#     pass
    for i in range(len(M[r1])):
        M[r1][i] = M[r1][i] + scale*M[r2][i]
'''    num_col = len(M[0])

    if not scale == 0:
        scaleRow(M,r2,scale)
        M[r1]=[x+y for x,y in zip(M[r1],M[r2])]

    else:
        raise False
        '''

A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]
addScaledRow(A,0,2,2)
print A


# ## 2.3  Gaussian Jordan 消元法求解 Ax = b

# In[ ]:




# ### 提示：
# 
# 步骤1 检查A，b是否行数相同
# 
# 步骤2 构造增广矩阵Ab
# 
# 步骤3 逐列转换Ab为化简行阶梯形矩阵 [中文维基链接](https://zh.wikipedia.org/wiki/%E9%98%B6%E6%A2%AF%E5%BD%A2%E7%9F%A9%E9%98%B5#.E5.8C.96.E7.AE.80.E5.90.8E.E7.9A.84-.7Bzh-hans:.E8.A1.8C.3B_zh-hant:.E5.88.97.3B.7D-.E9.98.B6.E6.A2.AF.E5.BD.A2.E7.9F.A9.E9.98.B5)
#     
#     对于Ab的每一列（最后一列除外）
#         当前列为列c
#         寻找列c中 对角线以及对角线以下所有元素（行 c~N）的绝对值的最大值
#         如果绝对值最大值为0
#             那么A为奇异矩阵，返回None （请在问题2.4中证明该命题）
#         否则
#             使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行c） 
#             使用第二个行变换，将列c的对角线元素缩放为1
#             多次使用第三个行变换，将列c的其他元素消为0
#             
# 步骤4 返回Ab的最后一列
# 
# ### 注：
# 我们并没有按照常规方法先把矩阵转化为行阶梯形矩阵，再转换为化简行阶梯形矩阵，而是一步到位。如果你熟悉常规方法的话，可以思考一下两者的等价性。

# In[105]:

# TODO 实现 Gaussain Jordan 方法求解 Ax = b

""" Gaussian Jordan 方法求解 Ax = b.
    参数
        A: 方阵 
        b: 列向量
        decPts: 四舍五入位数，默认为4
        epsilon: 判读是否为0的阈值，默认 1.0e-16
        
    返回列向量 x 使得 Ax = b 
    返回None，如果 A，b 高度不同
    返回None，如果 A 为奇异矩阵
"""

def gj_Solve(A, b, decPts=4, epsilon = 1.0e-16):
    # 验证矩阵形状是否相符
    num_row_A , num_col_A = shape(A)
    num_row_b = len(b)
    if num_row_b != num_row_A:
        #print 'shape error'
        return None
    else:
        # 构造增广矩阵 Ab
        Ab = augmentMatrix(A,b)
        #print Ab        
        # 对于除最后一列外的每一列 c
        i = 0
        while i< num_col_A:
            Ab_T = transpose(Ab)
            c = Ab_T[i]
            # 通过绝对值的最大值判断矩阵形式
            c_a = [abs(ii) for ii in c[i:]]
            c_a_max =  max([abs(ii) for ii in c[i:]])
            c_a_max_index = c_a.index(c_a_max)+i #最大值的行号
            if c_a_max <= epsilon:
                #print '矩阵为奇异矩阵'
                return None
            else:
                # 使用第一个行变换交换最大值行和对角线行
                swapRows(Ab,i,c_a_max_index)                
                # 使用第二个行变换使对角线值归一
                scaleRow(Ab,i,1./Ab[i][i])
                # 使用第三个行变换遍历除对角线位置外的所有行，使其归零
                for ii in range(0,len(Ab)):
                    if ii != i:
                        addScaledRow(Ab,ii,i,-1.0*Ab[ii][i]/Ab[i][i])
                #print Ab
            i += 1
        #print Ab
    N = transpose(Ab)[-1]
    return [[N[j]] for j in range(len(N))]





# ## 2.4 证明下面的命题：
# 
# **如果方阵 A 可以被分为4个部分: ** 
# 
# $ A = \begin{bmatrix}
#     I    & X \\
#     Z    & Y \\
# \end{bmatrix} , \text{其中 I 为单位矩阵，Z 为全0矩阵，Y 的第一列全0}$，
# 
# **那么A为奇异矩阵。**
# 
# 
# 提示：从多种角度都可以完成证明
# - 考虑矩阵 Y 和 矩阵 A 的秩
# - 考虑矩阵 Y 和 矩阵 A 的行列式
# - 考虑矩阵 A 的某一列是其他列的线性组合

# TODO 请使用 latex （请参照题目的 latex 写法学习）
# 
# TODO 证明：$  \left|A \right| =  \left|I \right| \times  \left|Y \right| -  \left|Z \right| \times  \left|X \right| $
#            其中Z为全0矩阵，Y的第一列全为0 ,I为单位矩阵。
#            故有 $  \left|Z \right| = 0 ，\left|Y \right| = 0 ，\left|I \right| = 1 $ 
#            所以 $  \left|A \right| =  1 \times  0 -  0 \times  \left|X \right| $
#            所以 $  \left|A \right| = 0 $ ，A为奇异矩阵。
# 
#        
# 

# ## 2.5 测试 gj_Solve() 实现是否正确

# In[106]:

# TODO 构造 矩阵A，列向量b，其中 A 为奇异矩阵
A = [[0,1,2],[0,2,3],[0,0,1]]
b = [[1],[2],[3]]
print gj_Solve(A,b)
  

# TODO 构造 矩阵A，列向量b，其中 A 为非奇异矩阵
A = [[1,2,0],[0,1,0],[0,0,1]]
b = [[1],[2],[1]]


# TODO 求解 x 使得 Ax = b
x = gj_Solve(A,b)
print x
# TODO 计算 Ax
num_row_A , num_col_A = shape(A)
s = []
for i in range(0,3):
    s.append(x[i][0])
print s
Ax = [[0] for i in range(len(b))]
print Ax
for i in range(0,num_row_A):
    c = [m*y for m,y in zip(s[:],A[i][:])]    
    Ax[i] = [sum(c)]
# TODO 比较 Ax 与 b
print Ax == b


# # 3 线性回归: 
# 
# ## 3.1 计算损失函数相对于参数的导数 (两个3.1 选做其一)
# 
# 我们定义损失函数 E ：
# $$
# E = \sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# 
# 证明：
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}
# $$
# 
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# $$
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = 2X^TXh - 2X^TY
# $$
# 
# $$ 
# \text{其中 }
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$

# TODO 请使用 latex （参照题目的 latex写法学习）
# 
# TODO 证明：$$  X^T = \begin{bmatrix}
#     x_1 & x_2 & ... & x_n \\
#     1 & 1 & ... & 1\\
# \end{bmatrix}, $$
# 
# $$
# X^TXh = \begin{bmatrix}
#     x_1 & x_2 & ... & x_n \\
#     1 & 1 & ... & 1\\
# \end{bmatrix}  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix} \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}= \begin{bmatrix}
#      \sum_{i=1}^{n}\limits{x_i^2} & \sum_{i=1}^{n}\limits{x_i}\\
#     \sum_{i=1}^{n}\limits{x_i} & n\\
# \end{bmatrix} \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix} $$
# 
# $$
# X^TXh = \begin{bmatrix}
#      \sum_{i=1}^{n}\limits{x_i^2} & \sum_{i=1}^{n}\limits{x_i}\\
#     \sum_{i=1}^{n}\limits{x_i} & n\\
# \end{bmatrix} \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix} = \begin{bmatrix}
#      \sum_{i=1}^{n}\limits{x_i(mx_i+b)} \\
#      \sum_{i=1}^{n}\limits{mx_i+b}\\
# \end{bmatrix} $$
# 
# $$
# X^TY = \begin{bmatrix}
#     x_1 & x_2 & ... & x_n \\
#     1 & 1 & ... & 1\\
# \end{bmatrix} \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix} = \begin{bmatrix}
#     \sum_{i=1}^{n}\limits{x_iy_i} \\
#     \sum_{i=1}^{n}\limits{y_i} \\
# \end{bmatrix}
# $$
# 
# $$
# 2X^TXh-2X^TY = 2\begin{bmatrix}
#      \sum_{i=1}^{n}\limits{x_i(mx_i+b)} \\
#      \sum_{i=1}^{n}\limits{mx_i+b}\\
# \end{bmatrix}-2\begin{bmatrix}
#     \sum_{i=1}^{n}\limits{x_iy_i} \\
#     \sum_{i=1}^{n}\limits{y_i} \\
# \end{bmatrix} = \begin{bmatrix}
#      \sum_{i=1}^{n}\limits{-2x_i(y_i-mx_i-b)} \\
#      \sum_{i=1}^{n}\limits{-2(y_i-mx_i-b}\\
# \end{bmatrix} = \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} 
# $$
# 
# $$
# 所以\begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = 2X^TXh - 2X^TY 成立
# $$
#         

# ## 3.1 计算损失函数相对于参数的导数（两个3.1 选做其一）
# 
# 证明：
# 
# $$
# E = Y^TY -2(Xh)^TY + (Xh)^TXh
# $$ 
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix}  = \frac{\partial E}{\partial h} = 2X^TXh - 2X^TY
# $$
# 
# $$ 
# \text{其中 }
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$

# TODO 请使用 latex （请参照题目的 latex 写法学习）
# 
# TODO 证明：

# ## 3.2  线性回归
# 
# ### 求解方程 $X^TXh = X^TY $, 计算线性回归的最佳参数 h

# In[107]:

# TODO 实现线性回归
'''
参数：(x,y) 二元组列表
返回：m，b
'''
def linearRegression(points):
    x = []
    Y = []

        
    for i in range(len(points)):
        x.append(points[i][0])
        Y.append([points[i][1]])

    
    X = [[a,1] for a in x[:]]

    X_T = transpose(X)
    X_T_X = matxMultiply(X_T,X)
    X_T_Y = matxMultiply(X_T,Y)
    h = gj_Solve(X_T_X,X_T_Y)
    return h 


# ## 3.3 测试你的线性回归实现

# In[132]:

# TODO 构造线性函数
#y = 1 * x + 1 + delta
# TODO 构造 100 个线性函数上的点，加上适当的高斯噪音
import random
import numpy as np
x = np.arange(100)
delta = np.random.uniform(-1,1, size=(100,))
y = 1 * x + 1 + delta
x = x.tolist()
y = y.tolist()
#print x,y
points = [[i,ii] for i,ii in zip(x[:],y[:])]
#print points  

#TODO 对这100个点进行线性回归，将线性回归得到的函数和原线性函数比较
h = linearRegression(points)
print h
loss = ((h[0][0]-1)**2+(h[0][0]-1)**2)/2

print loss < 1e-2


# ## 4.1 单元测试
# 
# 请确保你的实现通过了以下所有单元测试。

# In[134]:

import unittest
import numpy as np

from decimal import *

class LinearRegressionTestCase(unittest.TestCase):
    """Test for linear regression project"""

    def test_shape(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.randint(low=-10,high=10,size=(r,c))
            self.assertEqual(shape(matrix.tolist()),(r,c))


    def test_matxRound(self):

        for decpts in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()
            dec_true = [[Decimal(str(round(num,decpts))) for num in row] for row in mat]

            matxRound(mat,decpts)
            dec_test = [[Decimal(str(num)) for num in row] for row in mat]

            res = Decimal('0')
            for i in range(len(mat)):
                for j in range(len(mat[0])):
                    res += dec_test[i][j].compare_total(dec_true[i][j])

            self.assertEqual(res,Decimal('0'))


    def test_transpose(self):
        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()
            t = np.array(transpose(mat))

            self.assertEqual(t.shape,(c,r))
            self.assertTrue((matrix.T == t).all())


    def test_matxMultiply(self):

        for _ in range(10):
            r,d,c = np.random.randint(low=1,high=25,size=3)
            mat1 = np.random.randint(low=-10,high=10,size=(r,d)) 
            mat2 = np.random.randint(low=-5,high=5,size=(d,c)) 
            dotProduct = np.dot(mat1,mat2)

            dp = np.array(matxMultiply(mat1,mat2))

            self.assertTrue((dotProduct == dp).all())


    def test_augmentMatrix(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            A = np.random.randint(low=-10,high=10,size=(r,c))
            b = np.random.randint(low=-10,high=10,size=(r,1))
           
            Ab = np.array(augmentMatrix(A.tolist(),b.tolist()))
            
            ab = np.hstack((A,b))
            
            
            self.assertTrue((Ab == ab).all())

    def test_swapRows(self):
        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            r1, r2 = np.random.randint(0,r, size = 2)
            swapRows(mat,r1,r2)

            matrix[[r1,r2]] = matrix[[r2,r1]]

            self.assertTrue((matrix == np.array(mat)).all())

    def test_scaleRow(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            rr = np.random.randint(0,r)
            with self.assertRaises(ValueError):
                scaleRow(mat,rr,0)

            scale = np.random.randint(low=1,high=10)
            scaleRow(mat,rr,scale)
            matrix[rr] *= scale

            self.assertTrue((matrix == np.array(mat)).all())
            
    
    def test_addScaleRow(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            r1,r2 = np.random.randint(0,r,size=2)

            scale = np.random.randint(low=1,high=10)
            addScaledRow(mat,r1,r2,scale)
            matrix[r1] += scale * matrix[r2]

            self.assertTrue((matrix == np.array(mat)).all())


    def test_gj_Solve(self):

        for _ in range(10):
            r = np.random.randint(low=3,high=10)
            A = np.random.randint(low=-10,high=10,size=(r,r))
            b = np.arange(r).reshape((r,1))
            #print A,b
            x = gj_Solve(A.tolist(),b.tolist())
            if np.linalg.matrix_rank(A) < r:
                self.assertEqual(x,None)
            else:
                #Ax = matxMultiply(A.tolist(),x)
                Ax = np.dot(A,np.array(x))
                loss = np.mean((Ax - b)**2)                
                #print Ax
                #print b
                #print loss                
                self.assertTrue(loss<0.1)


suite = unittest.TestLoader().loadTestsFromTestCase(LinearRegressionTestCase)
unittest.TextTestRunner(verbosity=3).run(suite)


# In[ ]:



