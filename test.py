def bigger(a,b):
    if a>b:
        return a
    else:
        return b
def swapRows(M, r1, r2):
#     pass

    M[r1],M[r2] = M[r2],M[r1]
def scaleRow(M, r, scale):
    pass
    
    if not scale == 0:
        M[r]=[scale*x for x in M[r]]
    else:
        raise ValueError 
def addScaledRow(M, r1, r2, scale):
    pass
    for i in range(len(M[r1])):
        M[r1][i] = M[r1][i] + scale*M[r2][i]
def shape(M):
    row_num = len(M)
    col_num = len(M[0])
    return row_num,col_num
def augmentMatrix(A, b):
    Ab = [[0]*(len(A[0])+1) for i in range (len(A))]
    for i in range (len(A)):
        for j in range (len(A[0])):
            Ab[i][j] = A[i][j]
        Ab[i][len(A[0])] = b[i]
    return Ab 

def gj_Solve(A, b, decPts=4, epsilon = 1.0e-16):

    num_row_A , num_col_A = shape(A)
    num_row_b = len(b)
    if num_row_A == num_row_b :
        Ab = augmentMatrix(A, b) 
        for i in range(0,num_col_A):
            
            m = i
            a = 0
            for j in range(i,num_row_A):
                a = bigger(a,abs(Ab[j][i]))
                if a == abs(Ab[j][i]):
                    m = j
                if not a == abs(Ab[j][i]):
                    m = m

            if a <= epsilon:
                    return None
            if a > epsilon:
                swapRows(Ab,i,m)
                scaleRow(Ab,i,1.0/a)
                for d in range(0,num_row_A):
                    if not d == m:
                        if Ab[d][i] != 0 :
                            addScaledRow(Ab,d,d,-1.0/Ab[d][i])
                            return Ab
                               
        return Ab
    if num_row_A != num_row_b:
        return None
    
A = [[1,2,0],[0,1,0],[0,0,1]]
b = [1,2,1]

print gj_Solve(A,b)