#  Data Is About Advertising Spending And Product Sales(unit)
import numpy as np
import math
import matplotlib.pyplot as plt


lst = np.array([[1000, 50],
                [1500, 89],
                [1650, 70],
                [1900, 100],
                [2300, 150],
                [2450, 100],
                [2500, 102],
                [2560, 143],
                [3000, 250],
                [3000, 50],
                [3520, 400],
                [4600, 600],
                [4900, 650],
                [5600, 670],
                [6000, 640],
                [6800, 875],
                [9000, 850],
                [9340, 860],
                [9400, 900],
                [9670, 1000],
                [9900, 1300],
                [10000, 1563]])


x_values = lst[:, 0]
y_values = lst[:, 1]




result_vector = y_values


A_Matrix = np.column_stack((np.ones_like(lst[:, 0]), lst[:, 0])) # I'm creating array in which every row's first element is 1 and second is x variable from 'lst' array respectably.







def inverse(matrix,vector): # this returns vector, in which first element is number intersected on y axis (b), second is slope (k).

    return np.linalg.inv(np.transpose(matrix).dot(matrix)).dot(np.transpose(matrix)).dot(vector)    # (A_T * A)^-1 * A_T * b





def CGM(matrix): # Returns columns of Q matrix
    frst_column = matrix[:,0]
    scnd_column = matrix[:,1]
    frst_length = math.sqrt(sum(i**2 for i in frst_column))


    U_1 = (1/frst_length) * frst_column

    U_2 = scnd_column - (np.dot(U_1 , scnd_column)) * U_1
    scnd_length = math.sqrt(sum(i**2 for i in U_2))
    U_2 = (1/scnd_length) * U_2
    return U_1,U_2





def MGM(matrix): # Returns Q and R matrices
    Q = matrix.astype(float)
    R = [[0,0],
         [0,0]]
    for k in range(len(matrix[0])):
        R[k][k] = np.linalg.norm(Q[:,k])
        Q[:,k] = (1/R[k][k]) * Q[:,k]
        for j in range(k+1,len(matrix[0])):
            R[k][j] = np.dot(np.transpose(Q[:,k]) , Q[:,j])
            Q[:,j] = Q[:,j] - (R[k][j]*Q[:,k])

    return Q,R




#  Line With Inverse 
solved = inverse(A_Matrix,result_vector)
y_line = solved[1] * x_values + solved[0]




# Line With CGM

U1,U2 = CGM(A_Matrix)
Q = np.hstack((U1.reshape(-1,1),U2.reshape(-1,1)))
R = np.transpose(Q).dot(A_Matrix)
R[1][0] = 0. # there was some slight miscalculation so i make R upper triangular manually.
QR_result = np.dot(np.transpose(Q) , result_vector)
QR_result = np.dot(np.linalg.inv(R) , QR_result)

y_line_qr = QR_result[1] * x_values + QR_result[0]





# Line With MGM

Q,R = MGM(A_Matrix)
QR_result_2 = np.dot(np.linalg.inv(R) , np.dot(np.transpose(Q),result_vector))
y_line_qr_MGM = QR_result_2[1] * x_values + QR_result_2[0]











"""  Constrained LS """


C = np.array([1, 1])
C_transp = np.array([[1], [1]])
D = np.array([5])

# Construct block matrix KKT
block1 = np.dot(2, np.dot(A_Matrix.T, A_Matrix))
block2 = C_transp
block3 = C
block4 = np.array([[0]])

KKT = np.block([[block1, block2], [block3, block4]])



block1 = np.dot(2 , np.dot(A_Matrix.T , result_vector))

final = np.concatenate((block1,D))
final = np.dot(np.linalg.inv(KKT) , final)
result = final[:2]
y_line_const = result[1] * x_values + result[0]




"""   -----   """









plt.scatter(x_values, y_values, label='Data Points')


plt.plot(x_values,y_line,label = f"{solved[1]}x + {solved[0]}",color = 'red')
plt.plot(x_values,y_line_qr, label = f"{QR_result[1]}x + {QR_result[0]}", color = 'blue')
plt.plot(x_values,y_line_qr_MGM, label = f"{QR_result_2[1]}x + {QR_result_2[0]}", color = 'black')
plt.plot(x_values,y_line_const,label = f"{result[0]}x + {result[1]}", color = 'silver')

plt.xlabel('Advertising Spending($)')
plt.ylabel('Product Sales(unit)')
plt.title('XY Plane')
plt.legend()
plt.grid(True)
plt.show()







# Inverse Method:

# Advantage: Directly provides coefficients.
# Drawback: Sensitive to numerical stability issues.

# Classic Gram-Schmidt:

# Advantage: Numerically stable.
# Drawback: May have slight numerical errors.

# Modified Gram-Schmidt:

# Advantage: Improved numerical stability.
# Drawback: Still iterative, but less prone to errors.

# Constrained Least Squares (KKT):

# Advantage: Incorporates a constraint.
# Drawback: Requires solving a system of linear equations.


# The choice of method should consider trade-offs between computational complexity, numerical stability, and the need for constraints.
# The inverse method is simple but may suffer from numerical instability.
# Gram-Schmidt methods are more stable but may still have numerical errors.
# Constrained least squares is suitable for incorporating constraints, but it involves solving a system of linear equations.