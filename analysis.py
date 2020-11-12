# this file contains the fixed point optimization
# also PCA and other analysis needed on the models
import torch
import random
import numpy as np
from scipy.optimize import minimize
from sklearn import decomposition
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


"""
    find approximate fixed points that are state vectors {h_1*, h_2*, h_3*, ...} where h∗i ≈F(h∗i,x=0)
    defining a loss function q = 1/N * ||(h - F(h,0) || _2 ^2
    and then minimizing q with respect to hidden states, h, using auto-differentiation methods.
    Run this optimization multiple times starting from different initial values of h.
    These initial conditions sampled randomly from the distribution of state activations explored by
    the trained network, which was done to intentionally sample states related to the operation of the RNN.
"""


# generate random integer to sample some random state of activations:


def loss_func(_hidden_state, model, num_hidden_nodes):
    # tensor full of zeros
    zero_input = torch.zeros(1, 1, dtype=torch.long)
    len_zero_input = torch.empty(1)
    len_zero_input[0] = zero_input.shape[1]
    # create hidden state tensor of previous step
    _hs_prev = torch.from_numpy(_hidden_state)
    _hs_prev = _hs_prev.reshape((1, 1, num_hidden_nodes))

    # len_zero_input holds the length of the zero input vector
    # deactivates autograd
    with torch.no_grad():
        # run through 1 time step, thus h_(t) = f(h_(t-1), 0), where
        # input x = 0, and h_(t-1) is the hidden state for time step t-1
        _, f_of_zero_input = model(zero_input, len_zero_input, _hs_prev)
        f_of_zero_input = f_of_zero_input.detach().numpy()
        elem_wise_sub = np.subtract(_hidden_state, f_of_zero_input)
        elem_wise_square = np.square(elem_wise_sub)
        elem_sum = np.sum(elem_wise_square)
        # q is the loss function we want to minimize as written in the paper
        q = elem_sum / elem_wise_sub.size
        return q


# def find_fixed_points(_hidden_list):
#     _fixed_point_list = []
#     for _ in range(700):
#         # print(hidden_list.shape[0])
#         rand_int = random.randrange(_hidden_list.shape[0])
#         # print(rand_int)
#         if rand_int in seen_rands:
#             continue
#         seen_rands.add(rand_int)
#         hidden_state = _hidden_list[rand_int]
#         hidden_state = hidden_state.detach().numpy()
#
#         # print(hidden_state)
#         # use minimize() with gradient based method
#         # Gradient descent basically consists in taking small steps in the direction of the gradient,
#         # that is the direction of the steepest descent.
#         """
#             one of the problems of the simple gradient descent algorithms, is that it tends to oscillate across a valley,
#             each time following the direction of the gradient, that makes it cross the valley. The conjugate gradient solves
#             this problem by adding a friction term: each step depends on the two last values of the gradient and sharp
#             turns are reduced.
#         """
#         res = minimize(loss_func, hidden_state, method="CG", tol=1e-8)
#         if res.success:
#             _fixed_point_list.append(res)
#
#     return _fixed_point_list


# def pca_hs(states_list):
#     states_list = StandardScaler().fit_transform(states_list)
#     pca = decomposition.PCA(n_components=2)
#     principal_components = pca.fit_transform(states_list)
#     principal_df = pd.DataFrame(data=principal_components,
#                                 columns=['principal component 1', 'principal component 2'])
#     target = ['0' for _ in range(len(states_list))]
#     principal_df['target'] = target
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(1, 1, 1)
#     ax.set_xlabel('Principal Component 1', fontsize=15)
#     ax.set_ylabel('Principal Component 2', fontsize=15)
#     ax.set_title('2 component PCA', fontsize=20)
#     targets = ['0']
#     colors = ['r']
#     for target, color in zip(targets, colors):
#         indicesToKeep = principal_df['target'] == target
#         ax.scatter(principal_df.loc[indicesToKeep, 'principal component 1']
#                    , principal_df.loc[indicesToKeep, 'principal component 2']
#                    , c=color
#                    , s=50)
#     ax.legend(target)
#     ax.grid()
#     plt.show()
#     print('\n ################## Explained Variance ################## \n')
#     print(pca.explained_variance_ratio_)


def pca_hs_2(states_list):
    states_list_scaled = StandardScaler().fit_transform(states_list)
    features = states_list_scaled.T
    cov_matrix = np.cov(features)
    values, vectors = np.linalg.eig(cov_matrix)
    explained_variances = []
    acc_explained_variances = []
    _sum = 0
    for i in range(len(values)):
        explained_variances.append(values[i] / np.sum(values))
        _sum += values[i] / np.sum(values)
        acc_explained_variances.append(_sum)

    print(np.sum(explained_variances), 'n', explained_variances)
    projected_1 = states_list_scaled.dot(vectors.T[0])
    projected_2 = states_list_scaled.dot(vectors.T[1])
    res = pd.DataFrame(projected_1, columns=['PC1'])
    res['PC2'] = projected_2
    res['Y'] = 0
    res.head()
    x = [(i + 1) for i in range(len(values))]
    plt.plot(x, acc_explained_variances)
    plt.show()


"""
    consider data pairs {(x1, y1), . . . , (xm, ym)} (7). We
    then define DMD in terms of the n × m data matrices
    X􏰀 := [x1 ··· xm], Y􏰀 := [y1 ··· ym].
    
    with x_{k} = z_{k−1} and y_{k} = z_{k} - we can consider
    z_{k-1) = h_{t-1} and z_{k} = h_{t}
    For a dataset given by (7), define the operator
    A := 􏰀YX+,    where X+ is the pseudoinverse of X.
    The dynamic mode decomposition of the pair (X,Y) is given by
    the eigendecomposition of A. That is, the DMD modes and eigenvalues
    are the eigenvectors and eigenvalues of A.
"""


def dmd_alg(x_matrix, y_matrix):
    # Compute the (reduced) SVD of X, writing X = UΣV .
    u, s, vh = np.linalg.svd(x_matrix, full_matrices=False)
    print('dmd alg:')
    print('u: ', u)
    print('s: ', s)
    print('vh: ', vh)

    # we define A􏰀~ := U^(*) Y V Σ^(-1)

    # conjugate transpose U*
    u_c = np.conjugate(u)
    u_ct = np.transpose(u_c)
    # V
    vct = np.conjugate(vh)
    v = np.transpose(vct)
    # Σ^(-1)
    s = np.diag(s)

    s_inv = np.linalg.inv(s)

    # U^* Y
    u_ct_y = np.matmul(u_ct, y_matrix)
    # V Σ^(-1)
    v_s_inv = np.matmul(v, s_inv)
    # A􏰀~ = U^(*) Y V Σ^(-1)
    A_wave = np.matmul(u_ct_y, v_s_inv)
    eigenvalues, norm_eigenvectors = np.linalg.eig(A_wave)
    return A_wave, eigenvalues, norm_eigenvectors




