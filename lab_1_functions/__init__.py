import numpy as np
SAMPLE_SIZE_N = 200


def generate_vector_X(A, M, n, N):
    left_border = 0
    right_border = 1
    m = (right_border + left_border) / 2
    number_of_realizations = 50
    Sn = np.zeros((n, N))
    for i in range(0, number_of_realizations, 1):
        Sn += np.random.uniform(left_border, right_border, (n, N)) - m
    standard_deviation = (right_border - left_border) / np.sqrt(12)
    E = Sn / (standard_deviation * np.sqrt(number_of_realizations))
    X = np.matmul(A, E) + np.reshape(M, (2, 1)) * np.ones((1, N))
    return X


def calculate_matrix_A(B):
    matrix_A = np.zeros((2, 2))
    matrix_A[0][0] = np.sqrt(B[0][0])
    matrix_A[0][1] = 0
    matrix_A[1][0] = B[0][1] / np.sqrt(B[0][0])
    matrix_A[1][1] = np.sqrt(B[1][1] - (B[0][1] ** 2) / B[0][0])
    return matrix_A


def calculate_mathematical_expectation_M(x):
    M = np.sum(x, axis=1) / SAMPLE_SIZE_N
    return M


def get_B_correlation_matrix_for_vector(x):
    M = calculate_mathematical_expectation_M(x)
    # M shape is (1, 2)
    B = np.zeros((2, 2))
    for i in range(0, SAMPLE_SIZE_N, 1):
        # sum for i xi * xi ^t  where x[:, i] = [ x, y ]^t  shape = (number of columns, number of rows) x.shape = (1,2)
        tmp = np.reshape(x[:, i], (2, 1))
        B += (np.matmul(tmp, np.transpose(tmp)))
    B /= SAMPLE_SIZE_N
    B -= np.matmul(np.reshape(M, (2, 1)), np.transpose(np.reshape(M, (2, 1))))
    return B