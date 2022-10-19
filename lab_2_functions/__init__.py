import numpy as np
import math
from scipy.special import erf, erfinv


# интегральная функция нормального распределения через функцию ошибки
# erf = 2/sqrt(pi) integral [e^(-t^2)] dt for  0 to x
# F = 1/sqrt(2pi) integral [e^(-z^2/2)] dz for  0 to x
def Phi(x):
    return 0.5 * (1 + erf(x / np.sqrt(2)))


# обратная интегральная функция нормального распределения через обратную функцию ошибки
def invPhi(x):
    return np.sqrt(2) * erfinv(2 * x - 1)


def calculate_Mahalanobis_distance(Ml, Mj, cor_matrix_B):
    diference_M = Ml - Mj
    diference_M_T = np.transpose(diference_M)
    inv_B = np.linalg.inv(cor_matrix_B)
    result = three_matrix_multiplier(diference_M_T, inv_B, diference_M)
    return result


def calculate_Lambda_tilda(P_omega_0, P_omega_1, C_matrix):
    lambda_ = float((P_omega_0 * (C_matrix[0][1] - C_matrix[0][0])) / (P_omega_1 * (C_matrix[1][0] - C_matrix[1][1])))
    return math.log(lambda_)


# считаются теоретические вероятности ошибочной классификации
def calculate_p_error(P_omega_0, P_omega_1, M1, M2, cor_matrix_B, C_matrix):
    Mahalnobis_distance = calculate_Mahalanobis_distance(M1, M2, cor_matrix_B)
    lambda_tilda = calculate_Lambda_tilda(P_omega_0, P_omega_1, C_matrix)
    p = np.zeros(2, float)
    # p01
    p[0] = 1 - Phi((lambda_tilda + 0.5 * (Mahalnobis_distance)) / np.sqrt(Mahalnobis_distance))
    # p10
    p[1] = Phi((lambda_tilda - 0.5 * (Mahalnobis_distance)) / np.sqrt(Mahalnobis_distance))
    return p


# d_lj = x^T * (inv(Bj)-inv(Bl)) * x  + 2*(Ml^T * inv(Bl) - Mj^T * inv(Bj))*x +
# [ln(det(Bl)/det(Bj)) +2ln (P_omega_l/P_omega_j) - Ml^T*inv(Bl)*Ml + Mj^T*inv(Bj)*Mj]
def calculate_Bayes_border(Ml, Mj, Bl, Bj, P_omega_l, P_omega_j, assumed_x1_value):
    d, e = calculate_coefficients_d_and_e(Ml, Mj, Bl, Bj)
    f = calculate_coefficient_f(Ml, Mj, Bl, Bj, P_omega_l, P_omega_j)

    if not np.array_equiv(Bl, Bj):
        a = calculate_coefficient_a(Bl, Bj)
        b = calculate_coefficient_b(Bl, Bj) * assumed_x1_value + d
        c = calculate_coefficient_c(Bl, Bj) * (assumed_x1_value ** 2) + e * assumed_x1_value + f
        discriminant = (b ** 2) - 4 * a * c
        square_root_disc = np.sqrt(discriminant)
        return (-b - square_root_disc) / (2 * a), (-b + square_root_disc) / (2 * a)
    else:
        return -1 / d * (e * assumed_x1_value + f)


def three_matrix_multiplier(mat_1, mat_2, mat_3):
    tmp_result = np.matmul(mat_1, mat_2)
    return np.matmul(tmp_result, mat_3)


def calculate_coefficient_a(Bl, Bj):
    Bl_inv = np.linalg.inv(Bl)
    Bj_inv = np.linalg.inv(Bj)
    return Bj_inv[0][0] - Bl_inv[0][0]


def calculate_coefficient_b(Bl, Bj):
    Bl_inv = np.linalg.inv(Bl)
    Bj_inv = np.linalg.inv(Bj)
    return (Bj_inv[1][0] - Bl_inv[1][0]) + (Bj_inv[0][1] - Bl_inv[0][1])


def calculate_coefficient_c(Bl, Bj):
    Bl_inv = np.linalg.inv(Bl)
    Bj_inv = np.linalg.inv(Bj)
    return Bj_inv[1][1] - Bl_inv[1][1]


def calculate_coefficients_d_and_e(Ml, Mj, Bl, Bj):
    Bj_inv = np.linalg.inv(Bj)
    Bl_inv = np.linalg.inv(Bl)
    Mj_T = np.transpose(Mj)
    Ml_T = np.transpose(Ml)
    Ml_T_x_Bl = np.matmul(Ml_T, Bl_inv)
    Mj_T_x_Bj = np.matmul(Mj_T, Bj_inv)
    result = 2 * (Ml_T_x_Bl - Mj_T_x_Bj)
    return result[0][0], result[0][1]


def calculate_coefficient_f(Ml, Mj, Bl, Bj, P_omega_l, P_omega_j):
    Bl_det = np.linalg.det(Bl)
    Bj_det = np.linalg.det(Bj)

    sum_part1 = np.log(Bl_det / Bj_det)
    sum_part2 = 2 * np.log(P_omega_l / P_omega_j)

    Bl_inv = np.linalg.inv(Bl)
    Bj_inv = np.linalg.inv(Bj)

    Ml_T = np.transpose(Ml)
    Mj_T = np.transpose(Mj)

    Ml_T_x_Bl_x_Ml = three_matrix_multiplier(Ml_T, Bl_inv, Ml)
    Mj_T_x_Bj_x_Mj = three_matrix_multiplier(Mj_T, Bj_inv, Mj)

    return sum_part1 + sum_part2 - Ml_T_x_Bl_x_Ml + Mj_T_x_Bj_x_Mj


def calculate_Bayes_border_data(Ml, Mj, Bl, Bj, P_omega_l, P_omega_j, left_border, right_border, step):
    elements_number = int((right_border - left_border) / step)
    if not np.array_equiv(Bl, Bj):
        result = np.zeros((3, int(abs((right_border - left_border) / step))), float)
        for i in range(0, elements_number, 1):
            x1_assumed_value = left_border + i * step
            result[0][i], result[1][i] = calculate_Bayes_border(Ml, Mj, Bl, Bj, P_omega_l, P_omega_j, x1_assumed_value)
            result[2][i] = x1_assumed_value
        return result
    else:
        result = np.zeros((2, int(abs((right_border - left_border) / step))), float)
        for i in range(0, elements_number, 1):
            x1_assumed_value = left_border + i * step
            result[0][i] = calculate_Bayes_border(Ml, Mj, Bl, Bj, P_omega_l, P_omega_j, x1_assumed_value)
            result[1][i] = x1_assumed_value
        return result


def calculate_min_max_border_data(Ml, Mj, B, left_border, right_border, step):
    elements_number = int((right_border - left_border) / step)
    result = np.zeros((2, int(abs((right_border - left_border) / step))), float)
    for i in range(0, elements_number, 1):
        x1_assumed_value = left_border + i * step
        result[0][i] = calculate_Bayes_border(Ml, Mj, B, B, 0.5, 0.5, x1_assumed_value)
        result[1][i] = x1_assumed_value
    return result



def calculate_NP_border(Ml, Mj, B, p0, x1_assumed_value):
    mahalanobis_distance = calculate_Mahalanobis_distance(Ml, Mj, B)
    lambda_tilda = -0.5 * mahalanobis_distance + np.sqrt(mahalanobis_distance) * invPhi(1 - p0)

    dif_M = Ml -Mj
    sum_M = Ml+ Mj
    inv_B = np.linalg.inv(B)
    dif_M_T_x_inv_B = np.matmul(np.transpose(dif_M), inv_B)
    b =  -0.5 * three_matrix_multiplier(np.transpose(sum_M), inv_B, dif_M )

    x0 = float((-lambda_tilda -b[0][0] - x1_assumed_value * dif_M_T_x_inv_B[0][1])/dif_M_T_x_inv_B[0][0])
    return x0



def calculate_NP_border_data(Ml, Mj, B, p0, left_border, right_border, step):
    elements_number = int((right_border - left_border) / step)
    result = np.zeros((2, int(abs((right_border - left_border) / step))), float)
    for i in range(0, elements_number, 1):
        x1_assumed_value = left_border + i * step
        result[0][i] = calculate_NP_border(Ml, Mj, B, p0, x1_assumed_value)
        result[1][i] = x1_assumed_value
    return result


def NP_decider(X_vector, Ml, Mj, B, p0):
    mahalanobis_distance = calculate_Mahalanobis_distance(Ml, Mj,B)
    m_distance_Ml = calculate_Mahalanobis_distance(X_vector, Ml, B)
    m_distance_Mj = calculate_Mahalanobis_distance(X_vector, Mj, B)
    condition_density_relation_power_of_exp = 0.5 * m_distance_Ml - 0.5 * m_distance_Mj
    lambda_tilda = -0.5 * mahalanobis_distance + np.sqrt(mahalanobis_distance) * invPhi(1 - p0)
    if condition_density_relation_power_of_exp > lambda_tilda:
        return 1
    else:
        return 0


def define_function_for_Bayes(X_vector, P_omega, M, B):
    ln_sqrt_det_B =  math.log(np.sqrt(np.linalg.det(B)))
    dif_x_M = X_vector - M
    inv_B = np.linalg.inv(B)
    return math.log(P_omega) - ln_sqrt_det_B -0.5*three_matrix_multiplier(np.transpose(dif_x_M), inv_B, dif_x_M)[0][0]


def Bayes_decider(X_vector, Ml, Mj, Bl, Bj, P_omega_l, P_omega_j):
    def_func_0 = define_function_for_Bayes(X_vector, P_omega_l, Ml, Bl)
    def_func_1 = define_function_for_Bayes(X_vector, P_omega_j, Mj, Bj)
    if def_func_0 > def_func_1 :
        # print("def_0_value:", def_func_0, "def_1_value:", def_func_1)
        return 0
    else:
        # print("def_0_value:", def_func_0, "def_1_value:", def_func_1)
        return 1


def min_max_decider(X_vector, Ml, Mj, Bl, Bj):
    return Bayes_decider(X_vector, Ml, Mj, Bl, Bj, 0.5, 0.5)


def calculate_experimental_probability(p, size_of_selection_N):
    if p != 0.0:
        return np.sqrt((1 - p) / (size_of_selection_N * p))
    else:
        return 1


def calculate_size_of_selection(E_required, Ml, Mj, Bl, Bj, test_data, N, P_omega_l, P_omega_j):
    i = 0
    p = 0.0
    sum = 0
    E = 1
    while E >= E_required and i < N - 1:
        tmp_data = np.array([[test_data[0][i]], [test_data[1][i]]])
        sum += Bayes_decider(tmp_data, Ml, Mj, Bl, Bj, P_omega_l, P_omega_j)
        i += 1
        p = float(sum / i)
        E = calculate_experimental_probability(p, N)
    return i


def calculate_E_experimental(Ml, Mj, Bl, Bj, test_data, N, P_omega_l, P_omega_j):
    i = 0
    p = 0.0
    sum = 0
    E = 1
    while i < N - 1:
        tmp_data = np.array([[test_data[0][i]], [test_data[1][i]]])
        sum += Bayes_decider(tmp_data, Ml, Mj, Bl, Bj, P_omega_l, P_omega_j)
        i += 1
        p = float(sum / i)
    E = calculate_experimental_probability(p, N)
    return E