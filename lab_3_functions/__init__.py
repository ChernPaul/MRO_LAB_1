import math
import numpy as np
from skimage.io import show, imshow
from matplotlib import pyplot as plt
from scipy.special import erf
import lab_2_functions

VALUE_ONE = 1
VALUE_ZERO = 0

LETTER_1 = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 0],
                     [0, 1, 0, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 0, 0, 0, 0, 1, 0]])

LETTER_2 = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 0, 0, 0, 1, 0, 0],
                     [0, 1, 1, 1, 1, 1, 0, 0, 0]])


def process_invert(p_value, src_matrix_value, p_probability_of_change):
    if p_value > p_probability_of_change:
        return src_matrix_value
    else:
        return 1 - src_matrix_value


def get_matrix_9x9(letter, p_probability_of_change):
    matrix_as_vector = letter.flatten()
    shape = np.shape(letter)
    random_vector = np.random.uniform(0, 1, shape[0] * shape[1])
    func = np.vectorize(process_invert)
    result = func(random_vector, matrix_as_vector, p_probability_of_change)
    return np.reshape(result, (shape[0], shape[1]))


def print_matrix_9x9(matrix):
    for i in range(0, 9, 1):
        tmp_str = []
        for j in range(0, 9, 1):
            tmp_str.append(matrix[i][j])
        print(tmp_str)


def generate_seed_data_for_classes(class_seed, selection_size, p_probability_of_change):
    shape = np.shape(class_seed)
    matrix_as_vector = class_seed.flatten()
    result = np.zeros((selection_size, shape[0] * shape[1]))

    func = np.vectorize(process_invert)
    for i in range(0, selection_size, 1):
        random_vector = np.random.uniform(0, 1, shape[0] * shape[1])
        result[i] = func(random_vector, matrix_as_vector, p_probability_of_change)
        # selection size strings AND 81 columns
    return result


def get_probability(dataset):
    ones_count = np.sum(dataset, axis=0)
    return ones_count / dataset.shape[0]

def get_threshold(
    aprior_0: float, aprior_1: float, prob_0: float, prob_1: float
) -> np.array:
    return np.log(aprior_1 / aprior_0) + np.sum(np.log((1 - prob_1) / (1 - prob_0)))

def phi_laplace(x) -> float:
    return (1 + math.erf(x / np.sqrt(2))) / 2

def get_errors(
    aprior_0: float, aprior_1: float, prob_0: float, prob_1: float
) -> list:
    w_1_0 = np.log(prob_1 * (1 - prob_0) / ((1 - prob_1) * prob_0))

    m_0 = np.sum(w_1_0 * prob_0)
    m_1 = np.sum(w_1_0 * prob_1)

    sigma_0 = np.sqrt(np.sum((w_1_0 ** 2) * prob_0 * (1 - prob_0)))
    sigma_1 = np.sqrt(np.sum((w_1_0 ** 2) * prob_1 * (1 - prob_1)))

    _lambda = -get_threshold(aprior_0, aprior_1, prob_0, prob_1)

    errors = [1 - lab_2_functions.Phi((_lambda - m_0) / sigma_0),
              lab_2_functions.Phi((_lambda - m_1) / sigma_1)]
    upper_bounds = [sigma_0 ** 2 / ((m_0 - _lambda) ** 2),
                    sigma_1 ** 2 / ((m_1 - _lambda) ** 2)]

    return errors, m_0, m_1, sigma_0, sigma_1

def calculate_array_of_condition_probabilities(array_of_vectors_seed_class):
    shape_of_array = np.shape(array_of_vectors_seed_class)
    shape_result_array = np.shape(array_of_vectors_seed_class[0])
    sum_of_vector_coordinates_values = np.zeros(shape_result_array)
    for i in range(0, shape_of_array[0], 1):
        sum_of_vector_coordinates_values += array_of_vectors_seed_class[i]

    return np.divide(sum_of_vector_coordinates_values, shape_of_array[0])


def calculate_w_lj_coefficients_array(array_of_condition_probabilities_omega_l,
                                      array_of_condition_probabilities_omega_j):
    # shape is (81)
    shape = np.shape(array_of_condition_probabilities_omega_l)
    result = np.zeros(shape)
    for i in range(0, shape[0], 1):
        result[i] = math.log(
            ((array_of_condition_probabilities_omega_l[i] / (1 - array_of_condition_probabilities_omega_l[i])) *
             ((1 - array_of_condition_probabilities_omega_j[i]) / array_of_condition_probabilities_omega_j[i]))
        )
    return result


def calculate_lambda_tilda(P_omega_l, P_omega_j, array_of_condition_probabilities_omega_l,
                           array_of_condition_probabilities_omega_j):
    # shape is (81)
    shape = np.shape(array_of_condition_probabilities_omega_l)
    pre_result = np.zeros(shape)
    for i in range(0, shape[0], 1):
        pre_result[i] = math.log((1 - array_of_condition_probabilities_omega_l[i]) /
                                 (1 - array_of_condition_probabilities_omega_j[i]))

    return math.log(P_omega_j / P_omega_l) + np.sum(pre_result)


def calculate_Lambda_tilda(X_vector, array_of_condition_probabilities_omega_l,
                           array_of_condition_probabilities_omega_j):
    array_wlj = calculate_w_lj_coefficients_array(array_of_condition_probabilities_omega_l,
                                                  array_of_condition_probabilities_omega_j)

    return np.sum(X_vector * array_wlj)


def binary_Bayes_classificator(X_vector, P_omega_l, P_omega_j, array_of_condition_probabilities_omega_l,
                               array_of_condition_probabilities_omega_j):
    Lamda_tilda = calculate_Lambda_tilda(X_vector, array_of_condition_probabilities_omega_l,
                                         array_of_condition_probabilities_omega_j)
    lambda_tilda = calculate_lambda_tilda(P_omega_l, P_omega_j, array_of_condition_probabilities_omega_l,
                                          array_of_condition_probabilities_omega_j)
    if Lamda_tilda >= lambda_tilda:
        return int(0)
    else:
        return int(1)


def classify_array_of_vectors(array_of_vectors, P_omega_l, P_omega_j, array_of_condition_probabilities_omega_l,
                              array_of_condition_probabilities_omega_j):
    shape = np.shape(array_of_vectors)
    result = np.zeros(shape[0], int)
    for i in range(0, shape[0], 1):
        result[i] = binary_Bayes_classificator(array_of_vectors[i], P_omega_l, P_omega_j,
                                               array_of_condition_probabilities_omega_l,
                                               array_of_condition_probabilities_omega_j)

    return result


def show_vector_picture(vector):
    matrix_view = np.reshape(vector, (9, 9))
    imshow(1 - matrix_view, cmap='gray')
    show()


def show_all_vectors_pictures(array_of_vectors, classification_vector, vector_name, show_wrong=True):
    shape = np.shape(classification_vector)
    if show_wrong:
        for i in range(0, shape[0], 1):
            if classification_vector[i] == 1:
                matrix_view = np.reshape(array_of_vectors[i], (9, 9))
                plt.title("Wrong classified vector " + vector_name + " index: " + str(i))
                imshow(1 - matrix_view, cmap='gray')
                show()
    else:
        for i in range(0, shape[0], 1):
            if classification_vector[i] == 0:
                matrix_view = np.reshape(array_of_vectors[i], (9, 9))
                plt.title("Right classified vector " + vector_name + " index: " + str(i))
                imshow(1 - matrix_view, cmap='gray')
                show()


def calculate_exp_error(classified_array):
    return float(np.sum(classified_array) / classified_array.size)


def calculate_exp_relative_error(p, size_of_selection_N):
    if p != 0:
        return np.sqrt((1 - p) / (size_of_selection_N * p))
    else:
        return 1


def calculate_common_relative_error(P_omega_1, P_omega_2, relative_error_1, relative_error_2):
    return P_omega_1 * relative_error_1 + P_omega_2 * relative_error_2


def calculate_binary_m(array_of_condition_probabilities_omega_0, array_of_condition_probabilities_omega_1):
    size = array_of_condition_probabilities_omega_0.size
    result_m0 = np.zeros(size)
    result_m1 = np.zeros(size)
    wlj_array = calculate_w_lj_coefficients_array(array_of_condition_probabilities_omega_1,
                                                  array_of_condition_probabilities_omega_0)
    for i in range(0, size, 1):
        result_m0[i] = wlj_array[i] * array_of_condition_probabilities_omega_0[i]

        result_m1[i] = wlj_array[i] * array_of_condition_probabilities_omega_1[i]

    return np.sum(result_m0), np.sum(result_m1)


def calculate_binary_SD(array_of_condition_probabilities_omega_0, array_of_condition_probabilities_omega_1):
    size = array_of_condition_probabilities_omega_0.size
    result_0 = np.zeros(size)
    result_1 = np.zeros(size)
    for i in range(0, size, 1):
        log_part = calculate_w_lj_coefficients_array(array_of_condition_probabilities_omega_1,
                                                     array_of_condition_probabilities_omega_0)[i]
        result_0[i] = np.power(log_part, 2) * array_of_condition_probabilities_omega_0[i] * (1 - array_of_condition_probabilities_omega_0[i])
        result_1[i] = np.power(log_part, 2) * array_of_condition_probabilities_omega_1[i] * (1 - array_of_condition_probabilities_omega_1[i])

    return np.sqrt(np.sum(result_0)), np.sqrt(np.sum(result_1))


def calculate_theoretical_errors(P_omega_0, P_omega_1, array_of_condition_probabilities_omega_0,
                                 array_of_condition_probabilities_omega_1):
    lambda_tilda = calculate_lambda_tilda(P_omega_0, P_omega_1, array_of_condition_probabilities_omega_0,
                                          array_of_condition_probabilities_omega_1)
    m0, m1 = calculate_binary_m(array_of_condition_probabilities_omega_0, array_of_condition_probabilities_omega_1)
    standard_deviation_0, standard_deviation_1 = calculate_binary_SD(array_of_condition_probabilities_omega_0,
                                                                     array_of_condition_probabilities_omega_1)
    p0 = 1 - lab_2_functions.Phi((lambda_tilda - m0) / standard_deviation_0)
    p1 = lab_2_functions.Phi((lambda_tilda - m1) / standard_deviation_1)
    result = np.array([p0, p1])
    return result


def calculate_theoretical_upper_bounds(P_omega_0, P_omega_1, array_of_condition_probabilities_omega_0,
                                       array_of_condition_probabilities_omega_1):
    m0, m1 = calculate_binary_m(array_of_condition_probabilities_omega_0, array_of_condition_probabilities_omega_1)
    standard_deviation_0, standard_deviation_1 = calculate_binary_SD(array_of_condition_probabilities_omega_0,
                                                                     array_of_condition_probabilities_omega_1)
    lambda_tilda = calculate_lambda_tilda(P_omega_0, P_omega_1, array_of_condition_probabilities_omega_0,
                                          array_of_condition_probabilities_omega_1)
    p0 = np.power(standard_deviation_0, 2) / np.power((m0 - lambda_tilda), 2)
    p1 = np.power(standard_deviation_1, 2) / np.power((m1 - lambda_tilda), 2)
    return p0, p1
