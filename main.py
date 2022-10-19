import matplotlib.pyplot as plt
import numpy as np
from skimage.io import show

import lab_1_functions
import lab_2_functions
import lab_3_functions
from skimage.io import imshow
TASK_VARIANT = 20

M1 = np.array([[1], [-1]])
M2 = np.array([[-2], [-2]])
M3 = np.array([[-1], [1]])

B1 = np.array([[0.5, 0.0],
               [0.0, 0.5]])
B2 = np.array([[0.4, 0.1],
               [0.1, 0.6]])
B3 = np.array([[0.6, -0.2],
               [-0.2, 0.6]])

NUMBER_OF_VECTOR_DIMENSIONS = 2
SAMPLE_SIZE_N = 200
PROBABILITY_HALF_OF_ONE = 0.5
PROBABILITY_ONE_OF_THREE = float(1 / 3)

C_MATRIX_OF_FINE = np.array([[0.0, 1.0],
                             [1.0, 0.0]])
P0_VALUE = 0.05

PROBABILITY_CLASS_P = 0.5
PROBABILITY_CLASS_B = 0.5

if __name__ == '__main__':

    A = lab_1_functions.calculate_matrix_A(B1)
    vector_1 = lab_1_functions.generate_vector_X(A, M1, NUMBER_OF_VECTOR_DIMENSIONS, SAMPLE_SIZE_N)
    vector_2 = lab_1_functions.generate_vector_X(A, M2, NUMBER_OF_VECTOR_DIMENSIONS, SAMPLE_SIZE_N)

    fig = plt.figure(figsize=(10, 10))
    plt.title("Generated data for 2 classes equal B")
    plt.plot(vector_1[0], vector_1[1], 'r+')
    plt.plot(vector_2[0], vector_2[1], 'b+')
    show()

    A1 = lab_1_functions.calculate_matrix_A(B1)
    A2 = lab_1_functions.calculate_matrix_A(B2)
    A3 = lab_1_functions.calculate_matrix_A(B3)

    vector_3 = lab_1_functions.generate_vector_X(A1, M1, NUMBER_OF_VECTOR_DIMENSIONS, SAMPLE_SIZE_N)
    vector_4 = lab_1_functions.generate_vector_X(A2, M2, NUMBER_OF_VECTOR_DIMENSIONS, SAMPLE_SIZE_N)
    vector_5 = lab_1_functions.generate_vector_X(A3, M3, NUMBER_OF_VECTOR_DIMENSIONS, SAMPLE_SIZE_N)

    fig1 = plt.figure(figsize=(10, 10))
    plt.title("Generated data for 3 classes unequal B")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.plot(vector_3[0], vector_3[1], 'r+')
    plt.plot(vector_4[0], vector_4[1], 'g+')
    plt.plot(vector_5[0], vector_5[1], 'b+')
    show()

    """
    B_vector_1 = get_B_correlation_matrix_for_vector(vector_1)
    B_vector_2 = get_B_correlation_matrix_for_vector(vector_2)
    B_vector_3 = get_B_correlation_matrix_for_vector(vector_3)
    B_vector_4 = get_B_correlation_matrix_for_vector(vector_4)
    B_vector_5 = get_B_correlation_matrix_for_vector(vector_5)

    print()
    print("Mathematical expectation M1: ", M1)
    print("Mathematical expectation vector_1: ", np.around(calculate_mathematical_expectation_M(vector_1), 4))
    print()
    print("Mathematical expectation M2: ", M2)
    print("Mathematical expectation vector_2: ", np.around(calculate_mathematical_expectation_M(vector_2), 4))
    print()
    print("Mathematical expectation M1: ", M1)
    print("Mathematical expectation vector_3: ", np.around(calculate_mathematical_expectation_M(vector_3), 4))
    print()
    print("Mathematical expectation M2: ", M2)
    print("Mathematical expectation vector_4: ", np.around(calculate_mathematical_expectation_M(vector_4), 4))
    print()
    print("Mathematical expectation M3: ", M3)
    print("Mathematical expectation vector_5: ", np.around(calculate_mathematical_expectation_M(vector_5), 4))
    print('\n')

    print("Correlation B1: \n", B1)
    print("Correlation B vector_1: \n", np.around(B_vector_1, 4))
    print()
    print("Correlation B1: \n", B1)
    print("Correlation B vector_2: \n", np.around(B_vector_2, 4))
    print()
    print("Correlation B1: \n", B1)
    print("Correlation B vector_3: \n", np.around(B_vector_3, 4))
    print()
    print("Correlation B2: \n", B2)
    print("Correlation B vector_4: \n", np.around(B_vector_4, 4))
    print()
    print("Correlation B3: \n", B3)
    print("Correlation B vector_5: \n", np.around(B_vector_5, 4))
     lab 3 code fragment =======================================================================================
    print("Относительная погрешность для класса П:", class_p_relative_error)
    print("Относительная погрешность классификации для класса Б:", class_b_relative_error)

    print("Общая относительная погрешность: ", lab_3_functions.calculate_common_relative_error(0.5, 0.5,
                                                                                               class_p_relative_error,
                                                                                               class_b_relative_error))
    
    """


    # Lab 2 main code started
    print("LAB 2 OUTPUT \n")
    lab_2_fig_bayes = plt.figure(figsize=(10, 10))
    plt.title("Bayes for equal B 2 classes")
    plt.plot(vector_1[0], vector_1[1], 'r+')
    plt.plot(vector_2[0], vector_2[1], 'b+')
    border = lab_2_functions.calculate_Bayes_border_data(M1, M2, B1, B1, PROBABILITY_HALF_OF_ONE, PROBABILITY_HALF_OF_ONE, -5, 2, 0.1)
    plt.plot(border[0], border[1], '-k')
    show()

    p = lab_2_functions.calculate_p_error(0.5, 0.5, M1, M2, B1, C_MATRIX_OF_FINE)
    print("Error probability p0 and p1: ", p)
    print("Sum error probability: ",  PROBABILITY_HALF_OF_ONE * p[0] + PROBABILITY_HALF_OF_ONE * p[1])


    lab_2_fig_min_max = plt.figure(figsize=(10, 10))
    plt.title("Min-Max for equal B 2 classes")
    plt.plot(vector_1[0], vector_1[1], 'r+')
    plt.plot(vector_2[0], vector_2[1], 'b+')
    border = lab_2_functions.calculate_min_max_border_data(M1, M2, B1, -5, 2, 0.1)
    plt.plot(border[0], border[1], '-k')
    show()

    lab_2_fig_min_max = plt.figure(figsize=(10, 10))
    plt.title("NP for equal B 2 classes")
    plt.plot(vector_1[0], vector_1[1], 'r+')
    plt.plot(vector_2[0], vector_2[1], 'b+')
    border = lab_2_functions.calculate_NP_border_data(M1, M2, B1, P0_VALUE, -5, 2, 0.1)
    plt.plot(border[0], border[1], '-k')
    show()



    lab_2_fig_bayes_3_classes = plt.figure(figsize=(10, 10))
    plt.title("Bayes for unequal B 3 classes")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.plot(vector_3[0], vector_3[1], 'r+')
    plt.plot(vector_4[0], vector_4[1], 'g+')
    plt.plot(vector_5[0], vector_5[1], 'b+')
    border_3_4 = lab_2_functions.calculate_Bayes_border_data(M1, M2, B1, B2, PROBABILITY_ONE_OF_THREE, PROBABILITY_ONE_OF_THREE, -5, -0.7, 0.1)
    border_3_5 = lab_2_functions.calculate_Bayes_border_data(M1, M3, B1, B3, PROBABILITY_ONE_OF_THREE, PROBABILITY_ONE_OF_THREE, -0.9, 5, 0.1)
    border_4_5 = lab_2_functions.calculate_Bayes_border_data(M2, M3, B2, B3, PROBABILITY_ONE_OF_THREE, PROBABILITY_ONE_OF_THREE, -0.9, 5, 0.1)

    plt.plot(border_3_4[0], border_3_4[2], '-y')
    plt.plot(border_3_5[0], border_3_5[2], '-c')
    plt.plot(border_4_5[0], border_4_5[2], '-k')

    plt.plot(border_3_4[1], border_3_4[2], '-y')
    plt.plot(border_3_5[1], border_3_5[2], '-c')
    plt.plot(border_4_5[1], border_4_5[2], '-k')
    show()
    size = 40000


    vector_test_data = lab_1_functions.generate_vector_X(A1, M1, NUMBER_OF_VECTOR_DIMENSIONS, size)


    size_of_selection = lab_2_functions.calculate_size_of_selection(0.05, M1, M2, B1, B2, vector_test_data,
                                                                    size,
                                                                    PROBABILITY_HALF_OF_ONE, PROBABILITY_HALF_OF_ONE)

    E_exp = lab_2_functions.calculate_E_experimental(M1, M2, B1, B2, vector_test_data,
                                                                    size,
                                                                    PROBABILITY_HALF_OF_ONE, PROBABILITY_HALF_OF_ONE)
    print("Size of selection is:", size_of_selection)
    print("E experimental is:", E_exp)


    print("Lab 3 OUTPUT")
    PROBABILITY_CLASS_P = 0.5
    PROBABILITY_CLASS_B = 0.5
    result_1 = lab_3_functions.get_matrix_9x9(lab_3_functions.LETTER_1, 0.3)
    result_2 = lab_3_functions.get_matrix_9x9(lab_3_functions.LETTER_2, 0.3)

    figure = plt.figure(figsize=(10, 10))
    plt.title("Letters")
    sub_figure_1 = figure.add_subplot(2, 2, 1)
    imshow(1 - lab_3_functions.LETTER_1, cmap='gray')
    sub_figure_1.set_title("Буква П")

    sub_figure_2 = figure.add_subplot(2, 2, 2)
    imshow(1 - result_1, cmap='gray')
    sub_figure_2.set_title("Буква П после обработки")

    sub_figure_3 = figure.add_subplot(2, 2, 3)
    imshow(1 - lab_3_functions.LETTER_2, cmap='gray')
    sub_figure_3.set_title("Буква Б")

    sub_figure_4 = figure.add_subplot(2, 2, 4)
    imshow(1 - result_2, cmap='gray')
    sub_figure_4.set_title("Буква Б после обработки")
    show()

    test_data_class_p = lab_3_functions.generate_seed_data_for_classes(lab_3_functions.LETTER_1, 200, 0.3)
    test_data_class_b = lab_3_functions.generate_seed_data_for_classes(lab_3_functions.LETTER_2, 200, 0.3)
    cond_prob_array_class_p = lab_3_functions.calculate_array_of_condition_probabilities(test_data_class_p)
    cond_prob_array_class_b = lab_3_functions.calculate_array_of_condition_probabilities(test_data_class_b)

    lab_3_fig_cond_prob = plt.figure(figsize=(10, 10))
    plt.title("Conditional probability for class P - red and B - blue")
    plt.plot(np.arange(81), cond_prob_array_class_p, '-r')
    plt.plot(np.arange(81), cond_prob_array_class_b, '-b')
    show()

    lab_3_fig_cond_prob_p = plt.figure(figsize=(10, 10))
    plt.title("Conditional probability for class P ")
    imshow(np.reshape(cond_prob_array_class_p, (9, 9)))
    show()
    lab_3_fig_cond_prob_p = plt.figure(figsize=(10, 10))
    plt.title("Conditional probability for class B ")
    imshow(np.reshape(cond_prob_array_class_b, (9, 9)))
    show()


    classified_array_class_p = lab_3_functions.classify_array_of_vectors(test_data_class_p, PROBABILITY_CLASS_P, PROBABILITY_CLASS_B,
                                                            cond_prob_array_class_p, cond_prob_array_class_b)
    print(classified_array_class_p)
    classified_array_class_b = lab_3_functions.classify_array_of_vectors(test_data_class_b, PROBABILITY_CLASS_B, PROBABILITY_CLASS_P,
                                                            cond_prob_array_class_b, cond_prob_array_class_p)
    print(classified_array_class_b)

    lab_3_functions.show_vector_picture(test_data_class_p[197])
    # lab_3_functions.show_all_vectors_pictures(test_data_class_p, res_class_p, "P")
    class_p_exp_error = lab_3_functions.calculate_exp_error(classified_array_class_p)
    class_b_exp_error = lab_3_functions.calculate_exp_error(classified_array_class_b)
    print("Экспериментальная ошибка классификации для класса П:", class_p_exp_error)
    print("Экспериментальная ошибка классификации для класса Б:", class_b_exp_error)
    class_p_relative_error = lab_3_functions.calculate_exp_relative_error(class_p_exp_error,
                                                                          classified_array_class_p.size)
    class_b_relative_error = lab_3_functions.calculate_exp_relative_error(class_b_exp_error,
                                                                          classified_array_class_b.size)
    theoretical_error = lab_3_functions.calculate_theoretical_errors(0.5, 0.5, cond_prob_array_class_p,
                                                               cond_prob_array_class_b)
    print("Теоритическая ошибка классификации для класса П:", theoretical_error[0])
    print("Теоритическая ошибка классификации для класса Б:", theoretical_error[1])
