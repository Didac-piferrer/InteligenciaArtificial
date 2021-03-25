__authors__ = ['1492383', '1497551', '1491223']
__group__ = 'DL.15-DJ.17'

import numpy as np
from Kmeans import *
from KNN import *
from utils_data import read_dataset, visualize_k_means, visualize_retrieval
import matplotlib.pyplot as plt
import cv2
import random


""" ····················ANALISIS CUANTITATIVO························"""

"""---------------------FUNCTIONS SHAPE ACCURACY--------------------"""
def TestShapeAccuracy(train_images, train_labels, test_images, test_labels, neigh, percentageTrain):
    limitTrain = int(train_labels.shape[0]*percentageTrain/100)
    knn = KNN(train_images[:limitTrain], train_labels[:limitTrain])
    preds = knn.predict(test_images, neigh)
    percentage = Get_shape_accuracy(preds, test_labels)
    return percentage


def Get_shape_accuracy(preds, GroundTruth):
    equalNumber = np.count_nonzero(np.equal(preds, GroundTruth))
    percentage = 100*(equalNumber / preds.shape[0])
    return percentage

"""--------------------FUNCTIONS COLOR ACCURACY----------------------"""
def TestColorAccuracy(test_images, test_labels, K, optionsTest):
    colors = []
    for image in test_images:
        km = KMeans(image, K, optionsTest)
        km.fit()
        colors.append(get_colors(km.centroids))
    percentage = Get_color_accuracy(colors, test_labels)
    return percentage

def Get_color_accuracy(colors, GroundTruth):
    containColor = []
    for index, ground in enumerate(GroundTruth):
        matches = np.isin(ground, colors[index])
        matchesCount = np.count_nonzero(matches)
        containColor.append(matchesCount/len(ground))

    containColorNumber = np.sum(containColor)
    percentage = 100 * (containColorNumber / GroundTruth.shape[0])
    return percentage



""" ····················ANALISIS CUALITATIVO························"""

"""--------------------FUNCTIONS RETRIEVAL COLOR----------------------"""

def Get_Color_List(test_images, k_max):
    colors = []
    # indices = np.random.randint(0, len(test_images), size)

    for image in test_images:
        km = KMeans(image)
        check = km.find_bestK(k_max)
        if check == True:
            km.centroids = np.delete(km.centroids, -1, 0)
            km.fit()

        colors.append(get_colors(km.centroids))

    return colors


def Retrieval_by_color(etiquetas_color, colour, cantidad):
    index_list = []
    index_list_def = []
    for i, j in enumerate(etiquetas_color):
        if len(colour[0]) == 1:  # comprobación si se trata solo de un string
            if colour in j:  # Ya que no existen colores con una sola letra
                index_list.append(i)
        else:
            if all(c in j for c in colour):
                index_list.append(i)
    if index_list:
        if len(index_list) > cantidad:
            index_list_def = random.sample(index_list, k=cantidad)
        else:
            index_list_def = random.sample(index_list, k=len(index_list))
    return index_list_def


def Test_Retrieval_by_color(test_imgs, k_max, color, cantidad):
    info_list = []
    etiquetas_color = Get_Color_List(test_imgs, k_max)
    index = Retrieval_by_color(etiquetas_color, color, cantidad)
    check = []
    if index:
        for i in index:
            info_list.append(etiquetas_color[i].tolist())
            if len(color[0]) == 1:
                if color in test_color_labels[i]:  # Comprovamos con el GroundTrhuth
                    check.append(True)
                else:
                    check.append(False)
            else:
                if all(c in test_color_labels[i] for c in color):
                    check.append(True)
                else:
                    check.append(False)
        GroundTruth = test_color_labels[index].tolist()
        titol = "Retrieval_by_color    Query:" + str(color)
        visualize_retrieval(test_imgs[index], cantidad, info=GroundTruth, ok=check, title=titol)
        print(info_list)
    else:
        print('-----Image not found-----')

"""--------------------FUNCTIONS RETRIEVAL SHAPE----------------------"""


def Retrieval_by_shape(result_knn, forma, cantidad):
    index_list = []
    for x, y in enumerate(result_knn):
        if y == forma:
            index_list.append(x)
    def_index = random.sample(index_list, cantidad)
    # se puede devolver tambien las imagenes haciendo return test_images[def_index]
    return def_index


def Test_Retrieval_by_shape(test_images, neigh, forma, cantidad):
    check = []
    knn = KNN(train_imgs, train_class_labels)
    result_knn = knn.predict(test_images, neigh)
    index_image = Retrieval_by_shape(result_knn, forma, cantidad)
    for i in index_image:
        if result_knn[i] == test_class_labels[i]:
            check.append(True)
        else:
            check.append(False)
    GroundTruth = test_class_labels[index_image]
    titol = "Retrieval_by_shape   Query:" + str(forma)
    visualize_retrieval(test_images[index_image], cantidad, info=GroundTruth, ok=check, title=titol)


"""--------------------FUNCTION VISUALIZE KMEANS----------------------"""

def Test_Visualize_Kmeans(test_images, cantidad, max_k):
    lista_index_img = [i for i in range(test_imgs.shape[0])]
    index = random.sample(lista_index_img, k=cantidad)
    for image in test_images[index]:
        km = KMeans(image)
        check = km.find_bestK(max_k)
        K_resultant = km.K
        if check == True:
            km.centroids = np.delete(km.centroids, -1, 0)
            km.fit()

        etiquetas_color = get_colors(km.centroids)
        print("--------Imatge representada amb K=" + str(K_resultant) + "-----------")
        print("Colors trobats pel kmeans:" + str(etiquetas_color.tolist()))
        visualize_k_means(km, [80, 60, 3])


def Test_total_RBC():
    total_colores = ['Orange', 'Brown', 'Green', 'Blue', 'Grey', 'White']
    for col in total_colores:
        Test_Retrieval_by_color(test_imgs, k_max, col, cantidad)

"""--------------------------------------------MAIN-------------------------------------------------------"""


if __name__ == '__main__':
    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json', w=60, h=80) #Modificar directorio y demnsiones para diferente tamaño de imagenes y resolucion

    # List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))


    # -------------------ANALISIS CUALITATIVO-----------------------------------
    """print("#--------------------TEST VISUALIZE KMEANS--------------------------")
    start_time = time.time()
    max_k = 10
    cantidad = 6
    Test_Visualize_Kmeans(test_imgs, cantidad, max_k)
    print("--- %s seconds ---" % (time.time() - start_time))"""


    """print("#--------------------TEST RETRIEVAL BY COLOR TOTAL----------------------")
    start_time = time.time()
    i = 0
    k_max = 3
    cantidad = 12
    num_iter = 1
    while (i < num_iter):
        Test_total_RBC()
        i = i+1
    print("--- %s seconds ---" % (time.time() - start_time))"""


    """print("--------------------TEST RETRIEVAL BY COLOR----------------------")
    # ['Red','Orange','Brown','Yellow','Green','Blue','Purple','Pink','Black','Grey','White'])
    start_time = time.time()
    k_max = 10
    color = 'Grey'
    cantidad = 12
    Test_Retrieval_by_color(test_imgs, k_max, color, cantidad)
    print("--- %s seconds ---" % (time.time() - start_time))"""


    """print("#--------------------TEST RETRIEVAL BY SHAPE---------------------")
    start_time = time.time()
    neigh = 3
    forma = "Shorts"
    cantidad = 12
    Test_Retrieval_by_shape(test_imgs, neigh, forma, cantidad)
    print("--- %s seconds ---" % (time.time() - start_time))"""

    # ----Test_detect_number_of_colors
    """colours = ['Red', 'Orange', 'Brown', 'Yellow', 'Green', 'Blue', 'Purple', 'Pink', 'Black', 'Grey', 'White']
    k_max = 3
    count_GT = 0
    count_kmeans = 0
    etiquetas_kmeans = Get_Color_List(test_imgs, k_max)
    resultat = []
    print('CANTIDAD DE PRENDAS POR COLOR: \n\n')
    print('---Resultados para k_max = ' + str(k_max) + ' ---\n')
    for c in colours:
        for x, y in zip(etiquetas_kmeans, test_color_labels):
            if c in x:
                count_kmeans = count_kmeans + 1
            if c in y:
                count_GT = count_GT + 1
        print('Color: ' + str(c) + ' ---------- GroundTruth: ' + str(count_GT) + '------- Kmeans: ' + str(count_kmeans))
        count_GT = 0
        count_kmeans = 0"""

    # ---------------------ANALISIS CUANTITATIVO--------------------------------------
    """Ejecucion test Color accuracy"""
    """print('#--------------------TEST COLOR ACCURACY-----------------------')
    start_time = time.time()
    K = 5
    optionsTest = {'tolerance': 0}
    resTestColor = TestColorAccuracy(test_imgs, test_color_labels, K, optionsTest)
    print(resTestColor)
    print("--- %s seconds ---" % (time.time() - start_time))"""


    """Ejecucion test shape accuracy"""
    """print('#--------------------TEST SHAPE ACCURACY----------------------')
    start_time = time.time()
    k = 5
    trainPrecentage = 100
    resTestShape = TestShapeAccuracy(train_imgs, train_class_labels, test_imgs, test_class_labels, k, trainPrecentage)
    print(resTestShape)
    print("--- %s seconds ---" % (time.time() - start_time))"""