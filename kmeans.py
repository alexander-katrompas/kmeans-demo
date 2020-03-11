#!/usr/bin/python3

############################################
#
# Author: Alex Katrompas
#
# A simple demonstration-quality k-means
# implementation for academic investigation.
#
############################################

import matplotlib.pyplot as plt
import numpy as np
from os import path
import pandas as pd
import random as rd
import sys
import warnings

# for setting k to reasonable
# value for this application
MINK = 2
MAXK = 10

# reasonable max iterations for kmeans for this application
ITER = 500

def write_files(output):
    f1 = open("output.txt", "w")
    f2 = open("output.csv", "w")
    clusters = len(output)
    for cluster in range(1, clusters + 1):
        length = len(output[cluster])
        for i in range(length):
            f1.write(str(int(output[cluster][i][0])) + " " + str(int(output[cluster][i][1])) + " " + str(cluster) + "\n")
            f2.write(str(int(output[cluster][i][0])) + "," + str(int(output[cluster][i][1])) + "," + str(cluster) + "\n")
    f1.close()
    f2.close()


def plot_clusters(dataset, centroids, output, k):
    plt.scatter(dataset[:, 0], dataset[:, 1], c='black', label='unclustered data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Plot of data points')
    plt.show()

    color = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
    labels = ['cluster1', 'cluster2', 'cluster3', 'cluster4', 'cluster5', 'cluster6', 'cluster7', 'cluster8', 'cluster9', 'cluster10']
    for i in range(k):
        plt.scatter(output[i + 1][:, 0], output[i + 1][:, 1], c=color[i], label=labels[i])
    plt.scatter(centroids[0, :], centroids[1, :], s=300, c='yellow', label='centroids')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


def process_command_line():
    # set-up vars to be "empty"
    k = 0
    filename = ""
    dataset = pd.DataFrame({'A': []})
    plot = False
    
    argc = len(sys.argv)
    if (argc == 3 or argc == 4) and \
        (sys.argv[1].isdigit() and path.exists(sys.argv[2]) and sys.argv[2][-4:] == ".csv"):

        # process k
        # sys.argv[1] is a string of digits, convert it and bound it
        k = int(sys.argv[1])
        if k > MAXK or k < MINK: k = MINK
        
        # process file
        # sys.argv[2] is a file that exists and is a potential csv file
        filename = sys.argv[2]
        try:
            dataset = pd.read_csv(filename)
        except:
            pass
        
        # process optional display plot
        if argc == 4 and sys.argv[3] == 'p':
            plot = True

    return dataset, filename, k, plot


def kmeans(data, k):
    rows = data.shape[0]
    cols = data.shape[1]

    print("k =", k, "| rows =", rows, "| cols =", cols, "| iterations =", ITER)

    centroids = np.array([]).reshape(cols, 0)
    for i in range(k):
        centroids = np.c_[centroids, data[rd.randint(0, rows-1)]]
    
    output = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for j in range(ITER):
            e_distance = np.array([]).reshape(rows, 0)
            for i in range(k):
                temp_dist = np.sum((data-centroids[:, i]) ** 2, axis=1)
                e_distance = np.c_[e_distance, temp_dist]
            c = np.argmin(e_distance, axis=1) + 1

            temp = {}
            for i in range(k):
                temp[i + 1] = np.array([]).reshape(2, 0)
            for i in range(rows):
                temp[c[i]] = np.c_[temp[c[i]], data[i]]

            for i in range(k):
                temp[i + 1] = temp[i + 1].T

            for i in range(k):
                centroids[:, i] = np.mean(temp[i + 1], axis=0)

            output = temp

    return centroids, output


def main():
    dataset, filename, k, plot = process_command_line()
    
    if not dataset.empty and k:
        print("file =", filename)
        dataset = dataset.to_numpy()
        centroids, output = kmeans(dataset, k)
        
        write_files(output)
        if plot: plot_clusters(dataset, centroids, output, k)
    else:
        print("ERROR")
        print("  Usage: kmeans k input.csv [p]")
        print("  Enter a k value between 2 and 12.")
        print("  Enter valid csv file in the current directory.")
        print("  Enter optional flag p flag for plots.\n")

print()
main()
print()
