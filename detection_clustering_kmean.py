#!/usr/bin/python

#tune: n_clusters

import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class DetectionClustering(object):
    def __init__(self, detections, plot=False, out_dir='out'): 
        self.clusters = {}
        for name, pos in detections.iteritems():    #4 labels, so run 4 times
            if name == "station":
                print "station"
                self.kmeans = KMeans(n_clusters = 4, random_state = 0, n_init = 100)
                midpoints = self.compute(name, pos, plot=plot, out_dir=out_dir)
                if midpoints: self.clusters[name] = midpoints
            if name == "supplier":
                print "supplier"
                self.kmeans = KMeans(n_clusters = 2, random_state = 0, n_init = 100)
                midpoints = self.compute(name, pos, plot=plot, out_dir=out_dir)
                if midpoints: self.clusters[name] = midpoints
            if name == "breaker":
                print "breaker"
                self.kmeans = KMeans(n_clusters = 2, random_state = 0, n_init = 100)
                midpoints = self.compute(name, pos, plot=plot, out_dir=out_dir)
                if midpoints: self.clusters[name] = midpoints
            if name == "elevator":
                print "elevator"
                self.kmeans = KMeans(n_clusters = 2, random_state = 0, n_init = 100)
                midpoints = self.compute(name, pos, plot=plot, out_dir=out_dir)
                if midpoints: self.clusters[name] = midpoints

    def compute(self, name, array, plot=False, out_dir='out'):    # Compute Kmeans
        array_limited = []    #limit the cluster area
        for element in array:
            if element[0] > 0:    #[0] means x coordinates
                array_limited.append(element)
        X = np.array(array_limited)
        km = self.kmeans.fit(X)
        labels = km.labels_
        # Compute midpoints
        label_types = set(labels)
        if -1 in label_types: label_types.remove(-1)
        result = [] 
        centroids = km.cluster_centers_
        result.append(centroids.tolist())

        if plot: 
            # Black is removed and used for noise instead.
            unique_labels = set(labels)
            colors = [plt.cm.Spectral(each)
                      for each in np.linspace(0, 1, len(unique_labels))]

            for k, col in zip(unique_labels, colors):
                if k == -1: col = [0, 0, 0, 1] # Black is used for noise.

                class_member_mask = (labels == k)

                xy = X[class_member_mask & core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=14)

                mean = xy.mean(axis=0)
                plt.plot(mean[0], mean[1], 'k+', markeredgewidth=2, markersize=10)

                xy = X[class_member_mask & ~core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=6)

                output_file = out_dir + '/' + name + '.png'
                print 'Writing graph to {}...'.format(output_file)
                plt.title("Clustering result of class '{}'".format(name))
                plt.savefig(output_file)

        return result

if __name__ == '__main__':
    # Custom Positive Integer type
    def positive_int(val):
        i = int(val)
        if i <= 0:
            raise argparse.ArgumentTypeError('invalid positive_int value: {}'.format(val))
        return i

    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, default='~/.ros/detections_raw.db')
    parser.add_argument('-o', '--output_file', type=str, default='detections_kmean.db')
    parser.add_argument('-p', '--plot', type=bool, default=False)
    parser.add_argument('-d', '--output_directory', type=str, default='out',
                        help='Directory to where graphs are saved when --plot is set.')

    args, _ = parser.parse_known_args()

    if args.plot and not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    print 'Reading from {}...'.format(args.input_file)
    with open(os.path.expanduser(args.input_file), 'r') as infile:
        detections = yaml.load(infile)

    dc = DetectionClustering(detections, args.plot, args.output_directory)

    print 'Writing to {}...'.format(args.output_file)
    with open(os.path.expanduser(args.output_file), 'w') as outfile:
        yaml.dump(dc.clusters, outfile)
