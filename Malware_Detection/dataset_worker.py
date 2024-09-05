import collections
import pickle
import os

import pandas
import numpy as np
import tensorflow as tf

from setting import database_path, processed_data_path, non_embedding_variable, embedding_variable, train_dataset_size

class dataset_worker:

    def __init__(self):
        self.dataset = pandas.read_csv(database_path, sep='|')

    def dataset_loader(self, load_from_raw=False):
        self.non_embedding_data, self.embedding_data = [], []
        if load_from_raw or (not os.path.isfile(processed_data_path[0])) or (not os.path.isfile(processed_data_path[1])):
            
            for row in range(self.dataset.shape[0]):
                new_non_embedding = []
                new_embedding = []
                for column_name in self.dataset.columns:
                    if column_name in non_embedding_variable:
                        new_non_embedding.append(float(self.dataset.at[row, column_name]))
                    elif column_name in embedding_variable:
                        new_embedding.append(float(self.dataset.at[row, column_name]))
                self.non_embedding_data.append(new_non_embedding.copy())
                self.embedding_data.append(new_embedding.copy())

            self.non_embedding_data = self.log(self.non_embedding_data)
            self.non_embedding_data = self.z_score(self.non_embedding_data)   

            self.embedding_data = self.log(self.embedding_data)
            self.embedding_data = self.z_score(self.embedding_data)
            
            with open(processed_data_path[0], 'wb') as fp:
                pickle.dump(self.non_embedding_data, fp)
            with open(processed_data_path[1], 'wb') as fp:
                pickle.dump(self.embedding_data, fp)

        else:
            with open(processed_data_path[0], 'rb') as fp:
                self.non_embedding_data = pickle.load(fp)
            with open(processed_data_path[1], 'rb') as fp:
                self.embedding_data = pickle.load(fp)

    def get_non_embedding_data(self):
        return self.non_embedding_data
    
    def get_embedding_data(self):
        return self.embedding_data
    
    def z_score(self, input):
        output = np.array(input)
        return (output - output.mean(axis=0, keepdims=True)) / output.std(axis=0, keepdims=True)
    
    def scaling(self, input):
        min_row = tf.reduce_min(input, axis=0)
        max_row = tf.reduce_max(input, axis=0)
        return (input - min_row) / (max_row - min_row)
    
    def log(self, input):
        output = np.array(input)
        return np.log(output+0.01)

    def dataset_analyser(self):
        count_table = collections.defaultdict(lambda: collections.defaultdict(int))
        non_embedding_data, embedding_data = [], []

        for row in range(self.dataset.shape[0]):
            new_non_embedding = []
            new_embedding = []
            for column_name in self.dataset.columns:
                count_table[column_name][self.dataset.at[row, column_name]] += 1
                if column_name in non_embedding_variable:
                    new_non_embedding.append(float(self.dataset.at[row, column_name]))
                elif column_name in embedding_variable:
                    new_embedding.append(float(self.dataset.at[row, column_name]))
            non_embedding_data.append(new_non_embedding.copy())
            embedding_data.append(new_embedding.copy())

        embedding_data = np.array(embedding_data)

        print("Original Dataset Characteristic")
        print(embedding_data.max(axis=0, keepdims=True))
        print(embedding_data.min(axis=0, keepdims=True))
        print(embedding_data.mean(axis=0, keepdims=True))
        print(embedding_data.std(axis=0, keepdims=True))
        
        embedding_data = self.log(embedding_data)

        print("Log Dataset Characteristic")
        print(embedding_data.max(axis=0, keepdims=True))
        print(embedding_data.min(axis=0, keepdims=True))
        print(embedding_data.mean(axis=0, keepdims=True))
        print(embedding_data.std(axis=0, keepdims=True))
        
        for column in count_table.keys():
            if len(count_table[column].keys()) < 100:
                print(column, len(count_table[column].keys()))
            
        print(self.dataset.columns)
        print(self.dataset.shape[0])
        print(count_table['legitimate'])
