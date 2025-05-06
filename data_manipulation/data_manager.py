import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

class DataManager(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.scaled_data = None
        self.pca_data = None
        self.kmeans_labels = None

    def load_data(self):
        if os.path.exists(self.data_path):
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded from {self.data_path}")
            return self.data
        else:
            raise FileNotFoundError(f"The file {self.data_path} does not exist.")

    def add_new_features(self,dataFrame):
        dataFrame['Sex'] = LabelEncoder().fit_transform(dataFrame['Sex'])
        dataFrame['BMI'] = dataFrame['Weight'] / (dataFrame['Height'] / 100) ** 2
        return dataFrame
    def tartget_transform(self,dataFrame):
        dataFrame['Calories'] = np.log1p(dataFrame['Calories'])
        return dataFrame
    
    