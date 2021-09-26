import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.combine import SMOTETomek
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv('C:/Users/Raisul Zulfikar/Desktop/Credit Card/Clean_dataset.csv')
print(dataset.head())

