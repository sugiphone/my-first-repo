# ライブラリインポート
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn import tree
import graphviz

# その他設定
pd.set_option('max_columns', 35)
pd.set_option('max_rows', 600)
sns.set_style('darkgrid')

# データ読み込み
breastcancer = pd.read_csv('../input/data.csv')

# 欠損値確認のためのDF作成
def missing_values_table(df):
	mis_val = df.isnull().sum()
	mis_val_percent = 100 * df.isnull().sum() / len(df)
	mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
	mis_val_table_ren_columns = mis_val_table.rename(
		columns= {0:'Missing Values', 1:'% of total values'})
	return mis_val_table_ren_columns

missing_values_table(breastcancer)