from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
def loadTrain(train_file,test_file):
    sub_df=pd.read_csv(train_file)
    train_X_df, test_X_df, train_Y_df, test_Y_df=train_test_split(sub_df.iloc[:,1:],sub_df['label'],test_size=0.3,stratify=sub_df['label'])
    print(train_X_df.shape)
    print(train_Y_df.shape)
    print(test_X_df.shape)
    print(test_Y_df.shape)
    return train_X_df.as_matrix(),train_Y_df.as_matrix(),test_X_df.as_matrix(),test_Y_df.as_matrix()
if __name__=='__main__':
    train_X,train_Y,test_X,test_Y=loadTrain('./train.csv')
