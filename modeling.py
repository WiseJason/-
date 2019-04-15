#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : WiseJason
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit, KFold
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
# from lightgbm.sklearn import LGBMClassifier


warnings.filterwarnings('ignore')



pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


data_model=pd.read_excel("process_code.xlsx")
# print(len(data_model.columns))
# print(data_model.head())


"""
相关系数矩阵图
"""
# data_model_corr=data_model.iloc[:,:-1].corr()
#
# # plt.subplots(figsize=(9, 9)) # 设置画面大小
# sns.heatmap(data_model_corr,  vmax=1, square=True, cmap="rainbow")
# plt.show()
"""
单变量分析,去掉相关系数很大的且IV值较低的
"""
f = open('iv_col.txt','rb')
iv_dict = pickle.loads(f.read())

list_col=list(data_model.columns)
list_col.remove("target")
for i  in list(range(len(list_col))):
    for j in list(range(len(list_col))):
        if i>j:
            try:
                col_1=list_col[i]
                col_2=list_col[j]
                corr=data_model[col_1].corr(data_model[col_2])
                if corr>=0.6:
                    if iv_dict[col_1]>=iv_dict[col_2]:
                        del data_model[col_2]
                    else:
                        del data_model[col_1]
            except:
                 pass
#vif
for col in data_model.columns:
    if col !="target":
        list_col = list(data_model.columns)
        list_col.remove(col)
        y=data_model[col]
        x=data_model[list_col]
        model = LinearRegression()
        model.fit(x,y)
        r=model.score(x,y)
        vif=1/(1-r)
        if vif >10:
            del data_model[col]



#数据集分割
x_list=list(data_model.columns)
X=data_model[x_list]
del X['target']
y=data_model[['target']]
smo = SMOTE(n_jobs=-1)
x_smo, y_smo = smo.fit_sample(X, y)

X_train, X_test, y_train, y_test = train_test_split(x_smo, y_smo, test_size=0.33, random_state=42)

#决策树
# from sklearn import tree
# # 数据标准化
# from sklearn.preprocessing import StandardScaler
# print("**********决策树***************")
# ss_X = StandardScaler()
# ss_y = StandardScaler()
# X_train = ss_X.fit_transform(X_train)
# X_test = ss_X.transform(X_test)
# param = {"splitter":["random"],
#             "max_depth":range(10,len(x_list)),
#          }
#
# model_tree = DecisionTreeClassifier()
# gsearch = GridSearchCV(model_tree, param_grid = param,scoring ='accuracy', cv=5)
# gsearch.fit(X_train, y_train)
#
# print('最佳参数：',gsearch.best_params_)
# print('训练集的最佳分数：', gsearch.best_score_)
# print('测试集的最佳分数：', gsearch.score(X_test, y_test))
#人工神经网络

# #模型评估
# from sklearn.metrics import accuracy_score, roc_auc_score
#
#
# def model_metrics(clf, X_train, X_test, y_train, y_test):
#     # 预测
#     y_train_pred = clf.predict(X_train)
#     y_test_pred = clf.predict(X_test)
#
#     y_train_proba = clf.predict_proba(X_train)[:, 1]
#     y_test_proba = clf.predict_proba(X_test)[:, 1]
#
#     # 准确率
#     print('[准确率]', end=' ')
#     print('训练集：', '%.4f' % accuracy_score(y_train, y_train_pred), end=' ')
#     print('测试集：', '%.4f' % accuracy_score(y_test, y_test_pred))
#
#     # auc取值：用roc_auc_score或auc
#     print('[auc值]', end=' ')
#     print('训练集：', '%.4f' % roc_auc_score(y_train, y_train_proba), end=' ')
#     print('测试集：', '%.4f' % roc_auc_score(y_test, y_test_proba))



print("**********逻辑回归***************")
# #逻辑回归
lr = LogisticRegression()
param = {'C': [1e-3,0.01,0.1,1,10,100,1e3], 'penalty':['l1', 'l2']}

gsearch = GridSearchCV(lr, param_grid = param,scoring ='roc_auc', cv=5)
gsearch.fit(X_train, y_train)

print('最佳参数：',gsearch.best_params_)
print('训练集的最佳分数：', gsearch.best_score_)
print('测试集的最佳分数：', gsearch.score(X_test, y_test))


#
# lr = LogisticRegression(C = 0.1, penalty = 'l1')
# lr.fit(X_train, y_train)
# model_metrics(lr, X_train, X_test, y_train, y_test)


#SVM模型

# # 1) 线性SVM
# svm_linear = svm.SVC(kernel = 'linear', probability=True)
# param = {'C':[0.01,0.1,1]}
# gsearch = GridSearchCV(svm_linear, param_grid = param,scoring ='roc_auc', cv=5)
# gsearch.fit(X_train, y_train)
#
# print("线性SVM")
# print('最佳参数：',gsearch.best_params_)
# print('训练集的最佳分数：', gsearch.best_score_)
# print('测试集的最佳分数：', gsearch.score(X_test, y_test))
# #
# # svm_linear = svm.SVC(C = 0.01, kernel = 'linear', probability=True)
# # svm_linear.fit(X_train, y_train)
# # model_metrics(svm_linear, X_train, X_test, y_train, y_test)
#
#
#
# # 2) 多项式SVM
# svm_poly = svm.SVC(kernel = 'poly', probability=True)
# param = {'C':[0.01,0.1,1]}
# gsearch = GridSearchCV(svm_poly, param_grid = param,scoring ='roc_auc', cv=5)
# gsearch.fit(X_train, y_train)
# print("*******")
# print("多项式SVM")
# print('最佳参数：',gsearch.best_params_)
# print('训练集的最佳分数：', gsearch.best_score_)
# print('测试集的最佳分数：', gsearch.score(X_test, y_test))
#
#
# # svm_poly =  svm.SVC(C = 0.01, kernel = 'poly', probability=True)
# # svm_poly.fit(X_train, y_train)
# # model_metrics(svm_poly, X_train, X_test, y_train, y_test)
#
#
# # 3) 高斯SVM
# svm_rbf = svm.SVC(probability=True)
# param = {'gamma':[0.01,0.1,1,10],
#          'C':[0.01,0.1,1]}
# gsearch = GridSearchCV(svm_poly, param_grid = param,scoring ='roc_auc', cv=5)
# gsearch.fit(X_train, y_train)
#
# print("******")
# print("高斯核")
# print('最佳参数：',gsearch.best_params_)
# print('训练集的最佳分数：', gsearch.best_score_)
# print('测试集的最佳分数：', gsearch.score(X_test, y_test))
#
#
# # svm_rbf =  svm.SVC(gamma = 0.01, C =0.01 , probability=True)
# # svm_rbf.fit(X_train, y_train)
# # model_metrics(svm_rbf, X_train, X_test, y_train, y_test)
#
# # 4) sigmoid - SVM
# svm_sigmoid = svm.SVC(kernel = 'sigmoid',probability=True)
# param = {'C':[0.01,0.1,1]}
# gsearch = GridSearchCV(svm_sigmoid, param_grid = param,scoring ='roc_auc', cv=5)
# gsearch.fit(X_train, y_train)
# print("******")
# print("sigmoid")
# print('最佳参数：',gsearch.best_params_)
# print('训练集的最佳分数：', gsearch.best_score_)
# print('测试集的最佳分数：', gsearch.score(X_test, y_test))

# svm_sigmoid =  svm.SVC(C = 0.01, kernel = 'sigmoid',probability=True)
# svm_sigmoid.fit(X_train, y_train)
# model_metrics(svm_sigmoid, X_train, X_test, y_train, y_test)




#决策树
# param = {'max_depth':range(3,14,2), 'min_samples_split':range(100,801,200)}
# gsearch = GridSearchCV(DecisionTreeClassifier(max_depth=8,min_samples_split=300,min_samples_leaf=20, max_features='sqrt' ,random_state =2333),
#                        param_grid = param,scoring ='roc_auc', cv=5)
#
# gsearch.fit(X_train, y_train)
# # gsearch.grid_scores_,
# gsearch.best_params_, gsearch.best_score_
#
#
# param = {'min_samples_split':range(50,1000,100), 'min_samples_leaf':range(60,101,10)}
# gsearch = GridSearchCV(DecisionTreeClassifier(max_depth=11,min_samples_split=100,min_samples_leaf=20, max_features='sqrt',random_state =2333),
#                        param_grid = param,scoring ='roc_auc', cv=5)
#
# gsearch.fit(X_train, y_train)
# # gsearch.grid_scores_,
# gsearch.best_params_, gsearch.best_score_
#
#
# param = {'max_features':range(7,20,2)}
# gsearch = GridSearchCV(DecisionTreeClassifier(max_depth=11,min_samples_split=550,min_samples_leaf=80, max_features='sqrt',random_state =2333),
#                        param_grid = param,scoring ='roc_auc', cv=5)
#
# gsearch.fit(X_train, y_train)
# # gsearch.grid_scores_,
# gsearch.best_params_, gsearch.best_score_
#
#
# dt = DecisionTreeClassifier(max_depth=11,min_samples_split=550,min_samples_leaf=80,max_features=19,random_state =2333)
# dt.fit(X_train, y_train)
# model_metrics(dt, X_train, X_test, y_train, y_test)


#xgboost
#
# xgb0 = XGBClassifier()
# xgb0.fit(X_train, y_train)
#
# model_metrics(xgb0, X_train, X_test, y_train, y_test)
#
# param_test = {'n_estimators':range(20,200,20)}
# gsearch = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=5,
#                                                   min_child_weight=1, gamma=0, subsample=0.8,
#                                                   colsample_bytree=0.8, objective= 'binary:logistic',
#                                                   nthread=4,scale_pos_weight=1, seed=27),
#                         param_grid = param_test, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#
# gsearch.fit(X_train, y_train)
# # gsearch.grid_scores_,
# gsearch.best_params_, gsearch.best_score_
#
#
# param_test = {'max_depth':range(3,10,2), 'min_child_weight':range(1,12,2)}
#
# gsearch = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=20, max_depth=5,
#                                                   min_child_weight=1, gamma=0, subsample=0.8,
#                                                   colsample_bytree=0.8, objective= 'binary:logistic',
#                                                   nthread=4,scale_pos_weight=1, seed=27),
#                         param_grid = param_test, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#
# gsearch.fit(X_train, y_train)
# # gsearch.grid_scores_,
# gsearch.best_params_, gsearch.best_score_
#
#
#
# param_test = {'max_depth':[3,4,5], 'min_child_weight':[3,4,5]}
#
# gsearch = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=20, max_depth=5,
#                                                   min_child_weight=1, gamma=0, subsample=0.8,
#                                                   colsample_bytree=0.8, objective= 'binary:logistic',
#                                                   nthread=4,scale_pos_weight=1, seed=27),
#                         param_grid = param_test, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#
# gsearch.fit(X_train, y_train)
# # gsearch.grid_scores_,
# gsearch.best_params_, gsearch.best_score_
#
#
# param_test = {'gamma':[i/10 for i in range(1,6)]}
#
# gsearch = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=20, max_depth=5,
#                                                   min_child_weight=5, gamma=0, subsample=0.8,
#                                                   colsample_bytree=0.8, objective= 'binary:logistic',
#                                                   nthread=4,scale_pos_weight=1, seed=27),
#                         param_grid = param_test, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#
# gsearch.fit(X_train, y_train)
# # gsearch.grid_scores_,
# gsearch.best_params_, gsearch.best_score_
#
# param_test = {'subsample':[i/10 for i in range(5,10)], 'colsample_bytree':[i/10 for i in range(5,10)]}
#
# gsearch = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=20, max_depth=5,
#                                                   min_child_weight=5, gamma=0.4, subsample=0.8,
#                                                   colsample_bytree=0.8, objective= 'binary:logistic',
#                                                   nthread=4,scale_pos_weight=1, seed=27),
#                         param_grid = param_test, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#
# gsearch.fit(X_train, y_train)
# # gsearch.grid_scores_,
# gsearch.best_params_, gsearch.best_score_
#
#
# param_test = { 'subsample':[i/100 for i in range(85,101,5)], 'colsample_bytree':[i/100 for i in range(85,101,5)]}
#
# gsearch = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=20, max_depth=5,
#                                                   min_child_weight=5, gamma=0.4, subsample=0.8,
#                                                   colsample_bytree=0.8, objective= 'binary:logistic',
#                                                   nthread=4,scale_pos_weight=1, seed=27),
#                         param_grid = param_test, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#
# gsearch.fit(X_train, y_train)
# # gsearch.grid_scores_,
# gsearch.best_params_, gsearch.best_score_
#
#
# #  'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
# param_test = {'reg_alpha':[1e-5, 1e-2, 0.1, 0, 1, 100]}
#
# gsearch = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=20, max_depth=5,
#                                                   min_child_weight=5, gamma=0.4, subsample=0.95,
#                                                   colsample_bytree=0.9, objective= 'binary:logistic',
#                                                   nthread=4,scale_pos_weight=1, seed=27),
#                         param_grid = param_test, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#
# gsearch.fit(X_train, y_train)
# # gsearch.grid_scores_,
# gsearch.best_params_, gsearch.best_score_
#
#
#
#
# #  'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
# param_test = {'reg_alpha':[1e-5, 1e-2, 0.1, 0, 1, 100]}
#
# gsearch = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=20, max_depth=5,
#                                                   min_child_weight=5, gamma=0.4, subsample=0.95,
#                                                   colsample_bytree=0.9, objective= 'binary:logistic',
#                                                   nthread=4,scale_pos_weight=1, seed=27),
#                         param_grid = param_test, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#
# gsearch.fit(X_train, y_train)
# # gsearch.grid_scores_,
# gsearch.best_params_, gsearch.best_score_
#
#
#
# param_test = {'n_estimators':range(20,200,20)}
#
# gsearch = GridSearchCV(estimator = XGBClassifier(learning_rate =0.01, n_estimators=60, max_depth=3,
#                                                   min_child_weight=5, gamma=0.4, subsample=0.5,
#                                                   colsample_bytree=0.9, reg_alpha=1, objective= 'binary:logistic',
#                                                   nthread=4,scale_pos_weight=1, seed=27),
#                         param_grid = param_test, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#
# gsearch.fit(X_train, y_train)
# # gsearch.grid_scores_,
# gsearch.best_params_, gsearch.best_score_





















# # for n in range(1,20):
# #     smo = SMOTE(random_state=None,k_neighbors=3,n_jobs=-1)
# #     x_smo, y_smo = smo.fit_sample(x, y)
# #
# #     # print(model.score(x,y))
# #
# cv = StratifiedKFold(n_splits=10, shuffle=True)
#
# def estimate(estimator, name='estimator'):
#     auc = cross_val_score(estimator, x_smo,y_smo, scoring='roc_auc', cv=cv).mean()
#     accuracy = cross_val_score(estimator, x_smo,y_smo, scoring='accuracy', cv=cv).mean()
#     recall = cross_val_score(estimator, x_smo,y_smo, scoring='recall', cv=cv).mean()
#     return auc,accuracy,recall
# #
#
#
#
#
#
# # auc=estimate(rf, 'LogisticRegression')[0]
# # accuracy=estimate(rf, 'LogisticRegression')[1]
# # recall=estimate(rf, 'LogisticRegression')[2]
#
#
#
