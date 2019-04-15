#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : WiseJason
import pandas as pd
import numpy as np
import re
import time
import datetime
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import chi2
import warnings
from random import randint
import math
import pickle
warnings.filterwarnings('ignore')



pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# data_loginfo=pd.read_csv("PPD_LogInfo_3_1_Training_Set.csv")
# data_Master=pd.read_csv('PPD_Training_Master_GBK_3_1_Training_Set.csv',encoding='gbk')
# data_Master_part=data_Master[['Idx','target']]
# positive_num=data_Master[data_Master['target']==0].count().sum()#不违约的样本数量
# negtive_num=data_Master[data_Master['target']==1].count().sum()#不违约的样本数量
# positive_negtive=positive_num/negtive_num
# print(positive_negtive)
# data_loginfo_target=pd.merge(data_loginfo,data_Master_part,how='left')
# data__loginfo_target_count=pd.pivot_table(data_loginfo_target,index=['target','LogInfo1','LogInfo2'],values=['Idx'],aggfunc=len)
# data__loginfo_target_count_positive=data__loginfo_target_count.query('target==0').reset_index()
# data__loginfo_target_count_negtive=data__loginfo_target_count.query('target==1').reset_index()
# data__loginfo_target_count=pd.merge(data__loginfo_target_count_positive,data__loginfo_target_count_negtive,how='left',on=['LogInfo1','LogInfo2'])
# data__loginfo_target_count['positive_negtive']=np.divide(data__loginfo_target_count['Idx_x'],data__loginfo_target_count['Idx_y'])
# data__loginfo_target_count.to_excel('操作坏账比例.xlsx')
# total=data_loginfo['LogInfo1'].count().sum()
# print(data_loginfo['LogInfo1'].value_counts())
# print(data_loginfo['LogInfo1'].value_counts()/total)
# data_count=pd.pivot_table(data_loginfo,index=['LogInfo1','LogInfo2'],values=['Idx'],aggfunc=len)
# print(data_count)
# print(data_loginfo['LogInfo2'].value_counts())

# '''决策树
# data_loginfo=pd.read_csv("PPD_LogInfo_3_1_Training_Set.csv")
# data_loginfo_part=data_loginfo[['Idx','LogInfo1']]
# data_loginfo_part_count=pd.pivot_table(data_loginfo_part,index=['Idx'],values=['LogInfo1'],aggfunc=len)
# data_Master=pd.read_csv('PPD_Training_Master_GBK_3_1_Training_Set.csv',encoding='gbk')
# data_Master_part=data_Master[['Idx','target']]
# data_loginfo_part_target=pd.merge(data_loginfo_part,data_Master_part,how='left')
# data_loginfo_part_target=data_loginfo_part_target[['Idx','LogInfo1','target']]
# data_loginfo_part_target.set_index(['Idx','LogInfo1'],drop=True,inplace=True)
# # data_loginfo_part_target=data_loginfo_part_target.unstack()
# print(data_loginfo_part_target.head(100))
#
# print(len(list(data_loginfo_part['LogInfo1'].unique())))


def plt_null_col(data):
    train_isnull = data.isnull().mean()  # 缺失值的比例
    print(train_isnull[train_isnull > 0].sort_values(ascending=False))  # 查看有缺失的数据
    train_isnull = train_isnull[train_isnull > 0].sort_values(ascending=False)
    train_isnull.plot.bar(figsize=(12, 8), title='数据缺失情况')
    plt.show()


def plt_null_row(data):
    plt.figure(figsize=(12, 8))
    plt.scatter(np.arange(data.shape[0]),
                data.isnull().sum(axis=1).sort_values().values)  # 每行缺失的个数
    plt.show()


'''查看数据缺失情况

WeblogInfo_1、WeblogInfo_3、UserInfo_11、UserInfo_12、UserInfo_13、WeblogInfo_20缺失比例超过20%，暂且做删除处理
'''

data_Master = pd.read_csv('PPD_Training_Master_GBK_3_1_Training_Set.csv', encoding='gbk')

# plt_null_col(data_Master)
# plt_null_row(data_Master)
'''
删除缺失值较大的列
'''

data_Master.drop(labels=['WeblogInfo_1', 'WeblogInfo_3', 'UserInfo_11', 'UserInfo_12', 'UserInfo_13', 'WeblogInfo_20'],
                 axis=1, inplace=True)

# print(data_Master.isnull().mean()[data_Master.isnull().mean()> 0].sort_values(ascending=False))
'''
处理缺失值：
离散型的，把缺失作为一类，用-1表示
连续型的，用中位值替换
'''

columns_cat_data = pd.read_excel('魔镜杯字段类型说明文档.xlsx')


# print(list(columns_cat_data['变量类型'].unique()))
def fill_na(data):
    col_obj = []  # 存放不是数值的列名
    col_list_not = []  # 存放需填充空值列存在-1的列
    for column_name in data.columns:
        if data[column_name].isnull().any():  # 是否存在缺失值
            if data[column_name].dtype == 'object':
                col_obj.append(column_name)
            else:
                column_type = columns_cat_data[columns_cat_data['变量名称'] == column_name].reset_index()['变量类型'][0]
                # print(column_type)
                na_value = 0
                if column_type == 'Index':
                    pass
                elif column_type == 'Categorical':
                    if -1 not in list(data[column_name].unique()):
                        na_value = -1

                    else:
                        col_list_not.append(column_name)
                        print(column_name)
                        print(data[column_name].unique())
                        na_value = input("请输入替换空值的值")

                else:
                    na_value = data[column_name].quantile(0.5)
                # print(na_value)
                data[column_name].fillna(value=na_value, inplace=True)
    return data, col_obj, col_list_not


data_Master = fill_na(data_Master)[0]
col_obj = fill_na(data_Master)[1]
col_list_not = fill_na(data_Master)[2]
print(col_obj)
print(col_list_not)

print(data_Master['WeblogInfo_21'].unique())
data_Master['WeblogInfo_21'].fillna(value='E', inplace=True)
print(data_Master['WeblogInfo_19'].unique())
data_Master['WeblogInfo_19'].fillna(value='A', inplace=True)
'''
对比UserInfo_2、UserInfo_4大部分值都相同，所以互相填充下
'''

# print(data_Master.isnull().mean()[data_Master.isnull().mean()> 0].sort_values(ascending=False))


'''
处理地理信息
1.UserInfo_24那看上去像地址的备注栏，大多数都没有填写，故考虑删除
2.地址一般有三种情况,1.居住地2.户口地3.贷款申请定位地点
3.考虑用市传入模型，因为无法知晓每个地址的业务含义，
4、取缺失值最少的  UserInfo_4
5删除缺失的数据
'''

data_Master_loc = data_Master[
    ['UserInfo_2', 'UserInfo_4', 'UserInfo_7', 'UserInfo_8', 'UserInfo_19', 'UserInfo_20', 'UserInfo_24', 'target']]

data_Master_loc.to_csv('地理信息.csv', encoding='gbk')

columns = ['UserInfo_2', 'UserInfo_4', 'UserInfo_20']

# for col in columns:
#     print(data_Master_loc[col].value_counts())
#     print("***************")


data_Master.drop(labels=['UserInfo_2', 'UserInfo_7', 'UserInfo_8', 'UserInfo_19', 'UserInfo_20', 'UserInfo_24'], axis=1,
                 inplace=True)

data_Master.dropna(inplace=True)

'''
处理Loginfo表
1.每笔借款登录天数
2.借款首尾时间间隔
3.最后一次操作时间距申请成功的时间间隔
4.每笔借款进行了多少种操作

'''
data_loginfo = pd.read_csv("PPD_LogInfo_3_1_Training_Set.csv")

data_loginfo['LogInfo3'] = pd.to_datetime(data_loginfo['LogInfo3'], format='%Y-%m-%d')
data_loginfo['Listinginfo1'] = pd.to_datetime(data_loginfo['Listinginfo1'], format='%Y-%m-%d')

data_loginfo_max_min = pd.pivot_table(data_loginfo, index=['Idx'], values=['LogInfo3'], aggfunc=[np.max, np.min])

data_loginfo_head_tail = data_loginfo_max_min['amax'] - data_loginfo_max_min['amin']
data_loginfo_head_tail.rename(columns={'LogInfo3': "首尾间隔时间"}, inplace=True)

data_loginfo_drop_duplicates = data_loginfo.drop_duplicates(subset=['Idx', 'LogInfo3'], inplace=False)
data_loginfo_days = pd.pivot_table(data_loginfo_drop_duplicates, index=['Idx'], values=['LogInfo3'],
                                   aggfunc=len)  # 登录天数
data_loginfo_days.columns = ['登录天数']
# data_loginfo_days.rename(columns={'LogInfo3':'登录天数'},inplace=True)


data_loginfo_max = pd.pivot_table(data_loginfo, index=['Idx', 'Listinginfo1'], values=['LogInfo3'],
                                  aggfunc=np.max).reset_index('Listinginfo1')
data_loginfo_max['最近一次登录与成功时间间隔'] = data_loginfo_max['Listinginfo1'] - data_loginfo_max['LogInfo3']
data_loginfo_max['最近一次登录与成功时间间隔'] = data_loginfo_max[['最近一次登录与成功时间间隔']]

data_loginfo_drop_duplicates_2 = data_loginfo.drop_duplicates(subset=['Idx', 'LogInfo1', 'LogInfo2'], inplace=False)
data_loginfo_count = pd.pivot_table(data_loginfo_drop_duplicates_2, index=['Idx'], values=['LogInfo2'], aggfunc=len)
data_loginfo_count.rename(columns={'LogInfo2': '操作次数'}, inplace=True)

data_loginfo_clean = pd.merge(data_loginfo_head_tail, data_loginfo_max, how='left', left_index=True, right_index=True)
data_loginfo_clean = data_loginfo_clean[['首尾间隔时间', '最近一次登录与成功时间间隔']]
data_loginfo_clean = pd.merge(data_loginfo_clean, data_loginfo_days, how='left', left_index=True, right_index=True)
data_loginfo_clean = pd.merge(data_loginfo_clean, data_loginfo_count, how='left', left_index=True, right_index=True)

'''
处理Userupdate_Info表
1.修改信息天数
2，修改信息个数
3.最后修改与成交时间间隔
'''

data_Userupdate_Info = pd.read_csv('PPD_Userupdate_Info_3_1_Training_Set.csv')
data_Userupdate_Info['ListingInfo1'] = pd.to_datetime(data_Userupdate_Info['ListingInfo1'])
data_Userupdate_Info['UserupdateInfo2'] = pd.to_datetime(data_Userupdate_Info['UserupdateInfo2'])
data_Userupdate_Info['UserupdateInfo1'] = data_Userupdate_Info['UserupdateInfo1'].str.lower()

data_Userupdate_Info_drop = data_Userupdate_Info.drop_duplicates(subset=['Idx', 'UserupdateInfo2'])
data_Userupdate_Info_days = pd.pivot_table(data_Userupdate_Info_drop, index=['Idx'], values=['UserupdateInfo2'],
                                           aggfunc=len)
data_Userupdate_Info_days.rename(columns={'UserupdateInfo2': "修改信息天数"}, inplace=True)

data_Userupdate_Info_drop_2 = data_Userupdate_Info.drop_duplicates(subset=['Idx', 'UserupdateInfo1'])
data_Userupdate_Info_count = pd.pivot_table(data_Userupdate_Info_drop_2, index=['Idx'], values=['UserupdateInfo1'],
                                            aggfunc=len)
data_Userupdate_Info_count.rename(columns={'UserupdateInfo1': "修改信息个数"}, inplace=True)

data_Userupdate_Info_day_process = pd.pivot_table(data_Userupdate_Info, index=['Idx', 'ListingInfo1'],
                                                  values=['UserupdateInfo2'], aggfunc=np.max).reset_index(
    'ListingInfo1')

data_Userupdate_Info_day = pd.DataFrame(
    data_Userupdate_Info_day_process['ListingInfo1'] - data_Userupdate_Info_day_process['UserupdateInfo2'],
    columns=['最后修改与成交时间间隔'])

data_Userupdate_Info_clean = pd.merge(data_Userupdate_Info_days, data_Userupdate_Info_count, how='left',
                                      left_index=True, right_index=True)
data_Userupdate_Info_clean = pd.merge(data_Userupdate_Info_clean, data_Userupdate_Info_day, how='left', left_index=True,
                                      right_index=True)

'''
数据清洗
'''
data_Master['UserInfo_9'] = data_Master['UserInfo_9'].str.strip()

'''
数据合并
'''
data_Master.reset_index(drop=True, inplace=True)
data_Master.set_index('Idx', inplace=True)

data_train = pd.merge(data_Master, data_Userupdate_Info_clean, how='left', left_index=True, right_index=True)
data_train = pd.merge(data_train, data_loginfo_clean, how='left', left_index=True, right_index=True)
# print(data_train.head(5))
data_train.fillna(value=-1, inplace=True)


'''
卡方分箱
'''


def get_chiSquare_distuibution(dfree=1, cf=0.1):
    """
    计算卡方阈值
    :param dfree:自由度
    :param cf:显著性水平
    :return:
    """
    percents = [0.95, 0.90, 0.5, 0.1, 0.05, 0.025, 0.01, 0.005]
    df = pd.DataFrame(np.array([chi2.isf(percents, df=i) for i in range(1, 30)]))

    df.columns = percents
    df.index = df.index + 1
    # pd.set_option('precision', 3)
    chiSquare_threashhold = df.loc[dfree, cf]
    return chiSquare_threashhold


def calc_chiSquare(num_table):
    """
    计算卡方值
    :param num_table:
    :return:
    """
    num_table_cal_chi = num_table.copy()
    num_table_chi = np.array([])
    for i in range(num_table_cal_chi.shape[0] - 1):
        chi = (num_table_cal_chi[i, 2] - num_table_cal_chi[i, 4]) ** 2 / num_table_cal_chi[i, 4] + (
                    num_table_cal_chi[i, 3] - num_table_cal_chi[i, 5]) ** 2 / num_table_cal_chi[i, 5]
        +(num_table_cal_chi[i + 1, 2] - num_table_cal_chi[i + 1, 4]) ** 2 / num_table_cal_chi[i + 1, 4] + (
                    num_table_cal_chi[i + 1, 3] - num_table_cal_chi[i + 1, 5]) ** 2 / num_table_cal_chi[i + 1, 5]
        chi = round(chi, 2)
        num_table_chi = np.append(num_table_chi, chi)
    return num_table_chi


def ChiMerge(data, feature_colname, target_colname, max_bins, sample=None):
    # 函数调用入口
    """
    :param data:series
    :param feature_colname: 特征名称
    :param target_colname: 类别列名称
    :param max_bins: 最大分箱数
    :param sample: 是否需要重采样
    :return:
    """
    if sample != None:
        df = data.sample(n=sample)
    else:
        data
    num_table = get_num_table(data, feature_colname, target_colname)
    num_table_chiSquare = calc_chiSquare(num_table)
    chi_threshold = get_chiSquare_distuibution()
    # print(num_table_chiSquare)
    # print(num_table)
    # print(data.head(100))
    # print(feature_colname)
    min_chiSquare = min(num_table_chiSquare)

    while min_chiSquare < chi_threshold:
        if len(num_table_chiSquare) >1:
            min_index = np.where(num_table_chiSquare == min(num_table_chiSquare))[0][0]
            # print(min_index)
            num_table_chiSquare, num_table = merge_chiSquare(num_table, min_index)
            min_chiSquare = min(num_table_chiSquare)
        else:
            break


    while not monotonicity(num_table, feature_colname):
        min_index = np.where(num_table_chiSquare == min(num_table_chiSquare))[0][0]
        num_table_chiSquare, num_table = merge_chiSquare(num_table, min_index)
        min_chiSquare = min(num_table_chiSquare)
    bins = num_table_chiSquare.shape[0]

    while bins > max_bins:
        min_index = np.where(num_table_chiSquare == min(num_table_chiSquare))[0][0]

        num_table_chiSquare, num_table = merge_chiSquare(num_table, min_index)

        min_chiSquare = min(num_table_chiSquare)
        bins = num_table_chiSquare.shape[0]


    #
    while float(get_bad_rate(num_table, feature_colname)[0]) == 0.0:
        min_index = get_bad_rate(num_table, feature_colname)[2][0]-1
        # print(min_index)
        num_table_chiSquare, num_table = merge_chiSquare(num_table, min_index)


    while float(get_bad_rate(num_table, feature_colname)[1]) == 1.0:
        min_index = get_bad_rate(num_table, feature_colname)[3][0] - 1
        num_table_chiSquare, num_table = merge_chiSquare(num_table, min_index)

    result_table = pd.DataFrame(num_table, columns=[feature_colname, "total_num", "positive_num", "negtive_num",
                                                    "theory_positive_num", "theory_negtive_num"])
    result_table['chiSquare'] = 1.0
    result_table['chiSquare'][0] = float("inf")
    result_table['chiSquare'][1:] = num_table_chiSquare
    return result_table


def get_bad_rate(num_table, feature_colname):
    num_table_c=num_table.copy()
    num_table_c = pd.DataFrame(num_table_c, columns=[feature_colname, "total_num", "positive_num", "negtive_num",
                                                 "theory_positive_num", "theory_negtive_num"])
    num_table_c['bad_rate_cal'] = np.divide(num_table_c['positive_num'], num_table_c['total_num'])
    return min(num_table_c['bad_rate_cal']), max(num_table_c['bad_rate_cal']), np.where(
        num_table_c['bad_rate_cal'] == min(num_table_c['bad_rate_cal'])), np.where(
        num_table_c['bad_rate_cal'] == max(num_table_c['bad_rate_cal']))


def monotonicity(num_table, feature_colname):
    """
    是否连续
    """
    num_table_m=num_table.copy()
    num_table_m = pd.DataFrame(num_table_m, columns=[feature_colname, "total_num", "positive_num", "negtive_num",
                                                 "theory_positive_num", "theory_negtive_num"])
    num_table_m['bad_rate'] = num_table_m['positive_num'] / num_table_m['total_num']
    num_table_m['bad_rate_dif'] = num_table_m['bad_rate'].diff(1)
    num_table_m = num_table_m.fillna(0)
    if len(num_table_m[num_table_m['bad_rate_dif'] > 0]) == (num_table_m.shape[0] - 1) or len(
            num_table_m[num_table_m['bad_rate_dif'] < 0]) == (num_table_m.shape[0] - 1):
        return True
    else:
        return False


def merge_chiSquare(num_table, min_index):
    """
    更新卡方值
    :param num_table:
    :param min_index:
    :return:yey4
    """
    num_table_merge = num_table.copy()
    num_table_merge[min_index, 0] = num_table_merge[min_index + 1, 0]
    num_table_merge[min_index, 1] = num_table_merge[min_index + 1, 1] + num_table_merge[min_index, 1]
    num_table_merge[min_index, 2] = num_table_merge[min_index + 1, 2] + num_table_merge[min_index, 2]
    num_table_merge[min_index, 3] = num_table_merge[min_index + 1, 3] + num_table_merge[min_index, 3]
    num_table_merge[min_index, 4] = num_table_merge[min_index + 1, 4] + num_table_merge[min_index, 4]
    num_table_merge = np.delete(num_table_merge, (min_index + 1), axis=0)
    num_table_chiSquare_copy = calc_chiSquare(num_table_merge)
    return num_table_chiSquare_copy, num_table_merge


def get_num_table(data, feature_col, tag_col):
    """
    构建卡方矩阵
    :param data:
    :param feature_col:
    :param tag_col:
    :return:
    """
    total_num = data.groupby([feature_col])[tag_col].count()
    total_num = pd.DataFrame({'total_num': total_num}).round(0)
    positive_num = data.groupby([feature_col])[tag_col].sum()
    positive_num = pd.DataFrame({'positive_num': positive_num}).round(0)
    positive_rate = data[tag_col].sum() / data.shape[0]
    negtive_rate = 1 - positive_rate
    num_table = pd.merge(total_num, positive_num, left_index=True, right_index=True,
                         how='inner')
    num_table.reset_index(inplace=True)
    # print(num_table)
    num_table['negtive_num'] = (num_table['total_num'] - num_table['positive_num']).round(0)  # 统计需分箱变量每个值负样本数
    num_table['theory_positive_num'] = (num_table['total_num'] * positive_rate).round(2)
    num_table['theory_negtive_num'] = (num_table['total_num'] * negtive_rate).round(2)
    num_table = np.array(num_table)
    return num_table


'''
woe编码
'''


def code(data):
    data = data.copy()
    positive_sum = data["positive_num"].sum()
    negtive_sum = data['negtive_num'].sum()
    data["WOE"] = data.apply(
        lambda x: math.log(np.divide(x['positive_num'] / positive_sum, x['negtive_num'] / negtive_sum), math.e), axis=1)
    data["系数"] = np.subtract(data['positive_num'] / positive_sum, data['negtive_num'] / negtive_sum)
    data['IV'] = (np.multiply(data["WOE"], data['系数'])).apply(lambda x: round(x, 2))
    data["WOE"] = (data["WOE"].map(lambda x: abs(x))).apply(lambda x: round(x, 2))
    del data["系数"]
    iv = data["IV"].sum()
    return data, iv


'''
对离散变量和连续变量进行分箱编码
'''

columns_cat_data_numerical_list_all = list(columns_cat_data[columns_cat_data['变量类型'] == 'Numerical']['变量名称'])
columns_cat_data_Categorical_list_all = list(columns_cat_data[columns_cat_data['变量类型'] == 'Categorical']['变量名称'])

columns_cat_data_numerical_list = []  # 连续变量
columns_cat_data_Categorical_list = []  # 离散变量
for i in columns_cat_data_numerical_list_all:
    if i in data_train.columns:
        columns_cat_data_numerical_list.append(i)

for j in columns_cat_data_Categorical_list_all:
    if j in data_train.columns:
        columns_cat_data_Categorical_list.append(j)

'''
判断离散变量取值个数，如果小于等于5 则不分箱，大于5则跟连续变量一样分箱
'''
# columns_cat_data_Categorical_list_more=[]
# columns_cat_data_Categorical_list_less=[]
# for col in columns_cat_data_Categorical_list:
#     if len(data_train[col].unique())>5:
#         columns_cat_data_Categorical_list_more.append(col)
#     else:
#         columns_cat_data_Categorical_list_less.append(col)


'''
对离散变量用坏样本率进行编码,这种编码方式需要样本与总体的分布一致，至少样本均值是无偏估计
'''







def bad_rate_code(data, col, target):
    total = data.groupby([col])[target].count()
    bad = data.groupby([col])[target].sum()
    total = pd.DataFrame({'total': total})
    bad = pd.DataFrame({'bad': bad})
    bad_rate = pd.merge(total, bad, how='left', left_index=True, right_index=True)
    bad_rate['bad_rate'] = (bad_rate['bad'] / bad_rate['total']).round(2)
    bad_rate.sort_values(by='bad_rate', inplace=True)
    bad_rate.reset_index(inplace=True)
    # bad_rate = bad_rate[[col, 'bad_rate']]
    if max(bad_rate['bad_rate'])==1.0:
        index_max=list(bad_rate[bad_rate['bad_rate']==1.0].index)
        last_index=min(index_max)-1
        index_max.append(last_index)
        total_num=0
        bad_num=0
        for index in index_max:
            total_num+=bad_rate['total'][index]
            bad_num+=bad_rate['bad'][index]
        bad_rate_num=round(bad_num/total_num,2)
        bad_rate['bad_rate'].loc[last_index]=bad_rate_num
        for index in index_max:
            if index!=last_index:
                bad_rate['bad_rate'].loc[index]=bad_rate_num

    return bad_rate


# def merge_zero_bad_rate(data):
#     for i in range(data.shape[0]-1):
#         if data['bad_rate'][i]==0:
#             data['bad_rate'][i]=min(data[data['bad_rate'] !=0]['bad_rate'])
#     return data

names = locals()

data_train_code = pd.DataFrame()
columns_cat_data_Categorical_list_more = []
columns_cat_data_Categorical_list_less = []
columns_cat_data_numerical_list_copy=columns_cat_data_numerical_list.copy()
for col in columns_cat_data_numerical_list_copy:
    if len(data_train[col].unique())<=50:
        columns_cat_data_numerical_list.remove(col)
        columns_cat_data_Categorical_list.append(col)

"""
修改时间格式
"""
for col in columns_cat_data_numerical_list:
    try:
        data_train[col]=pd.to_datetime(data_train[col],format='%Y-%m-%d')
    except:
        pass

col_list=[]
iv_list=[]
for col in columns_cat_data_Categorical_list:
    if col != 'target':
        bad_rate = bad_rate_code(data_train[[col, 'target']], col, 'target')
        if min(bad_rate['bad_rate']) == 0:
            min_bad_rate = min(bad_rate[bad_rate['bad_rate'] != 0]['bad_rate'])
            bad_rate['bad_rate'] = bad_rate['bad_rate'].replace(0, min_bad_rate)


        data = pd.merge(data_train[[col, 'target']], bad_rate, how='left')
        # names['data_%s' %  col]=data[['bad_rate','target']]
        data_col = data[[ col,'bad_rate', 'target']]

        num = len(list(bad_rate['bad_rate'].unique()))
        if num > 5:
            data_col_copy=data_col.copy()
            columns_cat_data_Categorical_list_more.append(col)
            result_table = ChiMerge(data_col_copy,'bad_rate', 'target', 10)
            code_table = code(result_table)[0]
            col_iv = code(result_table)[1]

            if col_iv >= 0.02:
                code_table['bad_rate'] = code_table['bad_rate'].round(2)
                code_table = code_table[['bad_rate', 'WOE']]
                bad_rate = pd.merge(bad_rate, code_table, how='left')
                bad_rate.fillna(method='bfill',inplace=True)
                data_col=pd.merge(data_col,bad_rate,how='left',on=['bad_rate',col])
                data_col=data_col[['WOE']]
                data_col.columns=[col+"_code"]
                # print(data_col.head(20))
                data_train_code=pd.concat([data_train_code,data_col],axis=1,sort=False)
                iv_list.append(col_iv)
                col_list.append(col + "_code")
                # bad_rate.insert(1,col+"_code",bad_rate['WOE'])
                # bad_rate=bad_rate[[col+"_code"]]
                # print(len(bad_rate))
                # data_train_code=pd.merge(data_train_code,bad_rate,how='outer',left_index=True,right_index=True)
                # data_train_code=data_train_code.append(bad_rate)
        else:
            columns_cat_data_Categorical_list_less.append(col)
            data_result=pd.pivot_table(data_col,index=['bad_rate'],values=['target'],aggfunc=[np.sum,len])
            data_result['neg_num']=data_result['len']-data_result['sum']
            data_result=data_result[['neg_num','sum']]
            data_result.columns=['negtive_num','positive_num']
            data_result.reset_index(inplace=True)

            code_table=code(data_result)[0]
            col_iv=code(data_result)[1]
            if col_iv>=0.02:
                code_table['bad_rate']=code_table['bad_rate'].round(2)
                # print(code_table)
                code_table=code_table[['bad_rate','WOE']]
                bad_rate=pd.merge(bad_rate,code_table,how='left')
                bad_rate.fillna(method='bfill', inplace=True)
                data_col = pd.merge(data_col, bad_rate, how='left', on=['bad_rate', col])
                data_col = data_col[['WOE']]
                data_col.columns = [col + "_code"]
                data_train_code = pd.concat([data_train_code, data_col], axis=1, sort=False)
                iv_list.append(col_iv)
                col_list.append(col + "_code")

print(len(columns_cat_data_numerical_list))
#处理连续变量

def cut_Equal_frequency(data,col,k):#k等频个数
    data_cut=data.copy()
    data_cut[col+'_bin']=pd.qcut(data_cut[col],k,labels=False,duplicates='drop')
    return data_cut


for col in columns_cat_data_numerical_list:
    print(col)
    data_col = data_train[[col, 'target']]
    data_cut_r=cut_Equal_frequency(data_col,col,50)
    data_cut_r=data_cut_r[[col+"_bin","target"]]
    data_col_copy = data_cut_r.copy()
    result_table = ChiMerge(data_col_copy, col+"_bin", 'target', 10)
    code_table = code(result_table)[0]
    col_iv = code(result_table)[1]
    if col_iv >= 0.02:
        data_col_copy.drop_duplicates(subset=col+"_bin",inplace=True)
        data_col_copy.sort_values(by=col+"_bin",inplace=True)
        code_table=code_table[[col+"_bin",'WOE']]
        data_col_copy=pd.merge(data_col_copy,code_table,how='left')
        data_col_copy.fillna(method='bfill', inplace=True)
        data_col = pd.merge(data_cut_r, data_col_copy, how='left', on=[col+"_bin"])
        data_col = data_col[['WOE']]
        data_col.columns = [col + "_code"]
        data_train_code = pd.concat([data_train_code, data_col], axis=1, sort=False)
        iv_list.append(col_iv)
        col_list.append(col+"_code")

    #     code_table['bad_rate'] = code_table['bad_rate'].round(2)
    #     code_table = code_table[['bad_rate', 'WOE']]
    #     bad_rate = pd.merge(bad_rate, code_table, how='left')
    #     bad_rate.fillna(method='bfill', inplace=True)
    #     data_col = pd.merge(data_col, bad_rate, how='left', on=['bad_rate', col])
    #     data_col = data_col[['WOE']]
    #     data_col.columns = [col + "_code"]
    #     # print(data_col.head(20))
    #     data_train_code = pd.concat([data_train_code, data_col], axis=1, sort=False)

#
# print(data_train[["target"]].info())
dict_iv=dict(zip(col_list,iv_list))
with open('iv_col.txt','wb') as f:
    f.write(pickle.dumps(dict_iv))   #dumps序列化源数据后写入文件
    f.close()


data_target=data_train[['target']].reset_index(drop=True)

data_train_code=pd.concat([data_train_code,data_target],axis=1,sort=False)
# print(data_train_code.head())
data_train_code.to_excel("process_code.xlsx")