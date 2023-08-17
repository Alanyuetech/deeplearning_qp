import pandas as pd

# df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
# print(df)


# #数据筛选，每个bucket只保留一条数据
# data_base = data_base[(data_base['F2023']!=None)&(data_base['F2024']!=None)].groupby('bucket').head(1)

# data_bucket = pd.DataFrame(columns={'ardid','bucket'})

# for index,row in data_new.iterrows():  #循环新的数据
#     for index_b,row_b in data_base.iterrows():   #循环数据库存在的数据
#         if (math.isclose(row['F2021'], row_b['F2021'],0.99))&(math.isclose(row['F2022'], row_b['F2022'],0.99)):   #具体判断条件你自己增减
#             data_bucket = data_bucket.append({'ardid':row['ardid'],'bucket':row_b['bucket']},ignore_index=True)
#         else:
#             if (index_b==data_base.index[-1])&(len(data_bucket[data_bucket['ardid']==row['ardid']])==0):  #判断是否为data_base : 数据库数据  的最后一行, 且data_bucket中没有该ardid
#                 data_bucket = data_bucket.append({'ardid':row['ardid'],'bucket':'new'},ignore_index=True)
    
# data_bucket_num = data_bucket.groupby('ardid').count().reset_index()
# data_bucket_num = data_bucket_num[data_bucket_num['bucket']>1]

# ###################################################################################################
# #数据筛选，每个bucket只保留一条数据
# data_base = data_base[(data_base['F2023']!=None)&(data_base['F2024']!=None)].groupby('bucket').head(1)

# data_bucket = pd.DataFrame(columns={'ardid','bucket'})

# #构建笛卡尔积
# data_new['key'] = 0  #key是新加的一列，之间表中不存在的
# data_base['key'] = 0

# merged_df = data_new.merge(data_base, how='outer', on='key')  #构建笛卡尔积，可能需要修改列名
# merged_df = merged_df.drop('key', axis=1)

# #绝对值
# merged_df['21_cal'] = abs(merged_df['F2021_x'] - merged_df['F2021_y'])/max(merged_df['F2021_x'],merged_df['F2021_y'])
# merged_df['22_cal'] = abs(merged_df['F2022_x'] - merged_df['F2022_y'])/max(merged_df['F2022_x'],merged_df['F2022_y'])
# #筛选21年22年都满足条件的，看你自己加条件
# merged_df = merged_df[(merged_df['21_cal']<0.01)&(merged_df['22_cal']<0.01)]
# merged_df['bucket_x'] = merged_df['bucket_y']  #bucket_x是新的数据修改为主要bucket
# data_new = pd.merge(data_new,merged_df[['ardid','bucket_x']],on='ardid',how='left')  #将新的数据和bucket_x合并,每个ardid不唯一的数据应该也会合并进去
# data_new['bucket_x'] = data_new['bucket_x'].fillna('new')  #全都不符合条件的ardid，bucket_x也会是nan,所以将nan填充为new












df1 = pd.DataFrame({'n': ['K0', 'K1', 'K2', 'K3'],
                    'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3']})
df2 = pd.DataFrame({'n': ['K0', 'K0', 'K1', 'K1', 'K2', 'K2', 'K2'],
                    'C': ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'],
                    'D': ['D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6']})
df1['key'] = 0
df2['key'] = 0

merged_df = df1.merge(df2, how='outer', on='key')
merged_df = merged_df.drop('key', axis=1)
len(merged_df)