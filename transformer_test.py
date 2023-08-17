import pandas as pd
import numpy as np

# 创建30个代码，每个代码重复两次
codes = np.repeat(range(1, 31), 2)

# 创建60个行业
industries = ['Industry ' + str(i) for i in range(1, 61)]

# 创建60个占比，每个为0~100的随机整数
ratios = np.random.randint(0, 101, size=60)

# 创建DataFrame
df = pd.DataFrame({
    '代码': codes,
    '行业': industries,
    '占比': ratios
})

# 对每个代码，根据占比进行降序排序
df.sort_values(['代码', '占比'], ascending=[True, False], inplace=True)  
df.reset_index(drop=True, inplace=True)

print(df)


grouped = df.groupby('代码').agg(list).reset_index()

def determine_industry(row):
    # 如果第一个行业的占比超过 50
    if row['占比'][0] > 50:
        return row['行业'][0]
    # 如果两个行业的占比和超过 60
    elif sum(row['占比']) > 60:
        return ', '.join(row['行业'])
    # 否则返回空
    else:
        return ''

# 应用 determine_industry 函数到每一行
grouped['行业'] = grouped.apply(determine_industry, axis=1)

print(grouped)