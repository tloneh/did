import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv(f"data/湛江.csv",encoding='GBK')
print(df.head())

# 设置年份列为索引
df.set_index('年份', inplace=True)

# 计算城乡收入差距（因变量：城镇居民人均可支配收入/乡村居民人均可支配收入）
df['城乡收入差距'] = df['城镇居民人均可支配收入/元'] / df['乡村居民人均可支配收入/元']

# 假设政策实施年份为2005年（可以根据实际情况调整）
policy_year = 2000

# 创建政策前后分组的标识变量，政策后为1，政策前为0
df['政策后'] = (df.index >= policy_year).astype(int)

# 查看数据
print(df[['城乡收入差距', '政策后']].head())

# 定义控制变量
# control_vars = ['第一产业产值/亿元', '第二产业产值/亿元', '第三产业产值/亿元', '总人口/万人', 
#                 'GDP/亿元', '粮食总产量/万吨', '固定资产投资额/亿元', '公共财政预算支出/亿元', 
#                 '土地面积/平方千米', '城镇职工平均工资/元']
control_vars = ['第一产业产值/亿元', '第二产业产值/亿元', '第三产业产值/亿元', '总人口/万人', 
                'GDP/亿元']
# 选择因变量和自变量
X = df[control_vars + ['政策后']]  # 控制变量 + 政策后标识
Y = df['城乡收入差距']  # 因变量：城乡收入差距

# 在X中添加常数项（截距项）
X = sm.add_constant(X)

# 拟合SCCS-DID模型
model = sm.OLS(Y, X).fit()

# 查看模型结果
print(model.summary())


# # 设置字体为黑体（SimHei），确保可以显示中文
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者使用 'Microsoft YaHei'
# # 创建一个图形和多个子图
# plt.figure(figsize=(14, 8))

# # 绘制城乡收入差距的折线图
# plt.subplot(3, 2, 1)
# plt.plot(df.index, df['城乡收入差距'], label='城乡收入差距', color='blue')
# plt.title('城乡收入差距变化趋势')
# plt.xlabel('年份')
# plt.ylabel('城乡收入差距')
# plt.grid(True)

# # 绘制GDP的折线图
# plt.subplot(3, 2, 2)
# plt.plot(df.index, df['GDP/亿元'], label='GDP/亿元', color='green')
# plt.title('GDP变化趋势')
# plt.xlabel('年份')
# plt.ylabel('GDP（亿元）')
# plt.grid(True)

# # 绘制第一产业产值的折线图
# plt.subplot(3, 2, 3)
# plt.plot(df.index, df['第一产业产值/亿元'], label='第一产业产值/亿元', color='orange')
# plt.title('第一产业产值变化趋势')
# plt.xlabel('年份')
# plt.ylabel('产值（亿元）')
# plt.grid(True)

# # 绘制第二产业产值的折线图
# plt.subplot(3, 2, 4)
# plt.plot(df.index, df['第二产业产值/亿元'], label='第二产业产值/亿元', color='red')
# plt.title('第二产业产值变化趋势')
# plt.xlabel('年份')
# plt.ylabel('产值（亿元）')
# plt.grid(True)

# # 绘制第三产业产值的折线图
# plt.subplot(3, 2, 5)
# plt.plot(df.index, df['第三产业产值/亿元'], label='第三产业产值/亿元', color='purple')
# plt.title('第三产业产值变化趋势')
# plt.xlabel('年份')
# plt.ylabel('产值（亿元）')
# plt.grid(True)

# # 调整布局
# plt.tight_layout()
# plt.show()


# 进行异方差性检验
from statsmodels.stats.diagnostic import het_white
_, pval, _, _ = het_white(model.resid, model.model.exog)

print(f'异方差性检验的p值: {pval}')

# cities = ['湛江','茂名','云浮','梅州','阳江','肇庆','河源','汕头','汕尾','潮州','韶关','江门','清远','中山','广州','深圳','东莞'
# ]  # 17个城市名称
# data = {}

# # 读取每个城市的数据
# for city in cities:
#     data[city] = pd.read_csv(f"data/{city}.csv")  # 假设每个城市的数据文件名是 `City.csv`



