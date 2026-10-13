import pandas as pd
import datetime as dt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '1'


# ==========================
# 一、数据加载
# ==========================
df_raw = pd.read_excel("data/Online Retail.xlsx")
print(f"原始数据: {len(df_raw)} 行")

# ==========================
# 二、数据清洗
# ==========================
df_clean = df_raw.dropna(subset=['CustomerID'])              # 删无客户ID
df_clean = df_clean[df_clean['Quantity'] > 0]               # 删退货
df_clean = df_clean[df_clean['UnitPrice'] > 0]              # 删异常单价
df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]  # 删取消单

df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']

print(f"清洗后数据: {len(df_clean)} 行")
print(f"最终客户数: {df_clean['CustomerID'].nunique()}")
print(f"保留比例: {len(df_clean)/len(df_raw):.1%}")

# ==========================
# 三、RFM 计算
# ==========================
snapshot_date = df_clean['InvoiceDate'].max() + dt.timedelta(days=1)

rfm = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo':   'nunique',
    'TotalPrice':  'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']
rfm = rfm.reset_index()

print("\nRFM 描述统计:")
print(rfm[['Recency', 'Frequency', 'Monetary']].describe())

# ==========================
# 四、异常值处理
# ==========================
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (series < lower) | (series > upper)

for col in ['Recency', 'Frequency', 'Monetary']:
    outlier_count = detect_outliers_iqr(rfm[col]).sum()
    print(f"{col} 异常值: {outlier_count} ({outlier_count/len(rfm):.2%})")

# 剔除 Monetary 异常值
rfm_no_outlier = rfm[~detect_outliers_iqr(rfm['Monetary'])].copy()
print(f"\n剔除异常值后样本量: {len(rfm_no_outlier)}")

# 可选：log 变换 Monetary（保留所有样本时使用）
# rfm_no_outlier['Monetary'] = np.log1p(rfm_no_outlier['Monetary'])

# ==========================
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']   # 中文显示
rcParams['axes.unicode_minus'] = False     # 负号正常显示

# ==========================
# 五、标准化 + 肘部法
# ==========================
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_no_outlier[['Recency', 'Frequency', 'Monetary']])

inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('聚类数 K')
plt.ylabel('SSE (Inertia)')
plt.title('肘部法确定最佳 K 值')
plt.savefig('output/elbow_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================
# 六、K=4 聚类 + 结果分析
# ==========================
best_k = 4
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
rfm_no_outlier['Cluster'] = kmeans.fit_predict(rfm_scaled)

cluster_summary = rfm_no_outlier.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
cluster_size = rfm_no_outlier['Cluster'].value_counts().sort_index()

print("\n聚类中心均值:")
print(cluster_summary)
print("\n各类别用户数:")
print(cluster_size)
print("\n各类别占比:")
print((cluster_size / len(rfm_no_outlier)).round(2))
# ==========================
# 七、用户命名 + 可视化
# ==========================

# 给每个簇命名
cluster_names = {
    0: '重要价值客户',
    1: '一般维护客户',
    2: '潜力发展客户',
    3: '流失风险客户'
}
rfm_no_outlier['客户类型'] = rfm_no_outlier['Cluster'].map(cluster_names)

# 打印带业务命名的汇总
summary_named = rfm_no_outlier.groupby('客户类型')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
print("\n用户类型均值:")
print(summary_named)

# 雷达图
import numpy as np
from math import pi

categories = ['Recency', 'Frequency', 'Monetary']
N = len(categories)

fig, axes = plt.subplots(2, 2, figsize=(10, 10), subplot_kw=dict(polar=True))
axes = axes.flatten()

for idx, (cluster_id, name) in enumerate(cluster_names.items()):
    values = summary_named.loc[name].values.tolist()
    values += values[:1]  # 闭合
    angles = [n / N * 2 * pi for n in range(N)]
    angles += angles[:1]

    ax = axes[idx]
    ax.plot(angles, values, 'o-', linewidth=2, label=name)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(name, fontsize=12)

plt.tight_layout()
plt.savefig('output/rfm_radar.png', dpi=300, bbox_inches='tight')
plt.show()
# ==========================
# 八、营收贡献分析 + 运营建议表
# ==========================

# 计算各类别的营收贡献
revenue_stats = rfm_no_outlier.groupby('客户类型').agg({
    'CustomerID': 'count',           # 人数
    'Monetary': ['sum', 'mean']      # 总营收、人均营收
}).round(2)

revenue_stats.columns = ['人数', '总营收', '人均营收']
revenue_stats['人数占比'] = (revenue_stats['人数'] / revenue_stats['人数'].sum()).round(2)
revenue_stats['营收占比'] = (revenue_stats['总营收'] / revenue_stats['总营收'].sum()).round(2)

# 按营收占比降序排列
revenue_stats = revenue_stats.sort_values('营收占比', ascending=False)

print("\n各类别营收贡献:")
print(revenue_stats)

# 保存高清图
plt.figure(figsize=(10, 6))
revenue_stats['营收占比'].plot(kind='bar', color='steelblue')
plt.title('各客户类型营收贡献占比')
plt.ylabel('营收占比')
plt.xticks(rotation=15)
plt.tight_layout()

# 营收贡献图
plt.savefig('output/revenue_contribution.png', dpi=300, bbox_inches='tight')
plt.show()


