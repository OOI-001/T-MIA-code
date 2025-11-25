# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
# from sklearn.metrics.cluster import contingency_matrix
# from sklearn.preprocessing import StandardScaler
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# from scipy.optimize import linear_sum_assignment
# from collections import defaultdict
# from scipy.stats import skew, kurtosis
#
# matplotlib.use('PDF')  # 强制使用PDF后端
#
# # 配置参数
# RANDOM_STATE = 42
# # 可修改的抽样数量
# SAMPLE_SIZE = 1000  # 可以修改这个值来控制抽样数量
# WINDOW_SIZE = 5  # 滑动窗口大小
#
#
# # 数据加载
# def load_dataset():
#     print("Loading data...")
#     seen = np.load("seen_Auto.npz")
#     unseen = np.load("unseen_Auto.npz")
#     seen_sequences, seen_labels = seen["sequences"], seen["labels"]
#     unseen_sequences, unseen_labels = unseen["sequences"], unseen["labels"]
#
#     return seen_sequences, unseen_sequences, seen_labels, unseen_labels
#
#
# def enhance_features(sequences):
#     """
#     使用滑动窗口提取统计特征来增强特征表示
#     """
#     enhanced = []
#     for seq in sequences:
#         # 如果序列长度小于窗口大小，跳过
#         if len(seq) < WINDOW_SIZE:
#             continue
#
#         for i in range(len(seq) - WINDOW_SIZE + 1):
#             window = seq[i:i + WINDOW_SIZE]
#             try:
#                 enhanced.append([
#                     np.mean(window), np.std(window),
#                     np.percentile(window, 25), np.percentile(window, 75),
#                     np.median(window), skew(window), kurtosis(window),
#                     window[-1] - window[0],  # 窗口内变化
#                     np.mean(np.diff(window)),  # 平均变化率
#                     np.mean(np.abs(window - np.mean(window)) ** 0.5),  # 广义标准差
#                     len([x for x in window if x > np.mean(window)]) / len(window)  # 高于均值的比例
#                 ])
#             except:
#                 # 如果计算统计量出错，使用简单特征
#                 enhanced.append([
#                     np.mean(window), np.std(window),
#                     np.percentile(window, 25), np.percentile(window, 75),
#                     np.median(window), 0, 0,
#                     window[-1] - window[0],
#                     np.mean(np.diff(window)),
#                     np.std(window),
#                     0.5
#                 ])
#     return np.array(enhanced)
#
#
# def purity_score(y_true, y_pred):
#     contingency = contingency_matrix(y_true, y_pred)
#     return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)
#
#
# def cluster_class_mapping_score(y_true, y_pred):
#     """
#     计算簇到类别的映射质量
#     允许一个类别对应多个簇，但一个簇只能对应一个类别
#     """
#     contingency = contingency_matrix(y_true, y_pred)
#     # 找到每个簇应该映射到哪个类别（按多数投票）
#     cluster_to_class = np.argmax(contingency, axis=0)
#
#     # 计算映射后的准确率
#     correct = 0
#     total = 0
#     for cluster_id, true_class in enumerate(cluster_to_class):
#         cluster_mask = (y_pred == cluster_id)
#         if np.sum(cluster_mask) > 0:
#             correct += np.sum((y_true[cluster_mask] == true_class))
#             total += np.sum(cluster_mask)
#
#     return correct / total if total > 0 else 0
#
#
# def multi_cluster_silhouette_score(X, y_true, y_pred):
#     """
#     计算考虑真实类别的轮廓系数变体
#     对于每个样本，计算其与同类别样本的平均距离（无论是否在同一簇）
#     和与不同类别样本的最小平均距离
#     """
#     from sklearn.metrics.pairwise import pairwise_distances
#
#     n_samples = X.shape[0]
#     distances = pairwise_distances(X)
#
#     intra_class_similarity = 0
#     inter_class_separation = 0
#
#     for i in range(n_samples):
#         # 同类样本（无论簇）
#         same_class_mask = (y_true == y_true[i])
#         same_class_distances = distances[i, same_class_mask]
#         a_i = np.mean(same_class_distances)
#
#         # 不同类样本
#         diff_class_mask = (y_true != y_true[i])
#         if np.sum(diff_class_mask) > 0:
#             diff_class_distances = distances[i, diff_class_mask]
#             b_i = np.min([np.mean(distances[i, y_true == c])
#                           for c in np.unique(y_true) if c != y_true[i]])
#         else:
#             b_i = a_i  # 如果只有一类
#
#         intra_class_similarity += a_i
#         inter_class_separation += b_i
#
#     # 归一化版本
#     intra_class_similarity /= n_samples
#     inter_class_separation /= n_samples
#
#     # 类似于轮廓系数的计算，但基于类别而不是簇
#     score = (inter_class_separation - intra_class_similarity) / max(inter_class_separation, intra_class_similarity)
#     return score
#
#
# def cluster_compactness_score(X, y_true, y_pred):
#     """
#     评估每个类别内部的簇紧凑性
#     值越高表示同一类别内的样本在簇中更紧凑
#     """
#     from sklearn.metrics.pairwise import pairwise_distances
#
#     unique_classes = np.unique(y_true)
#     total_compactness = 0
#
#     for class_label in unique_classes:
#         class_mask = (y_true == class_label)
#         class_clusters = y_pred[class_mask]
#
#         if len(np.unique(class_clusters)) <= 1:
#             # 如果这个类别只有一个簇，紧凑性为1
#             total_compactness += 1.0
#         else:
#             # 计算这个类别内各簇的紧凑性
#             class_compactness = 0
#             for cluster_id in np.unique(class_clusters):
#                 cluster_mask = class_mask & (y_pred == cluster_id)
#                 if np.sum(cluster_mask) > 1:
#                     cluster_points = X[cluster_mask]
#                     centroid = np.mean(cluster_points, axis=0)
#                     distances = np.linalg.norm(cluster_points - centroid, axis=1)
#                     cluster_compactness = 1 / (1 + np.mean(distances))
#                     class_compactness += cluster_compactness * np.sum(cluster_mask)
#
#             total_compactness += class_compactness / np.sum(class_mask)
#
#     return total_compactness / len(unique_classes)
#
#
# def class_separation_score(X, y_true, y_pred):
#     """
#     评估不同类别之间的分离程度
#     基于各类别中心之间的距离
#     """
#     unique_classes = np.unique(y_true)
#     if len(unique_classes) <= 1:
#         return 1.0  # 如果只有一类，分离度最高
#
#     # 计算每个类别的平均中心
#     class_centers = []
#     for class_label in unique_classes:
#         class_mask = (y_true == class_label)
#         class_center = np.mean(X[class_mask], axis=0)
#         class_centers.append(class_center)
#
#     class_centers = np.array(class_centers)
#
#     # 计算类别中心之间的最小距离
#     from sklearn.metrics.pairwise import pairwise_distances
#     center_distances = pairwise_distances(class_centers)
#     np.fill_diagonal(center_distances, np.inf)  # 忽略自身距离
#
#     min_separation = np.min(center_distances)
#
#     # 归一化到0-1范围（需要根据数据特性调整）
#     # 这里使用简单的sigmoid函数进行归一化
#     normalized_separation = 1 / (1 + np.exp(-min_separation / np.std(X)))
#
#     return normalized_separation
#
#
# def random_subsample(X, y, sample_size, random_state=RANDOM_STATE):
#     """
#     从数据集中随机抽取指定数量的样本
#     保持类别比例
#     """
#     np.random.seed(random_state)
#
#     if sample_size >= len(X):
#         print(f"Sample size {sample_size} >= total samples {len(X)}, using all data")
#         return X, y
#
#     # 计算每个类别的样本数
#     unique_classes, class_counts = np.unique(y, return_counts=True)
#     print(f"Original class distribution: {dict(zip(unique_classes, class_counts))}")
#
#     # 按类别比例抽样
#     sampled_indices = []
#     for class_label in unique_classes:
#         class_indices = np.where(y == class_label)[0]
#         class_sample_size = int(sample_size * len(class_indices) / len(X))
#
#         if class_sample_size > len(class_indices):
#             class_sample_size = len(class_indices)
#
#         sampled_class_indices = np.random.choice(class_indices, class_sample_size, replace=False)
#         sampled_indices.extend(sampled_class_indices)
#
#     # 如果抽样数量不足，随机补充
#     if len(sampled_indices) < sample_size:
#         remaining_indices = list(set(range(len(X))) - set(sampled_indices))
#         additional_indices = np.random.choice(remaining_indices, sample_size - len(sampled_indices), replace=False)
#         sampled_indices.extend(additional_indices)
#
#     sampled_indices = np.array(sampled_indices)
#     X_sampled = X[sampled_indices]
#     y_sampled = y[sampled_indices]
#
#     # 打印抽样后的类别分布
#     unique_sampled, counts_sampled = np.unique(y_sampled, return_counts=True)
#     print(f"Sampled class distribution: {dict(zip(unique_sampled, counts_sampled))}")
#     print(f"Sampled data shape: {X_sampled.shape}")
#
#     return X_sampled, y_sampled
#
#
# # 聚类可视化函数 - 修改后的版本，只保留一张True Label图
# def visualize_clusters_enhanced(X, y, cluster_results, k_values, tsne_results, pca_results, pca_model, kmeans_models):
#     """
#     优化后的可视化函数，只显示一张True Label图
#     """
#     n_k = len(k_values)
#
#     # 创建大图 - 调整为2行 x (n_k+1)列
#     fig, axes = plt.subplots(2, n_k + 1, figsize=(6 * (n_k + 1), 12))
#
#     # 第一行：True Label + 各k值的聚类结果
#     # True Label图（第一列）
#     scatter = axes[0, 0].scatter(tsne_results[:, 0], tsne_results[:, 1], c=y,
#                                  cmap=plt.cm.Paired, alpha=0.6, edgecolors='w')
#     axes[0, 0].set_title('True Labels(Amazon_Fasion)\n(0=Unseen, 1=Seen)')
#     axes[0, 0].set_xlabel('t-SNE 1')
#     axes[0, 0].set_ylabel('t-SNE 2')
#     axes[0, 0].legend(*scatter.legend_elements(), title="Classes")
#
#     # 各k值的聚类结果
#     markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
#     for i, k in enumerate(k_values):
#         y_cluster = cluster_results[k]
#         colors = plt.cm.tab10(np.linspace(0, 1, k))
#
#         for cluster_id in range(k):
#             cluster_mask = (y_cluster == cluster_id)
#             axes[0, i + 1].scatter(tsne_results[cluster_mask, 0], tsne_results[cluster_mask, 1],
#                                    c=[colors[cluster_id]], marker=markers[cluster_id % len(markers)],
#                                    alpha=0.6, edgecolors='w', label=f'Cluster {cluster_id}')
#
#         axes[0, i + 1].set_title(f'K={k} - Cluster Results')
#         axes[0, i + 1].set_xlabel('t-SNE 1')
#         axes[0, i + 1].set_ylabel('t-SNE 2')
#         if i == 0:  # 只在第一个图显示图例
#             axes[0, i + 1].legend(title="Clusters")
#
#     # 第二行：各k值的决策边界
#     for i, k in enumerate(k_values):
#         y_cluster = cluster_results[k]
#         colors = plt.cm.tab10(np.linspace(0, 1, k))
#         kmeans_model = kmeans_models[k]
#
#         # 生成网格数据
#         x_min, x_max = pca_results[:, 0].min() - 1, pca_results[:, 0].max() + 1
#         y_min, y_max = pca_results[:, 1].min() - 1, pca_results[:, 1].max() + 1
#         xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
#                              np.arange(y_min, y_max, 0.1))
#
#         # 预测网格点类别
#         grid_points = np.c_[xx.ravel(), yy.ravel()]
#         grid_points_original = pca_model.inverse_transform(grid_points)
#         Z = kmeans_model.predict(grid_points_original)
#         Z = Z.reshape(xx.shape)
#
#         axes[1, i + 1].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.tab10)
#         for cluster_id in range(k):
#             cluster_mask = (y_cluster == cluster_id)
#             axes[1, i + 1].scatter(pca_results[cluster_mask, 0], pca_results[cluster_mask, 1],
#                                    c=[colors[cluster_id]], marker=markers[cluster_id % len(markers)],
#                                    alpha=0.6, edgecolors='k', s=50, label=f'Cluster {cluster_id}')
#         axes[1, i + 1].set_title(f'K={k} - Decision Boundaries')
#         axes[1, i + 1].set_xlabel('PCA 1')
#         axes[1, i + 1].set_ylabel('PCA 2')
#         if i == 0:  # 只在第一个图显示图例
#             axes[1, i + 1].legend(title="Clusters")
#
#     # 隐藏第二行第一列（保持对齐）
#     axes[1, 0].axis('off')
#
#     plt.tight_layout()
#     return fig
#
#
# # 主流程
# def main():
#     # 加载数据
#     (seen_sequences, unseen_sequences, seen_labels, unseen_labels) = load_dataset()
#
#     print(f"Seen sequences shape: {seen_sequences.shape}")
#     print(f"Unseen sequences shape: {unseen_sequences.shape}")
#     print(f"Seen labels shape: {seen_labels.shape}")
#     print(f"Unseen labels shape: {unseen_labels.shape}")
#
#     # 检查标签分布
#     print(f"Seen labels distribution: {np.unique(seen_labels, return_counts=True)}")
#     print(f"Unseen labels distribution: {np.unique(unseen_labels, return_counts=True)}")
#
#     # 直接合并数据用于聚类
#     X = np.vstack([seen_sequences, unseen_sequences])
#     y = np.concatenate([np.ones(len(seen_sequences)), np.zeros(len(unseen_sequences))])
#
#     print(f"Combined data shape: {X.shape}")
#     print(f"Combined labels shape: {y.shape}")
#     print(f"Combined labels distribution: {np.unique(y, return_counts=True)}")
#
#     # 随机抽样
#     print(f"\nRandomly sampling {SAMPLE_SIZE} samples from the dataset...")
#     X_sampled, y_sampled = random_subsample(X, y, SAMPLE_SIZE)
#
#     # 使用抽样后的数据进行后续分析
#     X_analysis = X_sampled
#     y_analysis = y_sampled
#
#     # 特征增强
#     print("Enhancing features with sliding window statistics...")
#     X_enhanced = enhance_features(X_analysis)
#
#     # 扩展标签以匹配增强后的特征
#     sequences_per_sample = len(X_enhanced) // len(X_analysis)
#     y_enhanced = np.repeat(y_analysis, sequences_per_sample)[:len(X_enhanced)]
#
#     print(f"Enhanced data shape: {X_enhanced.shape}")
#     print(f"Enhanced labels shape: {y_enhanced.shape}")
#
#     # 使用增强后的特征
#     X_final = X_enhanced
#     y_final = y_enhanced
#
#     # 标准化数据
#     X_final = StandardScaler().fit_transform(X_final)
#
#     # 预先计算降维结果（用于所有k值的可视化）
#     print("Computing dimensionality reduction...")
#     tsne = TSNE(n_components=2, random_state=RANDOM_STATE)
#     X_tsne = tsne.fit_transform(X_final)
#
#     pca = PCA(n_components=2, random_state=RANDOM_STATE)
#     X_pca = pca.fit_transform(X_final)
#
#     # 存储不同k值的聚类结果
#     k_values = [2, 3, 4, 5]
#     kmeans_models = {}
#     cluster_results = {}
#     performance_metrics = {}
#
#     # 对每个k值进行聚类
#     for k in k_values:
#         print(f"\nTraining KMeans with k={k}...")
#         kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE)
#         kmeans.fit(X_final)
#
#         y_cluster = kmeans.labels_
#
#         # 计算传统评估指标
#         ari = adjusted_rand_score(y_final, y_cluster)
#         nmi = normalized_mutual_info_score(y_final, y_cluster)
#         purity = purity_score(y_final, y_cluster)
#
#         # 计算新的评估指标（适合多簇对应单类的情况）
#         mapping_score = cluster_class_mapping_score(y_final, y_cluster)
#         multi_silhouette = multi_cluster_silhouette_score(X_final, y_final, y_cluster)
#         compactness = cluster_compactness_score(X_final, y_final, y_cluster)
#         separation = class_separation_score(X_final, y_final, y_cluster)
#
#         # 存储结果
#         kmeans_models[k] = kmeans
#         cluster_results[k] = y_cluster
#         performance_metrics[k] = {
#             'ARI': ari,
#             'NMI': nmi,
#             'Purity': purity,
#             'Mapping_Score': mapping_score,
#             'Multi_Cluster_Silhouette': multi_silhouette,
#             'Cluster_Compactness': compactness,
#             'Class_Separation': separation
#         }
#
#         print(f"=== K={k} Clustering Performance ===")
#         print(f"Traditional Metrics:")
#         print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
#         print(f"  Normalized Mutual Information (NMI): {nmi:.4f}")
#         print(f"  Purity: {purity:.4f}")
#         print(f"Multi-Cluster per Class Metrics:")
#         print(f"  Cluster-Class Mapping Score: {mapping_score:.4f}")
#         print(f"  Multi-Cluster Silhouette: {multi_silhouette:.4f}")
#         print(f"  Cluster Compactness: {compactness:.4f}")
#         print(f"  Class Separation: {separation:.4f}")
#
#     # 使用优化后的可视化函数
#     # fig = visualize_clusters_enhanced(X_final, y_final, cluster_results, k_values, X_tsne, X_pca, pca, kmeans_models)
#     #
#     # # 修改这里：保存为PDF矢量图，移除dpi参数
#     # plt.savefig(f'clustering_comparison_enhanced_Fasion_{SAMPLE_SIZE}.pdf',
#     #             format='pdf',
#     #             bbox_inches='tight')  # 移除dpi参数以生成矢量图
#     # plt.show()
#
#     # 打印性能指标汇总
#     print("\n=== Performance Summary ===")
#     print("Traditional Metrics:")
#     for k in k_values:
#         metrics = performance_metrics[k]
#         print(f"K={k}: ARI={metrics['ARI']:.4f}, NMI={metrics['NMI']:.4f}, Purity={metrics['Purity']:.4f}")
#
#     print("\nMulti-Cluster per Class Metrics:")
#     for k in k_values:
#         metrics = performance_metrics[k]
#         print(f"K={k}: Mapping={metrics['Mapping_Score']:.4f}, M-Silhouette={metrics['Multi_Cluster_Silhouette']:.4f}, "
#               f"Compactness={metrics['Cluster_Compactness']:.4f}, Separation={metrics['Class_Separation']:.4f}")
#
#
# if __name__ == "__main__":
#     main()

from PyPDF2 import PdfReader, PdfWriter, PageObject
import copy


def merge_pdfs_vertically(pdf1_path, pdf2_path, output_path):
    # 读取两个PDF文件
    pdf1 = PdfReader(pdf1_path)
    pdf2 = PdfReader(pdf2_path)

    # 获取第一页
    page1 = pdf1.pages[0]
    page2 = pdf2.pages[0]

    # 获取页面尺寸
    width1 = page1.mediabox.width
    height1 = page1.mediabox.height
    width2 = page2.mediabox.width
    height2 = page2.mediabox.height

    # 验证页面规格是否相同
    if width1 != width2:
        print("警告：PDF宽度不一致，将使用第一个PDF的宽度")

    # 创建新的空白页面，高度为两个页面之和
    new_height = height1 + height2
    new_page = PageObject.create_blank_page(None, width1, new_height)

    # 创建页面的深拷贝以避免修改原始页面
    page1_copy = copy.deepcopy(page1)
    page2_copy = copy.deepcopy(page2)

    # 将第一页内容添加到新页面顶部
    new_page.merge_page(page1_copy)

    # 将第二页内容添加到新页面底部（需要垂直平移）
    # 创建变换矩阵：[1, 0, 0, 1, tx, ty]
    # 这里ty = -height1 表示向下移动height1个单位
    page2_copy.add_transformation([1, 0, 0, 1, 0, -height1])
    new_page.merge_page(page2_copy)

    # 输出PDF
    writer = PdfWriter()
    writer.add_page(new_page)

    with open(output_path, 'wb') as output_file:
        writer.write(output_file)


# 使用示例
merge_pdfs_vertically('clustering_comparison_enhanced_Auto_1000.pdf', 'clustering_comparison_enhanced_Fasion_1000.pdf',
                      'merged.pdf')