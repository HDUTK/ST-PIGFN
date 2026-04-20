#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于辅助工具
通用的小功能，比如根据坐标计算邻接矩阵
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2024.2.2
time: 2025/9/7 13:23
version: V1.0
"""

# utils.py
import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import cdist


def calculate_adjacency_matrix(coords_df, k, bridge_path=None,
                               grotto_id=None, z_penalty=10.0):
    """
    根据传感器坐标计算邻接矩阵 (K近邻)
    Args:
        coords_df (pd.DataFrame): 包含 'x', 'y', 'z' 列的DataFrame
        k (int): K近邻的K值
        bridge_path (str): 'inter_zone_bridges.csv' 的路径，如果为None则不处理桥梁
        grotto_id: 当前的目标石窟ID (9 或 10)
        z_penalty: 各向异性 Z 轴惩罚
    Returns:
        np.ndarray: K近邻邻接矩阵
    """
    # 1. 准备数据
    sensor_ids = coords_df['sensor_id'].values
    # coords = coords_df[['x', 'y', 'z']].values

    # 必须使用 .copy()，防止修改原始 DataFrame 的数据
    coords = coords_df[['x', 'y', 'z']].values.copy()

    # 对 Z 轴施加各向异性惩罚
    coords[:, 2] = coords[:, 2] * z_penalty

    zone_ids = coords_df['zone_id'].values

    n_sensors = len(sensor_ids)

    # 2. 计算全量的欧氏距离矩阵
    dist_mx = cdist(coords, coords, metric='euclidean')

    # 3. 构建结构掩码 (Mask)
    # 逻辑：mask[i, j] = 1 表示允许连接，0 表示禁止连接
    # 3.1 默认规则：只有同区域 (Same Zone) 才能连接
    # 利用广播机制生成同区域矩阵: (N, 1) == (1, N) -> (N, N)
    zone_mask = (zone_ids[:, None] == zone_ids[None, :]).astype(float)

    # 3.2 桥梁规则：读取Excel，开启特定的跨区域连接
    bridge_mask = np.zeros((n_sensors, n_sensors))

    if bridge_path and os.path.exists(bridge_path):
        try:
            bridge_df = pd.read_csv(bridge_path)

            # --- 根据 grottoe_id 过滤桥梁 ---
            if grotto_id is not None and 'grottoe_id' in bridge_df.columns:
                bridge_df = bridge_df[bridge_df['grottoe_id'] == grotto_id]
            # --------------------------------------

            # 建立 ID 到 索引 的映射字典
            id_to_idx = {sid: i for i, sid in enumerate(sensor_ids)}

            for _, row in bridge_df.iterrows():
                id1 = str(row['sensor_id_1'])
                id2 = str(row['sensor_id_2'])

                if id1 in id_to_idx and id2 in id_to_idx:
                    idx1 = id_to_idx[id1]
                    idx2 = id_to_idx[id2]

                    # 在掩码中打通这两个点（双向）
                    bridge_mask[idx1, idx2] = 1.0
                    bridge_mask[idx2, idx1] = 1.0
            print(f"[{grotto_id}号窟] 成功加载桥梁，激活了 {int(bridge_mask.sum()/2)} 条跨区通道。")
        except Exception as e:
            print(f"加载桥梁文件失败: {e}，将仅使用区域内连接。")

    # 3.3 合并掩码：同区域 OR 是桥梁
    final_allow_mask = np.logical_or(zone_mask, bridge_mask)

    # 4. 应用掩码到距离矩阵
    # 将不允许连接的地方距离设为无穷大，这样在找K近邻时它们就会被忽略
    masked_dist_mx = dist_mx.copy()
    masked_dist_mx[~final_allow_mask] = np.inf

    # 5. 计算 K 近邻邻接矩阵 (与之前逻辑一致，只是输入变成了 masked_dist_mx)
    adj = np.zeros_like(dist_mx, dtype=float)

    for i in range(n_sensors):
        # 找到最近的 k+1 个（包含自己）
        # 如果有效邻居少于 k 个（比如孤岛），argsort 仍然工作，但距离是 inf
        # 注意：argpartition 对于 inf 处理可能不稳定，建议用 sort

        # 获取第 i 行距离
        dists = masked_dist_mx[i]

        # 排序找到索引
        sorted_indices = np.argsort(dists)

        # 取最近的 k+1 个（排除自己后是 k 个）
        # 注意：要检查距离是否为 inf，如果是 inf 说明邻居不够，不能硬连
        nearest_indices = []
        for idx in sorted_indices[1:k + 1]:
            if dists[idx] != np.inf:
                nearest_indices.append(idx)
            else:
                break  # 后面都是 inf 了

        if len(nearest_indices) > 0:
            nearest_indices = np.array(nearest_indices)

            # 高斯权重计算
            valid_dists = dists[nearest_indices]
            sigma = max(1e-6, valid_dists.mean())  # 动态调整 sigma
            weights = np.exp(- (valid_dists ** 2) / (2 * sigma ** 2))

            adj[i, nearest_indices] = weights
            adj[nearest_indices, i] = weights  # 保证对称性 (虽然 K近邻通常不对称，但这里强制对称有助于图卷积稳定)

    # 6. 归一化
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    adj = adj / row_sums

    return adj
