#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于数据处理模块
第一步，把原始CSV文件转换成模型可以使用的PyTorch Tensor
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2024.2.2
time: 2025/9/7 13:24
version: V1.0
"""

# data_loader.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from LogicAlgorithm.DL_method.GWNPINN.utils import calculate_adjacency_matrix


class GrottoDataset(Dataset):
    def __init__(self, X, y, y_mask, ambient_y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.y_mask = torch.tensor(y_mask, dtype=torch.float32)  # 新增 mask
        self.ambient_y = torch.tensor(ambient_y, dtype=torch.float32)  # (Batch, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 返回 (Input, Target, Mask)
        return self.X[idx], self.y[idx], self.y_mask[idx], self.ambient_y[idx]


def load_and_preprocess_data(data_path, coords_path, target_grotto_id, seq_len=144,
                             pre_len=12, k_adj=8, z_penalty=10.0):
    """
    加载所有传感器数据，进行预处理，并创建滑动窗口数据集。
    """

    print(f"正在加载第 {target_grotto_id} 窟的数据...")

    # 1. 读取坐标文件并筛选
    coords_df = pd.read_csv(coords_path)

    # --- 筛选特定石窟的传感器 ---
    if 'grottoe_id' not in coords_df.columns:
        raise ValueError("_sensor_coords.csv 中缺少 'grottoe_id' 列！")
    # 只保留 target_grotto_id 的行
    coords_df = coords_df[coords_df['grottoe_id'] == target_grotto_id].copy()
    if coords_df.empty:
        raise ValueError(f"未找到 grottoe_id={target_grotto_id} 的传感器数据，请检查CSV。")

    # 生成该石窟的传感器列表
    sensor_list = coords_df['sensor_id'].tolist()
    print(f"第 {target_grotto_id} 窟共有 {len(sensor_list)} 个传感器。")
    # 重新整理 coords_df，确保顺序与 sensor_list 一致
    coords_df = coords_df.set_index('sensor_id').loc[sensor_list].reset_index()
    # ----------------------------------

    # --- 准备 Zone One-Hot 编码 ---
    # 假设 zone_id 分别为 0, 1, 2 (对应前室、后室、诵经道)
    # 如果 zone_id 不是 0-2，需要先 map 一下，这里假设已经是 0,1,2
    zone_ids = coords_df['zone_id'].values.astype(int)  # shape: (Nodes,)
    num_zones = 3
    # 生成 One-Hot 矩阵: (Nodes, 3)
    zone_one_hot = np.eye(num_zones)[zone_ids]
    # 转置为 (3, Nodes) 以便后续拼接
    zone_features = zone_one_hot.T
    # ---------------------------------

    # 1. 加载并合并所有传感器的时间序列数据
    all_dfs = []
    for sensor_id in sensor_list:
        file_path = os.path.join(data_path, f"{sensor_id}.csv")
        if not os.path.exists(file_path):
            print(f"警告: 找不到文件 {file_path}，将跳过或报错")
            continue

        df = pd.read_csv(file_path, usecols=['time', 'air_temperature'])
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
        df = df.rename(columns={'air_temperature': sensor_id})
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("没有加载到任何传感器数据文件！")

    merged_df = pd.concat(all_dfs, axis=1)

    # --- 生成掩码 & 鲁棒归一化 ---
    # 2.1. 生成掩码 (1=有数据, 0=缺失)
    mask_df = (~merged_df.isna()).astype(float)

    print(f"原始缺失率: {1.0 - mask_df.values.mean():.2%}")

    # 2.2. 手动计算 min/max (忽略 NaN)，避免填0后 Min 变成0导致数据分布被压缩
    data_values = merged_df.values
    temp_min = np.nanmin(data_values)
    temp_max = np.nanmax(data_values)
    print(f"数据范围 (忽略NaN): Min={temp_min:.2f}, Max={temp_max:.2f}")

    # 2.3. 归一化 (保留 NaN)
    # 避免除以0
    scale_range = temp_max - temp_min if temp_max != temp_min else 1.0
    data_normalized = (data_values - temp_min) / scale_range

    # 2.4. 填充 NaN 为 0 (网络输入不能有NaN，0作为"无信号"输入)
    # 注意：这里的0是归一化后的值的占位符，配合Mask使用
    data_normalized = np.nan_to_num(data_normalized, nan=0.0)

    # 掩码不需要归一化，保持 0/1
    mask_values = mask_df.values

    # === 加载外界环境数据 （边界条件）===
    # 注意文件名是处理好的 _core_area_5min.csv
    env_path = os.path.join(data_path, '_core_area_5min.csv')

    if os.path.exists(env_path):
        print(f"正在加载外界环境数据: {env_path}")
        env_df = pd.read_csv(env_path)
        env_df['time'] = pd.to_datetime(env_df['time'])
        env_df = env_df.set_index('time')

        # 即使是5min间隔，为了保险起见，还是做一个 reindex 确保和 merged_df 严格对齐
        # 比如 merged_df 有某几分钟缺失，env_df 也要对应处理
        aligned_env = env_df.reindex(merged_df.index).fillna(method='ffill').fillna(method='bfill')
        # 高版本python需要替换成以下行
        # aligned_env = env_df.reindex(merged_df.index).ffill().bfill()

        ambient_values = aligned_env['air_temperature'].values
    else:
        # 如果还没放进去，为了不报错，这里给个警告
        print(f"Warning: 未找到 {env_path}，将使用均值代替(仅供调试)！")
        ambient_values = np.nanmean(data_values, axis=1)  # 临时fallback

    # === 使用内部温度的 min/max 对外界温度进行归一化 ===
    ambient_norm = (ambient_values - temp_min) / scale_range
    ambient_norm = np.nan_to_num(ambient_norm, nan=0.0)

    # 4. 创建滑动窗口
    X, y, y_masks, ambient_ys = [], [], [], []
    # 预先将 zone 特征扩展到时间维度将消耗巨大内存，
    # 可以选择在循环内拼接，或者其他更聪明的做法：生成 (C, N, T)
    # data_normalized 转置为 (Nodes, Time)
    data_norm_T = data_normalized.T  # (Nodes, Time)
    mask_T = mask_values.T  # (Nodes, Time)

    num_samples = data_normalized.shape[0] - seq_len - pre_len + 1

    for i in range(num_samples):
        # Input: Temp + Zone Features
        # 取出当前窗口的温度: (Nodes, Seq_Len)
        temp_window = data_norm_T[:, i: i + seq_len]

        # 扩展维度为 (1, Nodes, Seq_Len)
        temp_window_expanded = temp_window[np.newaxis, :, :]  # (1, N, T)

        # 将静态的 zone_features (3, Nodes) 扩展到时间维度 -> (3, Nodes, Seq_Len)
        # 使用 np.tile 重复
        zone_repeated = np.tile(zone_features[:, :, np.newaxis], (1, 1, seq_len))  # (3, N, T)

        # 拼接: 温度 + Zone = (1+3, Nodes, Seq_Len) = (4, Nodes, Seq_Len)
        x_sample = np.concatenate([temp_window_expanded, zone_repeated], axis=0)  # (4, N, T)

        # Y 保持不变 (预测目标只有温度): (Nodes, Pre_Len)
        y_sample = data_norm_T[:, i + seq_len: i + seq_len + pre_len]

        # Target Mask: 对应 y 的掩码
        mask_sample = mask_T[:, i + seq_len: i + seq_len + pre_len]  # (N, T_out)

        # Target Ambient (取预测窗口的最后一个时间点作为约束目标)
        idx_target = i + seq_len + pre_len - 1
        amb_sample = ambient_norm[idx_target]

        X.append(x_sample)
        y.append(y_sample)
        y_masks.append(mask_sample)
        ambient_ys.append(amb_sample)

    # 转换为 numpy 数组
    # X shape: (Samples, 4, Nodes, Seq_Len)
    # y shape: (Samples, Nodes, Pre_Len)
    X = np.array(X)
    y = np.array(y)
    y_masks = np.array(y_masks)
    ambient_ys = np.array(ambient_ys).reshape(-1, 1)

    # 5. 加载坐标并创建邻接矩阵(传入 grotto_id)
    bridge_path = os.path.join(data_path, '_inter_zone_bridges.csv')
    adj_matrix = calculate_adjacency_matrix(
        coords_df,
        k=k_adj,
        bridge_path=bridge_path,
        grotto_id=target_grotto_id,
        z_penalty=z_penalty  # 将惩罚系数传给底层建图函数
    )

    # 6. 坐标标准化 (用于PINN解码器)
    coords_scaler = MinMaxScaler()
    # coords_normalized = coords_scaler.fit_transform(coords_df[['x', 'y', 'z']])
    coords_normalized = coords_scaler.fit_transform(coords_df[['x', 'y', 'z']].values)

    # 返回一个 scaler 对象以便兼容 inference 代码，是手动归一化的
    # 构建一个 dummy scaler 存储 min/max
    scaler = MinMaxScaler()
    scaler.min_ = np.array([temp_min])
    scaler.scale_ = np.array([1.0 / scale_range])
    scaler.data_min_ = np.array([temp_min])
    scaler.data_max_ = np.array([temp_max])

    print(f"Dataset构建完成. X:{X.shape}, y:{y.shape}, Mask:{y_masks.shape}")

    return GrottoDataset(X, y, y_masks, ambient_ys), adj_matrix, scaler, coords_df, coords_scaler, float(temp_min), float(temp_max)
