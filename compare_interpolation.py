#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于
核心空间基线对比测试（遮蔽20%）
集成 6 大方法：
1. Ours (ST-PIGFN) -> 完整框架
2. GWN+RBF (Data-only) -> 纯数据驱动时空图网络
3. Standard PINN (Static Source) -> 传统物理信息网络
4. Pure INR -> 纯坐标隐式神经表示
5. IDW -> 传统距离插值
6. Kriging -> 地统计学克里金插值
集成 6 大方法：ST-PIGFN, GWN+RBF, Standard PINN, Pure INR, IDW, Kriging
测试策略：物理隔离测试（基于训练时预留的 20% 传感器）
同时按全窟(Global)以及前室(0)、后室(1)、诵经道(2)分别计算并统计评价指标。
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2024.2.2
time: 2026/1/25 17:13
version: V1.0
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# 引入项目模块
from LogicAlgorithm.DL_method.GWNPINN.model_1 import SpatioTemporalPINN

try:
    from pykrige.ok3d import OrdinaryKriging3D
except ImportError:
    print("警告: 未安装 pykrige，Kriging 对比将失效。")


# === 1. 定义 Pure INR 模型结构 (必须与训练脚本一致) ===
class PurePINN(nn.Module):
    def __init__(self, layers=[4, 64, 64, 64, 64, 1]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.activation = nn.Tanh()

    def forward(self, x, y, z, t):
        inputs = torch.cat([x, y, z, t], dim=1)
        for i in range(len(self.layers) - 1):
            inputs = self.activation(self.layers[i](inputs))
        return self.layers[-1](inputs)


# === 2. 传统插值算法 ===
def idw_interpolation(train_coords, train_vals, query_coords, power=2):
    preds = []
    for q in query_coords:
        dists = np.linalg.norm(train_coords - q, axis=1)
        if np.min(dists) < 1e-6:
            preds.append(train_vals[np.argmin(dists)])
            continue
        weights = 1.0 / (dists ** power)
        preds.append(np.dot(weights / weights.sum(), train_vals))
    return np.array(preds)


def kriging_interpolation(train_coords, train_vals, query_coords):
    try:
        ok3d = OrdinaryKriging3D(train_coords[:, 0], train_coords[:, 1], train_coords[:, 2],
                                 train_vals, variogram_model='linear', verbose=False)
        pred, _ = ok3d.execute('points', query_coords[:, 0], query_coords[:, 1], query_coords[:, 2])
        return pred
    except:
        return np.full(len(query_coords), np.mean(train_vals))


# === 3. 辅助函数 ===
def rebuild_scaler(params):
    scaler = MinMaxScaler()
    scaler.min_, scaler.scale_ = np.array(params['min_']), np.array(params['scale_'])
    scaler.data_min_, scaler.data_max_ = np.array(params['data_min_']), np.array(params['data_max_'])
    return scaler


def load_data_and_coords(data_path, coords_path, grotto_id):
    c_df = pd.read_csv(coords_path)
    c_df = c_df[c_df['grottoe_id'] == grotto_id]
    s_ids = c_df['sensor_id'].tolist()

    dfs = []
    for sid in s_ids:
        f = os.path.join(data_path, f"{sid}.csv")
        if not os.path.exists(f): continue
        df = pd.read_csv(f, usecols=['time', 'air_temperature']).set_index('time')
        df.index = pd.to_datetime(df.index)
        dfs.append(df.rename(columns={'air_temperature': sid}))

    full_df = pd.concat(dfs, axis=1).sort_index().interpolate(method='time').ffill().bfill()
    return full_df, c_df[['x', 'y', 'z']].values, s_ids, c_df['zone_id'].values


def main():
    # ================= 配置区域 =================
    TARGET_GROTTO_ID = 10
    TEST_SAMPLE_STRIDE = 100  # 测试步长，调大可加快速度
    DATA_PATH = './data'
    COORDS_PATH = './data/_sensor_coords.csv'

    # 权重路径 (请确保这些文件夹和文件存在)
    CKPT_FULL = f'./results_slices_FULL_Z_10_Masked/grotto_{TARGET_GROTTO_ID}_model_checkpoint.pth'
    CKPT_DATA = f'./results_slices_DATA_DRIVEN_Masked/grotto_{TARGET_GROTTO_ID}_model_checkpoint.pth'
    CKPT_STATIC = f'./results_slices_STATIC_SOURCE_Masked/grotto_{TARGET_GROTTO_ID}_model_checkpoint.pth'
    CKPT_INR = './pure_inr_masked_checkpoint.pth'
    TEST_JSON = f'./results_slices_FULL_Z_10_Masked/test_sensors.json'  # 核心：读取物理隔离名单

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================= 1. 加载名单与环境 =================
    if not os.path.exists(TEST_JSON):
        raise FileNotFoundError("错误：找不到 test_sensors.json，请先运行开启遮蔽模式的 main.py 训练")

    with open(TEST_JSON, 'r') as f:
        test_sensor_ids = json.load(f)

    base_ckpt = torch.load(CKPT_FULL, map_location=device)
    temp_min, temp_max = base_ckpt['scalers_params']['global_min'], base_ckpt['scalers_params']['global_max']
    temp_scaler = rebuild_scaler(base_ckpt['scalers_params']['temp_scaler'])
    coords_scaler = rebuild_scaler(base_ckpt['scalers_params']['coords_scaler'])
    pre_adj_t = torch.tensor(np.array(base_ckpt['graph_data']['adj_matrix']), dtype=torch.float32, device=device)
    seq_len = base_ckpt['train_info']['seq_len']

    full_df, coords_raw, sensor_ids, zone_ids = load_data_and_coords(DATA_PATH, COORDS_PATH, TARGET_GROTTO_ID)
    data_norm = pd.DataFrame(temp_scaler.transform(full_df.values), index=full_df.index).values
    zone_one_hot = np.eye(3)[zone_ids].T

    # 确定测试节点和训练节点的索引
    test_idx = [sensor_ids.index(sid) for sid in test_sensor_ids if sid in sensor_ids]
    train_idx = [i for i in range(len(sensor_ids)) if i not in test_idx]

    # ================= 2. 加载模型矩阵 =================
    def init_model(path):
        if not os.path.exists(path): return None
        ckpt = torch.load(path, map_location=device)
        cfg = ckpt['config']
        m = SpatioTemporalPINN(num_nodes=cfg['num_nodes'], sensor_coords_norm=cfg['sensor_coords_norm'],
                               in_dim=cfg['input_dim'], out_dim=cfg['out_dim']).to(device)
        m.load_state_dict(ckpt['model_state_dict'])
        return m.eval()

    models = {
        'Ours (ST-PIGFN)': init_model(CKPT_FULL),
        'GWN+RBF (Data-only)': init_model(CKPT_DATA),
        'Standard PINN': init_model(CKPT_STATIC),
        'Pure INR': None
    }

    if os.path.exists(CKPT_INR):
        m_inr = PurePINN().to(device)
        m_inr.load_state_dict(torch.load(CKPT_INR, map_location=device))
        models['Pure INR'] = m_inr.eval()
        t_sec = (full_df.index - full_df.index.min()).total_seconds().values.reshape(-1, 1)
        inr_t_scaler = MinMaxScaler().fit(t_sec)

    # ================= 3. 初始化区域统计字典 =================
    ZONES = {'Global': 'All Sensors', 0: 'Front Chamber', 1: 'Rear Chamber', 2: 'Chanting Passage'}
    METHODS = list(models.keys()) + ['IDW', 'Kriging']
    results_store = {zk: {m: {'true': [], 'pred': []} for m in METHODS} for zk in ZONES.keys()}

    # ================= 4. 执行物理隔离对比循环 =================
    test_start = int(len(full_df) * 0.9)
    print(f"\n🚀 开始分区域对比实验 | 测试点数量: {len(test_idx)} | 采样步长: {TEST_SAMPLE_STRIDE}")

    for t in range(test_start, len(full_df), TEST_SAMPLE_STRIDE):
        if t < seq_len: continue

        # 提取真值与区域 ID
        true_vals = full_df.values[t, test_idx]
        t_zones = zone_ids[test_idx]

        # 准备 GNN 模型输入 (只使用 80% 训练节点的数据)
        x_in = torch.tensor(np.concatenate([
            data_norm[t - seq_len:t, train_idx].T[np.newaxis, :, :],  # (1, 80%Nodes, Seq)
            np.tile(zone_one_hot[:, train_idx, np.newaxis], (1, 1, seq_len))
        ], axis=0), dtype=torch.float32).unsqueeze(0).to(device)

        # 准备查询坐标 (20% 测试节点)
        q_coords_raw = coords_raw[test_idx]
        q_coords_norm = torch.tensor(coords_scaler.transform(q_coords_raw), dtype=torch.float32).unsqueeze(0).to(device)

        preds = {}
        # --- GNN 衍生模型推理 ---
        for name in ['Ours (ST-PIGFN)', 'GWN+RBF (Data-only)', 'Standard PINN']:
            if models[name] is None: continue
            with torch.no_grad():
                _, s_feats = models[name].gwn(x_in, pre_adj_t)
                interp_f = models[name].interpolator(q_coords_norm[..., :3], models[name].sensor_coords, s_feats)
                t_zero = torch.zeros((1, len(test_idx), 1), device=device)
                p_norm = models[name].decoder(torch.cat([q_coords_norm, t_zero, interp_f], dim=-1)).squeeze()
                preds[name] = p_norm.cpu().numpy() * (temp_max - temp_min) + temp_min

        # --- Pure INR 推理 ---
        if models['Pure INR'] is not None:
            t_n = inr_t_scaler.transform([[(full_df.index[t] - full_df.index.min()).total_seconds()]])[0, 0]
            with torch.no_grad():
                c_n = coords_scaler.transform(q_coords_raw)
                p_inr = models['Pure INR'](
                    torch.tensor(c_n[:, 0:1], dtype=torch.float32).to(device),
                    torch.tensor(c_n[:, 1:2], dtype=torch.float32).to(device),
                    torch.tensor(c_n[:, 2:3], dtype=torch.float32).to(device),
                    torch.full((len(test_idx), 1), t_n, dtype=torch.float32).to(device)
                ).cpu().numpy().flatten()
            preds['Pure INR'] = p_inr * (temp_max - temp_min) + temp_min

        # --- 传统插值 ---
        preds['IDW'] = idw_interpolation(coords_raw[train_idx], full_df.values[t, train_idx], q_coords_raw)
        preds['Kriging'] = kriging_interpolation(coords_raw[train_idx], full_df.values[t, train_idx], q_coords_raw)

        # --- 数据分类存入 ---
        for m_name, m_val in preds.items():
            if m_val is None: continue
            results_store['Global'][m_name]['true'].extend(true_vals)
            results_store['Global'][m_name]['pred'].extend(m_val)
            for zk in [0, 1, 2]:
                mask = (t_zones == zk)
                if np.any(mask):
                    results_store[zk][m_name]['true'].extend(true_vals[mask])
                    results_store[zk][m_name]['pred'].extend(m_val[mask])

    # ================= 5. 输出分区域豪华结果表 =================
    print("\n" + "=".upper().center(80, "="))
    print(" 📊 3D微气候场重建：多模型分区域性能评估报告 (物理隔离测试) ".center(80))
    print("=".upper().center(80, "="))

    for zk, zn in ZONES.items():
        print(f"\n📍 评估区域: {zn}")
        print(f"{'Method':<25} | {'RMSE':<10} | {'MAE':<10} | {'R2':<10}")
        print("-" * 75)
        for m_name in METHODS:
            data = results_store[zk][m_name]
            if not data['true']: continue
            y_t, y_p = np.array(data['true']), np.array(data['pred'])
            rmse, mae, r2 = mean_squared_error(y_t, y_p, squared=False), mean_absolute_error(y_t, y_p), r2_score(y_t,
                                                                                                                 y_p)
            print(f"{m_name:<25} | {rmse:<10.4f} | {mae:<10.4f} | {r2:<10.4f}")
    print("\n" + "=".upper().center(80, "="))


if __name__ == '__main__':
    main()