#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于标准的 PINN 实现，没有任何图网络结构，(需要遮蔽20%进行对比)
只用 MLP 拟合坐标和时间。运行这个文件得到纯 PINN 在测试集上的精度，用于和上面的结果做对比
Baseline: Standard Pure PINN (无时空图结构)
用于对比实验：证明 ST-PIGFN 优于纯 PINN
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2024.2.2
time: 2026/1/22 14:12
version: V1.0
"""


import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# === 1. 定义纯 PINN 模型 ===
class PurePINN(nn.Module):
    def __init__(self, layers=[4, 64, 64, 64, 64, 1]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.activation = nn.Tanh()  # PINN 常用 Tanh

    def forward(self, x, y, z, t):
        # 输入: (N, 4) -> x, y, z, t
        inputs = torch.cat([x, y, z, t], dim=1)
        for i in range(len(self.layers) - 1):
            inputs = self.activation(self.layers[i](inputs))
        output = self.layers[-1](inputs)
        return output


# === 2. 训练脚本 ===
def train_and_evaluate_pinn():
    # ================= 配置区域 =================
    GROTTO_ID = 10  # 确认石窟ID
    EPOCHS = 5000  # 训练轮数
    LR = 0.0005  # 学习率
    DATA_PATH = './data'
    COORDS_PATH = './data/_sensor_coords.csv'
    # ===========================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Pure PINN on {device}...")

    # --- A. 加载坐标信息 ---
    if not os.path.exists(COORDS_PATH):
        print(f"Error: {COORDS_PATH} not found.")
        return

    coords_df = pd.read_csv(COORDS_PATH)
    # 筛选特定石窟
    coords_df = coords_df[coords_df['grottoe_id'] == GROTTO_ID]
    # 只保留需要的列，防止列名冲突
    coords_info = coords_df[['sensor_id', 'x', 'y', 'z']]
    sensor_ids = coords_df['sensor_id'].tolist()

    print(f"Found {len(sensor_ids)} sensors for Grotto {GROTTO_ID}.")

    # 加载 main.py 生成的测试名单
    with open('./results_slices_FULL_Masked/test_sensors.json', 'r') as f:
        test_sensor_ids = json.load(f)

    # --- B. 加载传感器时序数据 ---
    dfs = []
    print("Loading sensor CSVs...")
    for sid in sensor_ids:
        if sid in test_sensor_ids:
            continue
        f = os.path.join(DATA_PATH, f"{sid}.csv")
        if os.path.exists(f):
            # 只读时间和温度
            df = pd.read_csv(f, usecols=['time', 'air_temperature'])
            df['sensor_id'] = sid  # 打上标签，用于合并
            dfs.append(df)
        else:
            print(f"Warning: File {f} missing.")

    if not dfs:
        print("Error: No sensor data loaded.")
        return

    full_data = pd.concat(dfs)

    # --- [关键修复] C. 合并坐标信息 ---
    # 将 x, y, z 根据 sensor_id 拼接到时序数据上
    print("Merging coordinates...")
    full_data = pd.merge(full_data, coords_info, on='sensor_id', how='left')

    # 清洗数据：去除空值
    full_data = full_data.dropna(subset=['x', 'y', 'z', 'air_temperature', 'time'])

    # --- D. 归一化处理 ---
    print("Normalizing data...")
    # 1. 坐标归一化
    coords_scaler = MinMaxScaler()
    full_data[['x', 'y', 'z']] = coords_scaler.fit_transform(full_data[['x', 'y', 'z']])

    # 2. 时间归一化 (转为 0-1)
    full_data['time'] = pd.to_datetime(full_data['time'])
    time_min = full_data['time'].min()
    # 计算相对于起始时间的总秒数
    full_data['t_norm'] = (full_data['time'] - time_min).dt.total_seconds()

    t_scaler = MinMaxScaler()
    full_data[['t_norm']] = t_scaler.fit_transform(full_data[['t_norm']])

    # 3. 温度归一化
    temp_scaler = MinMaxScaler()
    full_data[['temp_norm']] = temp_scaler.fit_transform(full_data[['air_temperature']])

    # --- E. 划分训练/测试集 ---
    # 策略：按时间排序，取最后 10% 作为测试集（与主模型保持一致）
    # 但由于数据量太大(几百万行)，PINN 直接训练会非常慢，这里进行采样

    # 先排序
    full_data = full_data.sort_values('t_norm')

    # 找到切分点 (90% 处)
    split_idx = int(len(full_data) * 0.9)

    train_pool = full_data.iloc[:split_idx]
    test_pool = full_data.iloc[split_idx:]

    # [采样] 为了演示和速度，从训练池中随机采样 N 个点
    # 如果显卡够强，可以适当增加 n
    train_df = train_pool.sample(n=300000, random_state=42)
    # 测试集不采样，或者采样大一点
    test_df = test_pool

    print(f"Train samples: {len(train_df)} | Test samples: {len(test_df)}")

    # 转 Tensor 工具函数
    def to_tensor(df):
        x = torch.tensor(df['x'].values, dtype=torch.float32).unsqueeze(1).to(device)
        y = torch.tensor(df['y'].values, dtype=torch.float32).unsqueeze(1).to(device)
        z = torch.tensor(df['z'].values, dtype=torch.float32).unsqueeze(1).to(device)
        t = torch.tensor(df['t_norm'].values, dtype=torch.float32).unsqueeze(1).to(device)
        temp = torch.tensor(df['temp_norm'].values, dtype=torch.float32).unsqueeze(1).to(device)
        return x, y, z, t, temp

    x_train, y_train, z_train, t_train, temp_train = to_tensor(train_df)

    # 这里的 test 数据量可能很大，一次性推理可能爆显存，建议用无梯度模式
    # x_test, y_test, z_test, t_test, temp_test_true = to_tensor(test_df)

    # --- F. 训练模型 ---
    model = PurePINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # ==========================================
    # 🌟 核心开关：切换 Pure INR 和 Standard PINN
    # ==========================================
    USE_PDE = True  # 设为 False 跑 Pure INR，设为 True 跑 Standard PINN
    ALPHA = 0.05  # 热扩散系数
    model_name = "Standard PINN" if USE_PDE else "Pure INR"
    save_name = 'standard_pinn_checkpoint.pth' if USE_PDE else 'pure_inr_checkpoint.pth'

    print(f"Start Training {model_name}...")

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        # 开启梯度追踪 (针对 Standard PINN)
        x_train.requires_grad_(True)
        y_train.requires_grad_(True)
        z_train.requires_grad_(True)
        t_train.requires_grad_(True)

        pred = model(x_train, y_train, z_train, t_train)
        loss_data = criterion(pred, temp_train)

        if USE_PDE:
            # Standard PINN 物理残差计算
            grads = torch.autograd.grad(outputs=pred, inputs=[x_train, y_train, z_train, t_train],
                                        grad_outputs=torch.ones_like(pred), create_graph=True)
            dT_dx, dT_dy, dT_dz, dT_dt = grads[0], grads[1], grads[2], grads[3]

            d2T_dx2 = torch.autograd.grad(dT_dx, x_train, torch.ones_like(dT_dx), create_graph=True)[0]
            d2T_dy2 = torch.autograd.grad(dT_dy, y_train, torch.ones_like(dT_dy), create_graph=True)[0]
            d2T_dz2 = torch.autograd.grad(dT_dz, z_train, torch.ones_like(dT_dz), create_graph=True)[0]

            laplacian = d2T_dx2 + d2T_dy2 + d2T_dz2
            pde_residual = dT_dt - ALPHA * laplacian
            loss_pde = torch.mean(pde_residual ** 2)

            loss = loss_data + 0.1 * loss_pde  # 混合 Loss
        else:
            loss = loss_data

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 加个防爆盾
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} Loss: {loss.item():.6f}")

    # --- G. 评估 ---
    print("\nEvaluating on Test Set...")
    model.eval()

    # 批量推理测试集，防止显存溢出
    batch_size = 10000
    all_preds = []
    all_trues = []

    # 提取测试数据 numpy 数组
    test_x_np = test_df['x'].values
    test_y_np = test_df['y'].values
    test_z_np = test_df['z'].values
    test_t_np = test_df['t_norm'].values
    test_temp_np = test_df['temp_norm'].values  # 归一化的真值

    num_test = len(test_df)

    with torch.no_grad():
        for i in range(0, num_test, batch_size):
            end = min(i + batch_size, num_test)

            # 构造小批次
            bx = torch.tensor(test_x_np[i:end], dtype=torch.float32).unsqueeze(1).to(device)
            by = torch.tensor(test_y_np[i:end], dtype=torch.float32).unsqueeze(1).to(device)
            bz = torch.tensor(test_z_np[i:end], dtype=torch.float32).unsqueeze(1).to(device)
            bt = torch.tensor(test_t_np[i:end], dtype=torch.float32).unsqueeze(1).to(device)

            # 推理
            b_pred = model(bx, by, bz, bt)
            all_preds.append(b_pred.cpu().numpy())
            all_trues.append(test_temp_np[i:end])

    # 拼接结果
    pred_norm = np.concatenate(all_preds, axis=0)
    true_norm = np.concatenate(all_trues, axis=0)

    # 反归一化
    # 注意：sklearn 的 inverse_transform 需要 (N, 1) 形状
    if pred_norm.ndim == 1: pred_norm = pred_norm.reshape(-1, 1)
    if true_norm.ndim == 1:
        true_norm = true_norm.reshape(-1, 1)  # 这里 true_norm 可能已经是 (N,) 了
    else:
        true_norm = true_norm.reshape(-1, 1)

    pred_real = temp_scaler.inverse_transform(pred_norm)
    true_real = temp_scaler.inverse_transform(true_norm)

    # 计算指标
    rmse = mean_squared_error(true_real, pred_real, squared=False)
    mae = mean_absolute_error(true_real, pred_real)
    r2 = r2_score(true_real, pred_real)

    print("\n" + "=" * 40)
    print("📊 [对比实验 ②] 纯 PINN 最终结果")
    print("=" * 40)
    print(f"🔹 Pure PINN RMSE : {rmse:.4f}")
    print(f"🔹 Pure PINN MAE  : {mae:.4f}")
    print(f"🔹 Pure PINN R²   : {r2:.4f}")
    print("=" * 40)

    torch.save(model.state_dict(), f'./{save_name}')
    print(f"✅ {model_name} 权重已保存至 ./{save_name}")


if __name__ == '__main__':
    train_and_evaluate_pinn()
