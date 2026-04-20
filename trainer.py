#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于训练模块
封装了训练循环，负责计算混合损失并更新模型
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2024.2.2
time: 2025/9/7 13:27
version: V1.0
"""

# trainer.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# from torch.cuda.amp import autocast, GradScaler
# 兼容性逻辑
try:
    # 尝试从新版 torch.amp 导入 (PyTorch 2.4+ 标准)
    # 注意：必须同时成功导入 autocast 和 GradScaler 才算成功
    from torch.amp import autocast, GradScaler
    USE_NEW_AMP = True
except ImportError:
    # 如果失败（说明是旧版本或过渡版本），回退到 torch.cuda.amp
    # 这是绝大多数现有 PyTorch 版本 (1.6 ~ 2.3) 的标准路径
    from torch.cuda.amp import autocast, GradScaler
    USE_NEW_AMP = False
# --------------------------------------------------


def masked_mse_loss(preds, labels, mask):
    """
    只计算非缺失值的 MSE Loss
    preds: (B, N, T)
    labels: (B, N, T)
    mask: (B, N, T)  1=Valid, 0=Missing
    """
    loss = (preds - labels) ** 2
    loss = loss * mask
    # 避免除以 0
    return loss.sum() / (mask.sum() + 1e-8)


def evaluate_model_on_loader(model, data_loader, pre_adj, device):
    """
    在给定 data_loader 上评估模型，返回 (MAE, RMSE, R2).
    评估时只使用 GraphWaveNet 部分（model.gwn），避免触发 PINN 的 autograd。
    """
    model.eval()
    preds = []
    trues = []

    # 确保 pre_adj 是 tensor
    if not isinstance(pre_adj, torch.Tensor):
        pre_adj_t = torch.tensor(pre_adj, dtype=torch.float32, device=device)
    else:
        pre_adj_t = pre_adj.to(device)

    with torch.no_grad():
        # 解包 mask (虽然验证时通常看所有点，但严谨来说应该只看真实存在的点)
        for xb, yb, mask_b, ambient_b in data_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mask_b = mask_b.to(device)

            # (B, N, T) -> (B, C, N, T) 如果需要的话
            # ensure channel dim: (B, 1, N, Seq_len)
            # data_loader 返回 4通道数据 (B, 4, N, T)，可能不需要 unsqueeze
            # 为了旧版本兼容性，保留判断
            if xb.dim() == 3:
                xb = xb.unsqueeze(1)

            # 直接用 GraphWaveNet 做预测，避免执行 PINN 解码器与自动求导
            # model.gwn 接收 (x, pre_adj) 并返回 y_pred_gwn——新版本变成→(y_pred, node_features)
            gwn_output = model.gwn(xb, pre_adj_t)

            if isinstance(gwn_output, tuple):
                y_pred_gwn, _ = gwn_output  # 只取第一个返回值 (预测值)，丢弃特征
            else:
                y_pred_gwn = gwn_output  # 兼容旧版本
            # y_pred_gwn = model.gwn(xb, pre_adj_t)

            # 过滤掉缺失值的评估（可选，或者评估时只看 mask=1 的点）
            # 这里为了简单，还是记录所有预测值，但在计算 metrics 时最好过滤
            # 为了兼容 sklearn，先把数据拉平，然后利用 mask 筛选

            pred_np = y_pred_gwn.detach().cpu().numpy().reshape(-1)
            true_np = yb.detach().cpu().numpy().reshape(-1)
            mask_np = mask_b.detach().cpu().numpy().reshape(-1)

            # ====== 【NaN 防御拦截】 ======
            if np.isnan(pred_np).any():
                print("\n[严重警告] 验证阶段发现模型预测输出 NaN！模型可能已经崩溃。")
                return float('nan'), float('nan'), float('nan')
            # ========================================

            # 只保留 mask=1 的点进行评估
            valid_indices = mask_np > 0.5
            if valid_indices.sum() > 0:
                preds.append(pred_np[valid_indices])
                trues.append(true_np[valid_indices])

    if len(preds) == 0:
        return float('nan'), float('nan'), float('nan')

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    mae = mean_absolute_error(trues, preds)

    # rmse = mean_squared_error(trues, preds, squared=False)
    # 高版本python需要 计算 MSE，然后手动计算 RMSE
    # mse = mean_squared_error(trues, preds)
    # rmse = np.sqrt(mse)  # 手动计算 RMSE
    try:
        rmse = mean_squared_error(trues, preds, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(trues, preds))

    r2 = r2_score(trues, preds)

    return mae, rmse, r2


def train(model, train_loader, pre_adj, coords_df_normalized,
          boundary_coords=None,   # 接收边界坐标
          epochs=10, lr=0.0005, lambda_pde=0.1,
          lambda_bc=0.5,   # 接收BC权重
          lambda_recon=1.0,  # 解码器重构损失权重
          tau=5.0, val_loader=None, test_loader=None, device=None):
    """
    Minimal-change training loop that also evaluates on val_loader per epoch.
    - val_loader and test_loader are optional DataLoader objects (created in main.py).
    - pre_adj: adjacency matrix (numpy or tensor) used by gwn.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history = {
        'epoch': [], 'total_loss': [], 'loss_data': [],
        'loss_recon': [], 'loss_pde': [], 'loss_bc': []
    }

    # --- 强制预热 CUDA 上下文，解决 cuBLAS 警告 ---
    # 在进行任何复杂计算前，先在 GPU 上做一个简单的运算，激活 cuBLAS 句柄
    if device.type == 'cuda':
        torch.zeros(1).to(device)
    # ----------------------------------------------------

    # 开启异常检测 (调试用，稳定后可关闭)
    # torch.autograd.set_detect_anomaly(True)

    # scaler = GradScaler()  # 初始化 scaler
    # --- 兼容性初始化 Scaler ---
    if USE_NEW_AMP:
        # 高版本：必须指定 device
        scaler = GradScaler('cuda')
    else:
        # 低版本：默认就是 cuda
        scaler = GradScaler()
    # ----------------------------------

    # 如果有边界点，转到 device
    if boundary_coords is not None:
        boundary_coords = boundary_coords.to(device)  # (N_bound, 3)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 不再使用全局 criterion = nn.MSELoss()，改用 masked_mse_loss
    # criterion = torch.nn.MSELoss()

    # ensure pre_adj is tensor on device
    # pre_adj_t = torch.tensor(pre_adj, dtype=torch.float32, device=device)
    if not isinstance(pre_adj, torch.Tensor):
        pre_adj_t = torch.tensor(pre_adj, dtype=torch.float32, device=device)
    else:
        pre_adj_t = pre_adj.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        loss_data_sum = 0.0
        loss_recon_sum = 0.0
        loss_bc_sum = 0.0
        loss_pde_sum = 0.0
        batch_count = 0

        # 解包 (x, y, mask), ambient
        for x_batch, y_batch, mask_batch, ambient_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)  # (B, N, T)
            ambient_batch = ambient_batch.to(device)  # (B, 1)

            # ensure channel dim for GWN: (B,1,N,Seq_len)
            if x_batch.dim() == 3:
                x_batch = x_batch.unsqueeze(1)

            optimizer.zero_grad()

            # --- 兼容性 Autocast 上下文 ---
            # 高版本需要 device_type='cuda'，低版本不需要
            # 利用一个临时变量来处理参数
            amp_args = {'device_type': 'cuda'} if USE_NEW_AMP else {}

            # 使用 autocast 上下文
            with autocast(**amp_args):

                # 运行 GWN (我们需要 features 来进行插值)
                # model.gwn 返回 (y_pred, features)
                y_pred_gwn, sensor_features = model.gwn(x_batch, pre_adj_t)

                # Data Loss
                loss_data = masked_mse_loss(y_pred_gwn, y_batch, mask_batch)

                # --- Loss B: Decoder 重构损失 (监督 Decoder) ---
                # 目的：强制 Decoder 在传感器位置的输出 = 真实温度
                # 取预测窗口的第1个时间步 (t=0) 进行对齐
                B, N, _ = y_batch.shape

                # 构造输入:
                # Coords: (B, N, 3) -> 传感器位置
                batch_sensor_coords = model.sensor_coords.unsqueeze(0).expand(B, -1, -1)
                # Time: (B, N, 1) -> t=0.0 (代表当前/下一时刻)
                batch_t_zero = torch.zeros((B, N, 1), device=device)

                # RBF 插值 (虽然是在传感器原位插值，但经过RBF平滑是必要的)
                interp_feats_sensors = model.interpolator(batch_sensor_coords, model.sensor_coords, sensor_features)

                # Decoder 预测
                decoder_input_sensors = torch.cat([batch_sensor_coords, batch_t_zero, interp_feats_sensors], dim=-1)
                pred_sensor_temp = model.decoder(decoder_input_sensors).squeeze(-1)  # (B, N)

                # 目标: y_batch 的第1个时间步
                target_sensor_temp = y_batch[:, :, 0]  # (B, N)
                mask_sensor = mask_batch[:, :, 0]  # (B, N)

                loss_recon = masked_mse_loss(pred_sensor_temp, target_sensor_temp, mask_sensor)
                # ----------------------------------------------------

                # sample PDE collocation points if your model requires; keep same logic as before
                num_pde_points = 512
                coords_arr = np.asarray(coords_df_normalized)
                min_bounds = torch.tensor(coords_arr.min(axis=0), dtype=torch.float32, device=device)
                max_bounds = torch.tensor(coords_arr.max(axis=0), dtype=torch.float32, device=device)
                pde_coords = torch.rand(num_pde_points, 3, device=device) * (max_bounds - min_bounds) + min_bounds
                pde_t = torch.rand(num_pde_points, 1, device=device)
                pde_coords_t = torch.cat([pde_coords, pde_t], dim=1).to(device)  # (Np, 4) or (B, Np, 4)

                # 调用 model forward 计算残差 (SpatiotemporalPINN 内部会处理 PDE 导数)
                # 注意：forward 内部也会跑一次 gwn
                # 如果追求极致性能，可以修改 forward 接收外部传入的 features
                # (自动扩展 pde_coords_t batch 维)
                _, pde_residual = model(x_batch, pre_adj_t, pde_coords_t,
                                        sensor_features_precalc=sensor_features)

                # ================= 空间自适应物理松弛 =================
                # 1. 拼接输入历史和预测目标，寻找局部时间窗口内的最大突变
                # x_batch shape: (B, 4, N, Seq) -> 索引 0 是温度特征
                history_temp = x_batch[:, 0, :, :]  # (B, N, Seq_len)
                full_temp_seq = torch.cat([history_temp, y_batch], dim=2)  # (B, N, Seq_len + Pre_len)

                # 2. 计算相邻时间步的温差绝对值
                temporal_diff = torch.abs(full_temp_seq[:, :, 1:] - full_temp_seq[:, :, :-1])
                max_jump_per_node, _ = torch.max(temporal_diff, dim=2)  # (B, N)

                # 3. 计算节点级松弛权重 (指数衰减)
                # 因为温度已经归一化到 0~1，0.7 的剧烈降温在 tau=5.0 下权重会掉到 0.03 (几乎取消物理约束)
                # 0.05 的正常波动在 tau=5.0 下权重保持在 0.77 (物理约束依然强力)
                # tau = 5.0
                node_relaxation = torch.exp(-tau * max_jump_per_node).unsqueeze(-1).detach()  # (B, N, 1)

                # 4. 利用 RBF 插值器，将传感器上的松弛权重，映射到 PDE 点上
                # pde_coords_t 在 trainer.py 是 2D (N_pde, 4)，先扩展为 3D (B, N_pde, 4)
                batch_pde_coords = pde_coords_t.unsqueeze(0).expand(B, -1, -1)
                pde_xyz = batch_pde_coords[:, :, :3]  # 现在可以安全地切片了，(B, N_pde, 3)
                pde_relaxation = model.interpolator(pde_xyz, model.sensor_coords, node_relaxation)  # (B, N_pde, 1)

                # 5. 对 PDE 物理残差进行加权并求均值
                # 原本 pde_residual 是 (B * N_pde, 1)，reshape 回 (B, N_pde, 1)
                pde_residual = pde_residual.view(B, num_pde_points, 1)
                # weighted_pde_residual = pde_residual * pde_relaxation
                # loss_pde = torch.mean(weighted_pde_residual ** 2)  # PDE Loss 通常不需要 mask，因为是虚拟点

                # ============== 、使用鲁棒的 Smooth L1 Loss ==============
                weighted_pde_residual = pde_residual * pde_relaxation
                # 用 smooth_l1_loss 替代平方误差
                target_zeros = torch.zeros_like(weighted_pde_residual)
                loss_pde = F.smooth_l1_loss(weighted_pde_residual, target_zeros)

                # ====================================================================

                # loss_pde = torch.mean(pde_residual ** 2)  # PDE Loss 通常不需要 mask，因为是虚拟点

                # BC Loss (边界条件)
                loss_bc = torch.tensor(0.0, device=device)
                if boundary_coords is not None:
                    B = x_batch.size(0)
                    N_bound = boundary_coords.size(0)

                    # A. 构造边界输入
                    # 坐标扩展到 Batch: (B, N_b, 3)
                    batch_bound_coords = boundary_coords.unsqueeze(0).expand(B, -1, -1)

                    # 时间: 假设约束的是 output 时刻 (归一化时间 1.0)
                    batch_t = torch.ones((B, N_bound, 1), device=device)

                    # B. 插值特征 (从 GWN 提取的 sensor_features 插值到 boundary_coords)
                    # input: (B, N_b, 3), (N_s, 3), (B, N_s, 32) -> (B, N_b, 32)
                    interp_feats_bc = model.interpolator(batch_bound_coords, model.sensor_coords, sensor_features)

                    # C. Decoder 预测
                    # input: [x, y, z, t, features]
                    decoder_input_bc = torch.cat([batch_bound_coords, batch_t, interp_feats_bc], dim=-1)
                    pred_bc_temp = model.decoder(decoder_input_bc).squeeze(-1)  # (B, N_b)

                    # D. 计算 MSE: 预测值 vs 外界气温
                    # ambient_batch (B, 1) -> expand to (B, N_b)
                    target_bc = ambient_batch.expand(-1, N_bound)

                    # --- 对边界损失也进行插值松弛 ---
                    bc_relaxation = model.interpolator(batch_bound_coords, model.sensor_coords,
                                                       node_relaxation).squeeze(-1)  # (B, N_b)
                    unreduced_loss_bc = (pred_bc_temp - target_bc) ** 2
                    loss_bc = torch.mean(unreduced_loss_bc * bc_relaxation)
                    # loss_bc = torch.mean((pred_bc_temp - target_bc) ** 2)

                # 数据误差 + pde误差 + BC边界条件误差
                loss = loss_data + lambda_recon * loss_recon + lambda_pde * loss_pde + lambda_bc * loss_bc

            # NaN 检测与跳过
            if torch.isnan(loss):
                print(f"[Warning] Epoch {epoch + 1} Batch {batch_count} Loss is NaN! Skipping...")
                optimizer.zero_grad()  # 清空梯度，不更新
                continue

            # --- 使用 scaler 进行反向传播 (高低版本通用) ---
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # 如果需要 clip_grad
            # 原本是 5.0，现在改为 2.0，严防梯度爆炸/1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            # --------------------------------------------

            total_loss += loss.item()
            loss_data_sum += loss_data.item()
            loss_recon_sum += loss_recon.item()
            loss_bc_sum += loss_bc.item()
            loss_pde_sum += loss_pde.item()
            batch_count += 1

        avg_loss = total_loss / max(1, batch_count)
        avg_data = loss_data_sum / max(1, batch_count)
        avg_recon = loss_recon_sum / max(1, batch_count)
        avg_bc = loss_bc_sum / max(1, batch_count)
        avg_pde = loss_pde_sum / max(1, batch_count)

        # --- validation evaluation ---
        if val_loader is not None:
            mae_val, rmse_val, r2_val = evaluate_model_on_loader(model, val_loader, pre_adj, device)
            val_str = f"Val MAE: {mae_val:.6f}, RMSE: {rmse_val:.6f}, R2: {r2_val:.6f}"
        else:
            val_str = ""

        print(
            f"Epoch {epoch + 1}/{epochs} Loss: {avg_loss:.6f} (GWN:{avg_data:.6f} Recon:{avg_recon:.6f} PDE:{avg_pde:.6f} BC:{avg_bc:.6f}) {val_str}")

        q_status = "Active!" if getattr(model, 'use_q_net', True) else "Disabled (Q=0)"
        print(f"Learned alpha: {model.alpha.item():.6f} | Spatial Q-Net is {q_status}")

        # 在每个 Epoch 的末尾把数据记录进去：
        history['epoch'].append(epoch + 1)
        history['total_loss'].append(avg_loss)
        history['loss_data'].append(avg_data)
        history['loss_recon'].append(avg_recon)
        history['loss_pde'].append(avg_pde)
        history['loss_bc'].append(avg_bc)

    # final test evaluation (optional)
    if test_loader is not None:
        mae_test, rmse_test, r2_test = evaluate_model_on_loader(model, test_loader, pre_adj, device)
        print(f"Test MAE: {mae_test:.6f}, RMSE: {rmse_test:.6f}, R2: {r2_test:.6f}")

    return model, history
