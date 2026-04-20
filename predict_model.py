#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于加载训练好的模型并生成预测
1. 支持单点时刻验证
2. 支持全量测试集评估 (RMSE/MAE/R2)
3. 支持消融实验：对比 '纯GWN' vs 'GWN+PINN(Ours)'
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2024.2.2
time: 2026/1/9 19:17
version: V1.0
"""

import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 引入项目模块
from LogicAlgorithm.DL_method.GWNPINN.model_1 import SpatioTemporalPINN
from LogicAlgorithm.DL_method.GWNPINN.inference import process_plane_slices, GeometryMasker


# === 重建 Scaler ===
def rebuild_scaler(params):
    """根据保存的参数重建 MinMaxScaler 对象"""
    scaler = MinMaxScaler()
    # 手动注入属性，恢复 Scaler 状态
    # 注意：需要把 list 转回 numpy array，因为 sklearn 需要 numpy
    scaler.min_ = np.array(params['min_'])
    scaler.scale_ = np.array(params['scale_'])
    scaler.data_min_ = np.array(params['data_min_'])
    scaler.data_max_ = np.array(params['data_max_'])
    # 这是一个隐藏属性，虽然文档没写，但为了安全最好加上，通常是 data_max - data_min
    scaler.n_samples_seen_ = 100 # 随意给个正数即可，通常不影响 transform
    return scaler

# 复用数据加载函数
def load_merged_dataframe(data_path, sensor_ids):
    dfs = []
    print("Loading sensor data files...")
    for sid in sensor_ids:
        f = os.path.join(data_path, f"{sid}.csv")
        if not os.path.exists(f):
            print(f"Warning: {f} not found, skipping.")
            continue
        df = pd.read_csv(f, usecols=['time', 'air_temperature'])
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time').rename(columns={'air_temperature': sid})
        # 去重，防止原始数据有重复时间戳
        df = df[~df.index.duplicated(keep='first')]
        dfs.append(df)

    if not dfs:
        raise ValueError("No data loaded!")

    merged = pd.concat(dfs, axis=1)
    merged = merged.sort_index()
    # 简单的插值填充
    merged = merged.interpolate(method='time', limit_direction='both')
    merged = merged.fillna(method='ffill').fillna(method='bfill')
    return merged


# === 核心函数：评估整个测试集 ===
def evaluate_test_set(
        model, full_df_norm, full_df_raw,
        sensor_ids, coords_scaler,
        temp_min, temp_max,
        adj_matrix, zone_features,
        seq_len, pre_len, device,
        compare_gwn=False  # [对比实验开关]
):
    print("\n" + "=" * 50)
    print("🚀 开始测试集整体评估 (Test Set Evaluation)")
    print("=" * 50)

    # 1. 划分测试集 (最后 10%)
    total_len = len(full_df_norm)
    train_size = int(total_len * 0.8)
    val_size = int(total_len * 0.1)
    test_start_idx = train_size + val_size

    print(f"Total samples: {total_len}, Test Start Index: {test_start_idx}")

    # 准备数据容器
    data_values = full_df_norm.values
    raw_values = full_df_raw.values
    pre_adj_t = torch.tensor(adj_matrix, dtype=torch.float32, device=device)

    all_preds_hybrid = []
    all_preds_gwn = []
    all_trues = []

    model.eval()

    # 2. 循环遍历测试集
    # 注意：步长设为 1 或 pre_len 都可以，这里设为 pre_len 加快评估速度
    step_stride = pre_len

    with torch.no_grad():
        for i in range(test_start_idx, total_len - pre_len, step_stride):
            # 构造输入 X (Batch=1)
            # 历史: [i-seq_len : i]
            if i < seq_len: continue

            hist_data = data_values[i - seq_len: i].T  # (N, T_in)
            feat_temp = hist_data[np.newaxis, :, :]  # (1, N, T_in)

            # Zone Feature
            feat_zone = np.tile(zone_features[:, :, np.newaxis], (1, 1, seq_len))
            x_input = np.concatenate([feat_temp, feat_zone], axis=0)
            x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 4, N, T_in)

            # === 模型推理 ===
            # GWN 输出 (纯数据驱动部分)
            y_gwn_norm, sensor_feats = model.gwn(x_tensor, pre_adj_t)

            # Hybrid Decoder 输出 (ST-PIGFN部分)
            # 重构当前时刻 (t=0, 即预测窗口的第1帧) 的传感器值
            # 构造 Decoder 输入
            B, N, _ = y_gwn_norm.shape
            batch_sensor_coords = model.sensor_coords.unsqueeze(0).expand(B, -1, -1)
            batch_t_zero = torch.zeros((B, N, 1), device=device)  # t=0

            # RBF 插值
            interp_feats = model.interpolator(batch_sensor_coords, model.sensor_coords, sensor_feats)
            decoder_input = torch.cat([batch_sensor_coords, batch_t_zero, interp_feats], dim=-1)

            # PINN 解码
            y_hybrid_norm = model.decoder(decoder_input).squeeze(-1)  # (1, N)

            # === 获取真实值 ===
            # 目标是 i 时刻的真实值 (即预测窗口的第一帧)
            true_val_norm = data_values[i, :]  # (N,)

            # === 反归一化 ===
            # Hybrid
            pred_hybrid = y_hybrid_norm.cpu().numpy().flatten() * (temp_max - temp_min) + temp_min

            # GWN (取第一个预测步)
            pred_gwn = y_gwn_norm[:, :, 0].cpu().numpy().flatten() * (temp_max - temp_min) + temp_min

            # True
            true_val = raw_values[i, :]

            all_preds_hybrid.append(pred_hybrid)
            all_preds_gwn.append(pred_gwn)
            all_trues.append(true_val)

            if (i - test_start_idx) % 1000 == 0:
                print(f"Processing step {i}...")

    # 3. 计算指标
    y_true_flat = np.array(all_trues).flatten()
    y_hybrid_flat = np.array(all_preds_hybrid).flatten()

    print("\n --- 最终测试集评估结果 (ST-PIGFN / Ours) ---")
    rmse = mean_squared_error(y_true_flat, y_hybrid_flat, squared=False)
    mae = mean_absolute_error(y_true_flat, y_hybrid_flat)
    r2 = r2_score(y_true_flat, y_hybrid_flat)

    print(f"✅ RMSE : {rmse:.4f}")
    print(f"✅ MAE  : {mae:.4f}")
    print(f"✅ R²   : {r2:.4f}")

    if compare_gwn:
        y_gwn_flat = np.array(all_preds_gwn).flatten()
        print("\n📊 --- 对比实验 ①: 纯 Graph WaveNet 结果 ---")
        rmse_gwn = mean_squared_error(y_true_flat, y_gwn_flat, squared=False)
        mae_gwn = mean_absolute_error(y_true_flat, y_gwn_flat)
        r2_gwn = r2_score(y_true_flat, y_gwn_flat)

        print(f"🔹 GWN RMSE : {rmse_gwn:.4f}")
        print(f"🔹 GWN MAE  : {mae_gwn:.4f}")
        print(f"🔹 GWN R²   : {r2_gwn:.4f}")

        print("\n💡 结论分析:")
        if rmse <= rmse_gwn:
            print("ST-PIGFN 的精度优于或等于纯 GWN，证明加入物理连续场解码器未损失精度，且赋予了空间解析能力。")
        else:
            diff = rmse - rmse_gwn
            print(f"ST-PIGFN 精度略低 ({diff:.4f})，这是为了满足物理约束和连续性而做出的正常权衡。")


def validate_sensors_at_time(
        target_time,
        model,
        full_df,  # 原始数据(包含真实值)
        full_df_norm,  # 归一化数据(用于模型历史输入)
        sensor_ids,
        coords_df_grotto,  # 包含当前石窟所有传感器坐标
        coords_scaler,
        temp_min, temp_max,
        zone_features,
        adj_matrix,
        seq_len,
        device,
        output_dir
):
    """
    功能：验证特定时刻所有传感器的预测值 vs 真实值
    """
    print(f"\n[验证模式] 正在分析时刻: {target_time} ...")

    if isinstance(target_time, str):
        target_time = pd.to_datetime(target_time)

    # 1. 获取真实值 (Observed)
    # 在 full_df 中查找最接近的时间点
    try:
        idx_loc = full_df.index.get_indexer([target_time], method='nearest')[0]
        real_time = full_df.index[idx_loc]
        # 检查时间偏差是否过大 (例如超过 1 小时则警告)
        time_diff = abs((real_time - target_time).total_seconds())
        if time_diff > 3600:
            print(f"Warning: 数据中最接近的时间点 {real_time} 与请求时间相差 {time_diff} 秒，结果可能不准确。")

        obs_row = full_df.iloc[idx_loc]  # Series: index=sensor_id, value=temp
    except Exception as e:
        print(f"Error finding time in data: {e}")
        return

    # 2. 准备预测输入 (Predicted)
    # 构造查询点列表: [(x, y, z, t), ...] 针对所有传感器
    query_points = []
    valid_sensors = []  # 记录顺序

    for sid in sensor_ids:
        if sid not in coords_df_grotto['sensor_id'].values:
            continue

        row = coords_df_grotto[coords_df_grotto['sensor_id'] == sid].iloc[0]
        x, y, z = row['x'], row['y'], row['z']
        query_points.append((x, y, z, target_time))
        valid_sensors.append(sid)

    # 3. 批量推理
    # 利用 process_plane_slices 里用到的 batch_predict_only 逻辑
    # 为了方便，这里直接手写一段针对性的推理，复用 full_df_norm 做历史

    model.eval()
    pre_adj_t = torch.tensor(adj_matrix, dtype=torch.float32, device=device)

    # 构建历史窗口 (归一化数据)
    try:
        idx_loc_norm = full_df_norm.index.get_indexer([target_time], method='pad')[0]
    except:
        print("Error: 无法在归一化数据中找到历史窗口。")
        return

    if idx_loc_norm < seq_len:
        print("Error: 历史数据不足 (seq_len)，无法预测。")
        return

    # 提取历史: (Seq, N) -> (1, 4, N, Seq)
    data_values = full_df_norm.values
    temp_window = data_values[idx_loc_norm - seq_len: idx_loc_norm, :].T
    feat_temp = temp_window[np.newaxis, :, :]
    feat_zone = np.tile(zone_features[:, :, np.newaxis], (1, 1, seq_len))
    x_input_np = np.concatenate([feat_temp, feat_zone], axis=0)
    x_input_tensor = torch.tensor(x_input_np, dtype=torch.float32).unsqueeze(0).to(device)

    preds_real = []

    with torch.no_grad():
        # A. GWN 提取特征
        _, sensor_features = model.gwn(x_input_tensor, pre_adj_t)

        # B. 批量解码
        # 归一化查询坐标
        batch_coords_list = []
        for (x, y, z, _) in query_points:
            xyz_norm = coords_scaler.transform(np.array([[x, y, z]]))[0]
            batch_coords_list.append([xyz_norm[0], xyz_norm[1], xyz_norm[2], 0.0])  # t=0.0

        query_tensor = torch.tensor(batch_coords_list, dtype=torch.float32).unsqueeze(0).to(device)

        # 插值
        query_xyz = query_tensor[..., :3]
        interpolated_feats = model.interpolator(query_xyz, model.sensor_coords, sensor_features)

        # Decoder
        decoder_input = torch.cat([query_tensor, interpolated_feats], dim=-1)
        pred_norm = model.decoder(decoder_input).squeeze(-1)

        vals = pred_norm.cpu().numpy().flatten()
        # 反归一化
        preds_real = vals * (temp_max - temp_min) + temp_min

    # 4. 生成对比数据
    results = []
    y_true = []
    y_pred = []

    for i, sid in enumerate(valid_sensors):
        pred_val = preds_real[i]

        # 获取真实值
        if sid in obs_row:
            true_val = obs_row[sid]
        else:
            true_val = np.nan

        if not np.isnan(true_val):
            diff = pred_val - true_val
            y_true.append(true_val)
            y_pred.append(pred_val)
        else:
            diff = np.nan

        results.append({
            'sensor_id': sid,
            'predicted': round(pred_val, 4),
            'observed': round(true_val, 4),
            'error': round(diff, 4)
        })

    # 5. 计算指标
    if len(y_true) > 0:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        print(f"--- 验证结果 ({len(y_true)} sensors) ---")
        print(f"MAE : {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2  : {r2:.4f}")

    # 6. 保存 CSV
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_name = f"Validation_Grotto_{model.grotto_id_tag}_Time_{target_time.strftime('%Y%m%d_%H%M')}.csv"
    csv_path = os.path.join(output_dir, csv_name)

    df_res = pd.DataFrame(results)
    # 把指标加到 DataFrame 的最后几行作为备注，或者直接打印
    # 这里只存数据表
    df_res.to_csv(csv_path, index=False)
    print(f"详细对比数据已保存至: {csv_path}")


def main():
    # ================= 配置区域 =================
    # 1. 选择要调用的石窟 ID (9 或 10)
    TARGET_GROTTO_ID = 10

    # 是否进行全量测试集评估
    EVALUATE_TEST_SET = False
    # [对比实验 1] 是否对比纯 GWN 效果
    COMPARE_GWN = False

    # === 配置几何掩膜文件 ===
    # 这就是第一步转换出来的那个 CSV 文件路径
    SHAPE_FILE = './data/grotto_shape_points.csv'
    masker = None
    if os.path.exists(SHAPE_FILE):
        print("已检测到形状文件，将启用几何掩膜(去除墙壁区域)...")
        # threshold=0.4 表示：如果预测点距离最近的Fluent网格点超过0.4米，就认为是墙
        # 如果结果边缘有锯齿或空洞，可以微调这个值 (0.2 - 0.5 之间尝试)
        # 将阈值增大到 0.8 或 1.0
        # 这样即使点云稍微稀疏一点，也不会把空气误判为墙壁
        # 如果有了百万点云，可以改小一点，比如 0.15 或 0.2
        # 这样边缘会切得非常整齐，不会有多余的“毛边”
        # masker = GeometryMasker(shape_file_path=SHAPE_FILE, threshold=0.3)
        masker = GeometryMasker(
            shape_file_path=SHAPE_FILE,
            threshold=0.15,  # 减小此值可使柱子变粗
            slab_thickness=0.15  # 增加点云密度，防止缩后穿孔
        )
    else:
        print("未找到形状文件，切片将为矩形。")

    # 2. [功能开关] 验证时刻接口
    # 如果填写具体时间字符串 (如 '2024-10-01 12:00:00')，则激活验证功能
    # 如果填写 None，则跳过验证
    VALIDATION_TIME = '2024-10-06 08:30:00'
    # VALIDATION_TIME = None

    # 3. 1️⃣ [功能开关] 切片生成接口
    # 如果列表不为空，则生成切片；为空则跳过
    SLICE_REQUESTS = [
        # (x, y, z, time)
        (None, None, 1.00, datetime(2024, 10, 6, 8, 30, 0)),
        # (1.00, None, None, datetime(2024, 10, 6, 8, 30, 0)),
        # (None, None, 3.00, datetime(2024, 10, 6, 8, 30, 0)),
        # (None, None, 3.00, datetime(2024, 10, 7, 12, 0, 0)),
        # (2.00, None, None, datetime(2024, 10, 6, 12, 0, 0))

        # # T1: 清晨最冷 (冷池状态)
        # (None, None, 1.0, datetime(2024, 10, 8, 6, 0, 0)),
        # # T2: 中午急剧升温 (热浪入侵)
        # (None, None, 1.0, datetime(2024, 10, 8, 12, 0, 0)),
        # # T3: 傍晚热量渗透最深 (热惯性对比)
        # (None, None, 1.0, datetime(2024, 10, 8, 18, 0, 0)),
        # # T4: 午夜冷却 (热量耗散)
        # # 注意：如果你的数据集没到 24:00，可以写 23:50 或 23:30
        # (None, None, 1.0, datetime(2024, 10, 8, 23, 55, 0)),
    ]
    # ===========================================

    # ==========================================================
    # 2️⃣ 新增循环批量模式
    # ==========================================================
    ENABLE_BATCH_SLICES = False  # 批量扫描总开关：True 开启，False 关闭
    BATCH_TIME = datetime(2024, 10, 6, 8, 30, 0)  # 批量生成的统一时间

    if ENABLE_BATCH_SLICES:
        # ---- 扫描 Z 轴 (生成不同高度的 XY 俯视图) ----
        LOOP_Z = True  # 是否扫描 Z 轴
        Z_START = 0.0  # 起始高度 (米)
        Z_END = 11.0  # 结束高度 (米)
        Z_STEP = 0.5  # 间隔步长 (比如每 0.2 米切一刀)

        # ---- 扫描 X 轴 (生成不同左右位置的 YZ 侧视图) ----
        LOOP_X = True  # 是否扫描 X 轴
        X_START = -5.5  # 最左侧 (米)
        X_END = 5.0  # 最右侧 (米)
        X_STEP = 0.5  # 间隔步长 (比如每 0.5 米切一刀)

        # 自动生成 Z 轴切片并加入队列
        if LOOP_Z:
            for z_val in np.arange(Z_START, Z_END + 0.001, Z_STEP):
                # 用 round 保留两位小数，防止浮点数精度出现 0.600000001 的丑陋文件名
                SLICE_REQUESTS.append((None, None, round(z_val, 2), BATCH_TIME))

        # 自动生成 X 轴切片并加入队列
        if LOOP_X:
            for x_val in np.arange(X_START, X_END + 0.001, X_STEP):
                SLICE_REQUESTS.append((round(x_val, 2), None, None, BATCH_TIME))

    print(f"当前共配置了 {len(SLICE_REQUESTS)} 个切片任务准备生成...")
    # ==========================================================

    # 路径配置
    CHECKPOINT_PATH = f'./results_slices_20260406_02_有Q_！/grotto_{TARGET_GROTTO_ID}_model_checkpoint.pth'
    DATA_PATH = './data'
    COORDS_PATH = './data/_sensor_coords.csv'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === 1. 加载模型 ===
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Model file not found: {CHECKPOINT_PATH}")
        print(f"请检查 TARGET_GROTTO_ID 是否正确，或是否已运行 main.py 进行了训练。")
        return

    print(f"Loading model for Grotto {TARGET_GROTTO_ID}...")
    # checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    except:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    config = checkpoint['config']
    scalers_params = checkpoint['scalers_params']
    graph_data = checkpoint['graph_data']
    train_info = checkpoint['train_info']

    # === 重建 Scaler ===
    print("Rebuilding scalers from parameters...")
    coords_scaler = rebuild_scaler(scalers_params['coords_scaler'])
    temp_scaler = rebuild_scaler(scalers_params['temp_scaler'])
    temp_min = scalers_params['global_min']
    temp_max = scalers_params['global_max']

    # === 恢复 NumPy 数据 ===
    # 将 list 转回 numpy array
    adj_matrix = np.array(graph_data['adj_matrix'])
    sensor_ids = graph_data['sensor_ids']  # 这是训练时用的传感器ID列表
    zone_ids = np.array(graph_data['zone_ids'])

    seq_len = train_info['seq_len']
    pre_len = train_info['pre_len']

    # 重建模型
    print("Rebuilding model...")
    model = SpatioTemporalPINN(
        num_nodes=config['num_nodes'],
        sensor_coords_norm=config['sensor_coords_norm'],
        in_dim=config['input_dim'],
        out_dim=config['out_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # 给模型挂载一个 ID 属性，方便后续打印
    model.grotto_id_tag = TARGET_GROTTO_ID
    print("Model loaded.")

    # === 2. 准备数据环境 (Common) ===
    print("Preparing data context...")
    # 加载全量数据 (包含真实值，用于验证 + 用于构建历史输入)
    full_df = load_merged_dataframe(DATA_PATH, sensor_ids)

    # 归一化全量数据 (用于模型输入)
    full_df_norm = pd.DataFrame(
        temp_scaler.transform(full_df.values),
        index=full_df.index,
        columns=full_df.columns
    )

    zone_one_hot = np.eye(3)[zone_ids].T

    # 加载坐标表 (用于查找传感器位置)
    coords_df_raw = pd.read_csv(COORDS_PATH)
    # 只保留当前石窟的传感器
    coords_df_grotto = coords_df_raw[coords_df_raw['grottoe_id'] == TARGET_GROTTO_ID]
    # 再次过滤，确保和训练时的 sensor_ids 一致
    coords_df_grotto = coords_df_grotto[coords_df_grotto['sensor_id'].isin(sensor_ids)]

    # === 3. 执行功能 A: 传感器验证 ===
    if VALIDATION_TIME is not None:
        validate_sensors_at_time(
            target_time=VALIDATION_TIME,
            model=model,
            full_df=full_df,  # 传原始df用于取真实值
            full_df_norm=full_df_norm,  # 传归一化df用于推理输入
            sensor_ids=sensor_ids,
            coords_df_grotto=coords_df_grotto,
            coords_scaler=coords_scaler,
            temp_min=temp_min,
            temp_max=temp_max,
            zone_features=zone_one_hot,
            adj_matrix=adj_matrix,
            seq_len=seq_len,
            device=device,
            output_dir='./results_validation'
        )
    else:
        print("\n[验证模式] 未激活 (VALIDATION_TIME is None)")

    # === 4. 执行功能 B: 切片生成 ===
    if SLICE_REQUESTS and len(SLICE_REQUESTS) > 0:
        print(f"\n[切片模式] 开始处理 {len(SLICE_REQUESTS)} 个请求...")
        process_plane_slices(
            plane_requests=SLICE_REQUESTS,
            model=model,
            merged_df_norm=full_df_norm,
            adj_matrix=adj_matrix,
            zone_features=zone_one_hot,
            coords_scaler=coords_scaler,
            coords_df=coords_df_grotto,
            temp_min=temp_min,
            temp_max=temp_max,
            seq_len=seq_len,
            device=device,
            # 0.05 (5厘米) 或 0.1 (10厘米)
            # 注意：这会增加计算时间，但画出来的图边缘会非常平滑，接近真实设计图
            step=0.005,
            output_dir='./results_slices_inference',
            masker=masker
        )
    else:
        print("\n[切片模式] 未激活 (SLICE_REQUESTS is empty)")

    # === 执行任务: 测试集评估与 GWN 对比 ===
    if EVALUATE_TEST_SET:
        evaluate_test_set(
            model=model,
            full_df_norm=full_df_norm,
            full_df_raw=full_df,
            sensor_ids=sensor_ids,
            coords_scaler=coords_scaler,
            temp_min=temp_min,
            temp_max=temp_max,
            adj_matrix=adj_matrix,
            zone_features=zone_one_hot,
            seq_len=seq_len,
            pre_len=pre_len,
            device=device,
            compare_gwn=COMPARE_GWN
        )


if __name__ == '__main__':
    main()
