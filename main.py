#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于主程序入口
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2024.2.2
time: 2025/9/7 13:22
version: V1.0

grotto_temperature_project/
├────── data/                 # 在外面，存放150个csv文件和sensor_coords.csv
│      ├── a01.csv            # 序号、'time'、'air_temperature'
│      ├── a02.csv
│      └── ...
│      └── sensor_coords.csv  # 'sensor_id'、'x'、'y'、'z'、'zone_id'、'zone_name'
├── main.py               # 主程序入口：整合所有模块，执行训练和推理
├── data_loader.py        # 模块1: 负责数据读取、预处理、构建图、生成PyTorch数据集
├── model_1.py            # 模块2: 定义GraphWaveNet、PINNDecoder和整合后的SpatioTemporalPINN模型
├── trainer.py            # 模块3: 封装训练过程，计算混合损失函数
└── inference.py          # 模块4: 封装推理功能，如查询任意点温度、生成温度场切片
└── utils.py              # 辅助工具模块，例如计算邻接矩阵等
└── compare_interpolation.py # 实现ST-PIGFN 模型与IDW (反距离加权) 和 Kriging (克里金插值) 对比
└── convert_ascii.py      # 此文件用于fluent导出的数据转换 (ASCII -> CSV)
└── evaluate_ablation.py  # 绘制 不同的Z 的对比时序图（敏感性分析）
└── evaluate_sensitivity_Z.py # 绘制 Z 轴敏感性分析柱状图
└── extract_pure_data.py  # 从原始Fluent文件中提取中间很多的细节点（原始，现已不用）
└── fix_data_tool.py      # 从原始Fluent文件中提取中间很多的细节点
└── plot_combined.py      # 合并侧视图和俯视图的效果（共用一根热力轴）
└── plot_figure2.py       # 训练收敛曲线 (Loss components)和 4个传感器在测试集上的连续时序预测对比
└── plot_individual_sensors.py # 单独生成指定传感器的时间序列预测对比图（真实值 vs ST-PIGFN）
└── plot_sensor_trends_sci.py  # 绘制多个传感器的温度时间序列对比图（全真实值）
└── plot_spatiotemporal_sci.py # 1D 时序曲线 + 横向排布 4 张 2D 切片，共享同一个全局热力轴
└── train_baseline_pinn.py     # 标准的 PINN 实现

数据放置: 将150个传感器CSV文件和sensor_coords.csv放入data文件夹。
修改main.py: 在main.py的SENSOR_IDS列表中，填入完整的150个传感器ID。
运行: 运行main.py

"""

# main.py
import os
import json
import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from datetime import datetime
from scipy.spatial.distance import cdist
from LogicAlgorithm.DL_method.GWNPINN.data_loader import load_and_preprocess_data
from LogicAlgorithm.DL_method.GWNPINN.model_1 import SpatioTemporalPINN
from LogicAlgorithm.DL_method.GWNPINN.trainer import train
from LogicAlgorithm.DL_method.GWNPINN.inference import process_plane_slices

# --- helper: load merged dataframe from sensor CSVs (same order as SENSOR_IDS) ---
def load_merged_dataframe(data_path, sensor_ids):
    """
    返回：merged_df (pd.DataFrame) 索引为 pd.DatetimeIndex，列为 sensor_id（与 sensor_ids 顺序一致）
    要求每个 sensor CSV 有 ['time','air_temperature'] 列。
    """
    dfs = []
    for sid in sensor_ids:
        f = os.path.join(data_path, f"{sid}.csv")
        df = pd.read_csv(f, usecols=['time', 'air_temperature'])
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time').rename(columns={'air_temperature': sid})
        dfs.append(df)
    merged = pd.concat(dfs, axis=1)
    # 若有缺失，使用线性时间插值（时间方向）
    merged = merged.sort_index()
    merged = merged.interpolate(method='time', limit_direction='both')
    return merged


# --- helper: get observed temperature at arbitrary (x,y,z,t) via IDW on nearest k sensors ---
def get_observed_temperature_at_point(merged_df, coords_df, x, y, z, t, k=3, time_interp=True):
    """
    merged_df: index datetime, columns sensor_id -> observed temperatures (real units)
    coords_df: dataframe with sensor_id, x,y,z columns (same sensor ids)
    x,y,z: query location
    t: datetime query time
    k: number of nearest sensors to use for IDW
    time_interp: if True, interpolate in time if exact timestamp not present
    返回: observed_temp (float) or np.nan if cannot compute
    """
    # ensure index is datetime
    if not isinstance(merged_df.index, pd.DatetimeIndex):
        merged_df.index = pd.to_datetime(merged_df.index)

    # if time not exactly in index and time_interp True, create interpolated row
    if t in merged_df.index:
        row = merged_df.loc[t]
    else:
        if time_interp:
            # create a small time series reindex that includes t, then interpolate
            # find min/max to not extrapolate beyond data range
            if t < merged_df.index.min() or t > merged_df.index.max():
                # outside observation time range -> return NaN or nearest time value (here use nearest)
                nearest_time = merged_df.index.get_indexer([t], method='nearest')[0]
                row = merged_df.iloc[nearest_time]
            else:
                # linear interpolate with pandas
                tmp = merged_df.copy()
                # insert NaN row at time t
                tmp.loc[t] = np.nan
                tmp = tmp.sort_index()
                tmp = tmp.interpolate(method='time', limit_direction='both')
                row = tmp.loc[t]
        else:
            # use nearest time point
            idx = merged_df.index.get_indexer([t], method='nearest')[0]
            row = merged_df.iloc[idx]

    # row: Series indexed by sensor_id with temperature values
    sensor_ids = list(row.index)
    sensor_vals = row.values.astype(float)

    # compute distances
    coord_mat = coords_df.set_index('sensor_id').loc[sensor_ids][['x', 'y', 'z']].values  # shape (N,3)
    query = np.array([[x, y, z]], dtype=float)
    dists = cdist(coord_mat, query).ravel()  # distances to each sensor

    # if any sensor exactly at query position, return its value (and if its value is nan, still continue)
    zero_mask = (dists == 0.0)
    if zero_mask.any():
        idx0 = np.where(zero_mask)[0][0]
        val = sensor_vals[idx0]
        return float(val) if not np.isnan(val) else np.nan

    # choose k nearest with finite values
    finite_mask = np.isfinite(sensor_vals)
    if not finite_mask.any():
        return np.nan
    dists_finite = dists.copy()
    dists_finite[~finite_mask] = np.inf
    nearest_idx = np.argsort(dists_finite)[:k]
    # remove infinite distances
    nearest_idx = [i for i in nearest_idx if np.isfinite(dists_finite[i])]
    if len(nearest_idx) == 0:
        return np.nan

    chosen_vals = sensor_vals[nearest_idx]
    chosen_dists = dists_finite[nearest_idx]
    # avoid zero distances (shouldn't happen because we handled exact 0 above)
    # compute IDW weights: w = 1 / (d + eps)^p
    eps = 1e-6
    p = 2.0
    weights = 1.0 / ((chosen_dists + eps) ** p)
    # handle NaNs in chosen_vals by masking
    mask_valid = np.isfinite(chosen_vals)
    if not mask_valid.any():
        return np.nan
    weights = weights[mask_valid]
    vals = chosen_vals[mask_valid]

    temp_obs = float(np.sum(weights * vals) / np.sum(weights))
    return temp_obs


# --- 根据查询时间构建模型输入 ---
def prepare_model_input_for_time(query_time, merged_df, coords_df, seq_len, pre_len,
                                 input_dim=4):
    """
    根据查询时间 t，回溯 seq_len 长度的历史数据，构建模型输入 tensor。
    """
    # 1. 找到查询时间在 dataframe 中的位置
    # merged_df index 必须是 datetime
    if query_time not in merged_df.index:
        # 如果时间点不存在，找最近的一个之前的时间点
        # 这里为了简单，假设 merged_df 是连续且填充好的
        # 实际操作：使用 get_loc 或 searchsorted
        try:
            # 找到最近的索引
            idx = merged_df.index.get_indexer([query_time], method='pad')[0]
        except:
            return None  # 时间太早，没有历史数据
    else:
        idx = merged_df.index.get_loc(query_time)

    # 2. 检查是否有足够的历史数据
    if idx < seq_len:
        print(f"Warning: Not enough history for time {query_time}")
        return None

    # 3. 截取窗口 [idx - seq_len : idx]
    # 这一段是用来预测未来的，所以取“截止到 query_time 之前”的数据作为输入
    # 假设 query_time 是要预测的时刻，那么输入应该是 (query_time - seq_len) 到 query_time
    history_window = merged_df.iloc[idx - seq_len: idx].values  # (Seq_Len, Nodes)

    # 4. 构建 4通道特征 (Temp + 3 Zone One-hot)
    # 这部分逻辑必须和 data_loader 保持完全一致！

    # 温度归一化 (使用全局 scaler)
    # 注意：这里需要拿到 main scope 里的 temp_scaler，或者作为参数传入
    # 为简化，在外部做完归一化再传入 merged_df，或者在这里做
    # 建议：merged_df 传入前最好已经是归一化的，或者在这里用 scaler.transform
    # 这里假设 merged_df 已经是原始温度，我们需要 transform
    # *在主流程里处理归一化*

    pass  # 具体实现在下面的主函数里写更方便


def compare_points_predictions_with_observed(
        points,
        model,
        merged_df_norm,
        adj_matrix,
        zone_features,
        coords_scaler,
        temp_min, temp_max,
        seq_len,
        device,
        k_neighbors=3
):
    model.eval()
    results = []
    pre_adj_t = torch.tensor(adj_matrix, dtype=torch.float32, device=device)

    data_values = merged_df_norm.values if hasattr(merged_df_norm, 'values') else merged_df_norm
    time_index = merged_df_norm.index

    # 1. 按时间分组 (Group by time)
    # 这样同一时刻的 1000 个点只需要构建一次历史输入
    points_by_time = {}
    for p in points:
        t = p[3]
        if t not in points_by_time:
            points_by_time[t] = []
        points_by_time[t].append(p)

    print(f"\n开始推理: 共 {len(points)} 个点，聚合为 {len(points_by_time)} 个时间步。")

    with torch.no_grad():
        for t_query, batch_points in points_by_time.items():
            # --- A. 针对该时间点，构建一次历史输入 (x_data) ---
            if isinstance(t_query, str):
                t_query_dt = pd.to_datetime(t_query)
            else:
                t_query_dt = t_query

            try:
                # 找 query 时间点之前的最近索引
                idx_loc = time_index.get_indexer([t_query_dt], method='pad')[0]
            except:
                print(f"时间 {t_query_dt} 超出范围，跳过。")
                continue

            if idx_loc < seq_len:
                print(f"时间 {t_query_dt} 历史数据不足，跳过。")
                continue

            # 提取历史窗口 (Seq, N) -> (1, 4, N, Seq)
            temp_window = data_values[idx_loc - seq_len: idx_loc, :].T
            feat_temp = temp_window[np.newaxis, :, :]
            feat_zone = np.tile(zone_features[:, :, np.newaxis], (1, 1, seq_len))
            x_input_np = np.concatenate([feat_temp, feat_zone], axis=0)
            x_input_tensor = torch.tensor(x_input_np, dtype=torch.float32).unsqueeze(0).to(device)

            # --- B. 运行 GWN 一次 (提取特征) ---
            # sensor_features: (1, N, 32)
            _, sensor_features = model.gwn(x_input_tensor, pre_adj_t)

            # --- C. 批量处理该时间下的所有空间点 ---
            # 解析坐标
            batch_coords_list = []
            for (x, y, z, _) in batch_points:
                xyz_norm = coords_scaler.transform(np.array([[x, y, z]]))[0]
                batch_coords_list.append([xyz_norm[0], xyz_norm[1], xyz_norm[2], 0.0])  # t_norm=0.0 (当前时刻)

            # 构造 Query Tensor: (1, N_points, 4)
            query_tensor = torch.tensor(batch_coords_list, dtype=torch.float32).unsqueeze(0).to(device)

            # --- D. 批量插值与解码 ---
            # 特征插值: (1, N_p, 32)
            query_xyz = query_tensor[..., :3]
            interpolated_feats = model.interpolator(query_xyz, model.sensor_coords, sensor_features)

            # Decoder
            decoder_input = torch.cat([query_tensor, interpolated_feats], dim=-1)
            pred_norm = model.decoder(decoder_input).squeeze(-1)  # (1, N_points)

            # --- E. 保存结果 ---
            pred_vals = pred_norm.cpu().numpy().flatten()

            for i, pred_val in enumerate(pred_vals):
                real_temp = pred_val * (temp_max - temp_min) + temp_min
                orig_p = batch_points[i]
                results.append({
                    'x': orig_p[0], 'y': orig_p[1], 'z': orig_p[2], 't': orig_p[3],
                    'predicted_temp': real_temp
                })

    return results


if __name__ == '__main__':
    # ==============================================================
    # 🔥 Z轴惩罚系数批量测试开关 (Parameter Sensitivity Test) 🔥
    # Z轴高程惩罚系数，用于构建各向异性图拓扑 (推荐范围: 5.0 ~ 20.0)
    # ==============================================================
    RUN_Z_PENALTY_TEST = False  # 专门为3.6节跑数据时设为 True，平时不用设为 False
    Z_PENALTY_LIST = [1.0, 5.0, 10.0, 15.0, 20.0]  # 要测试的 Z 轴惩罚系数列表 (10.0 已经跑过可不加)

    # 如果开启批量测试，就遍历列表；否则就只跑一个默认的 [10.0]
    test_z_values = Z_PENALTY_LIST if RUN_Z_PENALTY_TEST else [10.0]

    for current_z in test_z_values:
        print(f"\n" + "🌟" * 30)
        print(f"🚀 开始自动执行 Z_PENALTY = {current_z} 的模型训练")
        print("🌟" * 30 + "\n")

        # ==========================
        # 🔥 核心：消融实验主控开关 🔥
        # ==========================
        # 'FULL'          : 完整的 ST-PIGFN (带 PDE, 带 BC, 带动态 Q)
        # 'DATA_DRIVEN'   : 纯数据驱动 (强制关闭 PDE 和 BC)
        # 'STATIC_SOURCE' : 静态源 PINN (保留 PDE 和 BC, 但强制 Q=0)
        ABLATION_MODE = 'STATIC_SOURCE'

        # ======== 👇 空间遮蔽测试开关 👇 ========
        ENABLE_SPATIAL_MASKING = True  # 设置为 True 时，隔离 20% 节点用于对比 IDW/Kriging；为 False 时，使用 100% 节点训练
        MASK_RATIO = 0.2  # 遮蔽比例
        # ===============================================

        # 动态匹配模型参数与存储文件夹
        if ABLATION_MODE == 'FULL':
            LAMBDA_PDE = 0.1
            LAMBDA_BC = 0.5  # 边界约束权重
            USE_Q_NET = True
            # SAVE_DIR = './results_slices_FULL'
            # 把 Z 的值拼接到文件夹名字里，防止互相覆盖！
            if RUN_Z_PENALTY_TEST:
                SAVE_DIR = f'./results_slices_FULL_Z_{int(current_z)}'
            else:
                SAVE_DIR = './results_slices_FULL'
        elif ABLATION_MODE == 'DATA_DRIVEN':
            LAMBDA_PDE = 0.0  # 关闭物理方程
            LAMBDA_BC = 0.0  # 关闭边界条件
            USE_Q_NET = False  # 关闭源项反演
            SAVE_DIR = './results_slices_DATA_DRIVEN'
        elif ABLATION_MODE == 'STATIC_SOURCE':
            LAMBDA_PDE = 0.1  # 开启物理方程
            LAMBDA_BC = 0.5  # 开启边界条件
            USE_Q_NET = False  # 【核心消融】：强制源项 Q=0
            SAVE_DIR = './results_slices_STATIC_SOURCE'
        else:
            raise ValueError("未知的消融实验模式！")

        # ==========================================================
        # ======== 👇 根据遮蔽开关动态追加文件夹后缀 👇 ========
        # ==========================================================
        if ENABLE_SPATIAL_MASKING:
            # 加上 _Masked 后缀
            # 例如：results_slices_FULL 会变成 results_slices_FULL_Masked
            SAVE_DIR = f"{SAVE_DIR}_Masked"
        # ==========================================================

        # --- 1. 参数配置 --- (必须是全大写的否则不能写入config.txt)
        DATA_PATH = './data'
        COORDS_PATH = './data/_sensor_coords.csv'

        # ==========================
        # 选择要训练的石窟 ID (9 或 10)
        TARGET_GROTTO_ID = 10
        # ==========================
        # (必须是全大写的否则不能写入config.txt)
        SEQ_LEN = 12 * 3  # 使用3小时数据,代表模型会根据过去x小时的数据来预测未来,模型的“短期记忆”长度
        PRE_LEN = int(12 * 0.5)  # 预测0.5小时数据,非常精准的短期预测,用于实时控制
        BATCH_SIZE = 16
        EPOCHS = 150
        # LAMBDA_PDE = 0.1
        LAMBDA_RECON = 5.0  # 重构误差权重  1.0/5.0/10.0 强制学习锚点信息就增加
        # LAMBDA_BC = 0.5  # 边界约束权重  0.05/0.5/10/20
        Z_PENALTY = current_z  # Z轴高程惩罚系数，用于构建各向异性图拓扑 (推荐范围: 5.0 ~ 20.0)

        TAU_RELAXATION = 10.0  # 物理松弛敏感度：越大代表模型对温度突变越敏感，越容易放弃物理平滑约束
        K_ADJ = 4  # 图邻居数量,每个传感器都要强制听取周围K_ADJ个邻居的意见

        # 核心配置
        INPUT_DIM = 4  # 1 (Temp) + 3 (One-hot Zones)

        print(f"\n==============================================")
        print(f"🚀 当前正运行消融实验模式: {ABLATION_MODE}")
        print(f"📁 结果将保存在: {SAVE_DIR}")
        print(f"⚙️  LAMBDA_PDE={LAMBDA_PDE}, LAMBDA_BC={LAMBDA_BC}, USE_Q_NET={USE_Q_NET}")
        print(f"==============================================\n")

        # ======================== 👇 自动保存配置到 config.txt 👇 ========================
        # 1. 确保输出结果的文件夹存在
        # save_dir = './results_slices'
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        # 2. 定义配置文件路径
        config_file_path = os.path.join(SAVE_DIR, 'config.txt')

        # 3. 动态抓取并写入配置
        with open(config_file_path, 'w', encoding='utf-8') as f:
            f.write(f"=== 实验运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write("=== 模型运行配置参数 ===\n")

            # # locals() 会获取当前代码块下所有的变量
            # for key, value in locals().copy().items():
            #     # 过滤条件：变量名是全大写（常量规范），且不是以 '_' 开头（排除内置变量）
            #     if key.isupper() and not key.startswith('_'):
            #         f.write(f"{key} = {value}\n")

            # 这里为了在循环里正确写入配置，我们直接把需要的变量写进去
            config_dict = {
                'ABLATION_MODE': ABLATION_MODE, 'LAMBDA_PDE': LAMBDA_PDE, 'LAMBDA_BC': LAMBDA_BC,
                'USE_Q_NET': USE_Q_NET, 'Z_PENALTY': Z_PENALTY, 'SEQ_LEN': SEQ_LEN, 'PRE_LEN': PRE_LEN,
                'BATCH_SIZE': BATCH_SIZE, 'EPOCHS': EPOCHS, 'LAMBDA_RECON': LAMBDA_RECON,
                'TAU_RELAXATION': TAU_RELAXATION, 'K_ADJ': K_ADJ, 'INPUT_DIM': INPUT_DIM,
                'ENABLE_SPATIAL_MASKING': ENABLE_SPATIAL_MASKING, 'MASK_RATIO': MASK_RATIO  # <-- 新增
            }
            for key, value in config_dict.items():
                f.write(f"{key} = {value}\n")

        print(f"✅ 当前运行配置已自动归档至: {config_file_path}")
        # ======================== 👆 结束 👆 ========================

        # --- 1.5 处理空间物理隔离 (向下兼容逻辑) ---
        actual_coords_path = COORDS_PATH  # 默认使用全量传感器坐标文件

        if ENABLE_SPATIAL_MASKING:
            print(f"\n[⚠️ 开启空间物理隔离] 随机遮蔽 {MASK_RATIO * 100}% 的传感器用于纯空间插值测试...")

            RANDOM_SEED = 42
            random.seed(RANDOM_SEED)
            np.random.seed(RANDOM_SEED)

            # 读取全部传感器
            all_coords_df = pd.read_csv(COORDS_PATH)
            # 筛选出当前石窟的所有数据
            grotto_sensors_df = all_coords_df[all_coords_df['grottoe_id'] == TARGET_GROTTO_ID]

            test_sensor_ids = []

            # 🌟 按区域 (zone_id) 进行分层抽样 (Stratified Sampling) 🌟
            for zone_id, group in grotto_sensors_df.groupby('zone_id'):
                zone_sensors = group['sensor_id'].unique().tolist()
                # 计算该区域需要遮蔽的数量 (至少保留1个，防止某区域传感器太少被抹零)
                num_test_zone = max(1, int(len(zone_sensors) * MASK_RATIO))

                # 在该区域内随机抽取
                sampled_for_zone = random.sample(zone_sensors, num_test_zone)
                test_sensor_ids.extend(sampled_for_zone)

                print(f"  - 区域 {zone_id} 共有 {len(zone_sensors)} 个传感器，遮蔽了 {num_test_zone} 个用于测试。")

            # 剩下的全部作为训练集
            grotto_sensors_all = grotto_sensors_df['sensor_id'].unique().tolist()
            train_sensor_ids = [s for s in grotto_sensors_all if s not in test_sensor_ids]

            # 保存测试名单供 compare_interpolation.py 读取
            if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
            with open(os.path.join(SAVE_DIR, 'test_sensors.json'), 'w') as f:
                json.dump(test_sensor_ids, f)

            print(f"✅ 传感器拆分完成：训练使用 {len(train_sensor_ids)} 个，测试预留 {len(test_sensor_ids)} 个")

            # 生成一个临时只包含 80% 传感器的 CSV 文件
            temp_coords_path = os.path.join(SAVE_DIR, 'temp_train_coords.csv')
            all_coords_df[all_coords_df['sensor_id'].isin(train_sensor_ids)].to_csv(temp_coords_path, index=False)

            # 告诉后续的数据加载器去读这个“残缺版”的文件
            actual_coords_path = temp_coords_path

        else:
            print("\n[ℹ️ 使用全量传感器] 模型将看到所有 100% 的节点 (标准预测模式)...")
            # 如果不开启遮蔽，建议生成一个空的 test_sensors.json 避免后续代码报错
            if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
            with open(os.path.join(SAVE_DIR, 'test_sensors.json'), 'w') as f:
                json.dump([], f)

        # --- 2. 加载和预处理数据 ---
        dataset, adj_matrix, temp_scaler, coords_df, coords_scaler, temp_min, temp_max = \
            load_and_preprocess_data(
                DATA_PATH,
                actual_coords_path,
                # COORDS_PATH,
                target_grotto_id=TARGET_GROTTO_ID,  # 传入目标ID
                seq_len=SEQ_LEN,
                pre_len=PRE_LEN,
                k_adj=K_ADJ,
                z_penalty=Z_PENALTY
            )

        ### adj_matrix这个矩阵不是对称矩阵的原因？

        # === 加载并筛选边界坐标（用于边界条件） ===
        boundary_path = os.path.join(DATA_PATH, '_boundary_coords.csv')  # 注意用带下划线的这个新文件

        if os.path.exists(boundary_path):
            bound_df = pd.read_csv(boundary_path)

            # --- 根据 grottoe_id 筛选 ---
            if 'grottoe_id' in bound_df.columns:
                bound_df = bound_df[bound_df['grottoe_id'] == TARGET_GROTTO_ID].copy()
                print(f"已筛选出 {TARGET_GROTTO_ID} 号窟的 {len(bound_df)} 个边界约束点。")
            else:
                print("Warning: 边界文件中没有 grottoe_id 列，将使用所有点！")

            if not bound_df.empty:
                # 使用相同的 coords_scaler 归一化 (必须！)
                # bound_norm_arr = coords_scaler.transform(bound_df[['x', 'y', 'z']])
                bound_norm_arr = coords_scaler.transform(bound_df[['x', 'y', 'z']].values)
                boundary_coords_tensor = torch.tensor(bound_norm_arr, dtype=torch.float32)
            else:
                print(f"Warning: {TARGET_GROTTO_ID} 号窟没有找到边界点！")
                boundary_coords_tensor = None
        else:
            print("Warning: 未找到 _boundary_coords.csv，将跳过边界约束！")
            boundary_coords_tensor = None

        # 获取当前石窟实际的传感器列表 (从 coords_df 中获取)
        SENSOR_IDS = coords_df['sensor_id'].tolist()
        print(f"当前正在对 {TARGET_GROTTO_ID} 号窟的 {len(SENSOR_IDS)} 个传感器进行训练。")

        # train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        # === 时间切分：80% / 10% / 10% （按样本顺序，也就是时间上的先后） ===
        total_samples = len(dataset)
        num_nodes = adj_matrix.shape[0]  # 根据邻接矩阵获取节点数量
        train_end = int(total_samples * 0.8)
        val_end = int(total_samples * 0.9)

        train_idx = list(range(0, train_end))
        val_idx = list(range(train_end, val_end))
        test_idx = list(range(val_end, total_samples))

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        test_subset = Subset(dataset, test_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

        print(f"成功加载数据: dataset samples = {total_samples}, sensors = {num_nodes}")
        print(f"Split counts -> train: {len(train_subset)}, val: {len(val_subset)}, test: {len(test_subset)}")

        # --- 3. 初始化模型 ---
        # <<< 修正 3: 使用 SENSOR_IDS 列表的长度来正确设置节点数
        num_nodes = len(SENSOR_IDS)
        # 获取归一化后的坐标数据的 tensor
        # coords_df 已经筛选过，只包含当前石窟的
        # coords_normalized_array = coords_scaler.transform(coords_df[['x', 'y', 'z']])
        coords_normalized_array = coords_scaler.transform(coords_df[['x', 'y', 'z']].values)
        coords_tensor = torch.tensor(coords_normalized_array, dtype=torch.float32)
        # model = SpatioTemporalPINN(num_nodes=num_nodes)
        model = SpatioTemporalPINN(num_nodes=num_nodes, sensor_coords_norm=coords_tensor,
                                   in_dim=INPUT_DIM, out_dim=PRE_LEN, use_q_net=USE_Q_NET)
        print(f"模型已初始化，节点数: {num_nodes}, 输入维度: {INPUT_DIM}")

        # --- 4. 训练模型 ---
        print("\n--- 开始训练模型 ---")
        model, history = train(
            model,
            train_loader,
            adj_matrix,  # pre_adj
            # coords_scaler.transform(coords_df[['x', 'y', 'z']]),
            coords_scaler.transform(coords_df[['x', 'y', 'z']].values),  # coords_df_normalized
            boundary_coords=boundary_coords_tensor,  # <--- 传入筛选后的边界点
            epochs=EPOCHS,
            lr=5e-4, # 原来是 1e-3，现在降低一半让步伐更稳
            lambda_pde=LAMBDA_PDE,
            lambda_bc=LAMBDA_BC,  # 边界约束权重
            lambda_recon=LAMBDA_RECON,  # 重构误差权重
            tau=TAU_RELAXATION,
            val_loader=val_loader,
            test_loader=test_loader,
            device=None
        )
        print("--- 模型训练完毕 ---\n")

        # 保存 CSV：
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(os.path.join(SAVE_DIR, f'grotto_{TARGET_GROTTO_ID}_loss_history.csv'), index=False)
        print("Loss 历史记录已保存！")

        # --- 5. 推理与可视化 ---
        print("--- 开始推理与可视化 ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # 假设想看一个标准化的时间点 t=0.5 时，在 z=0 平面的温度分布
        # # 注意: t=0.5 代表整个时间序列一半的位置，这是一个标准化的概念
        # target_time = 0.5
        # target_z_value = coords_df['z'].mean()

        # 1. 准备全量归一化数据 (用于推理时抓取历史窗口)
        # merged_df 在 load_and_preprocess_data 内部被创建了，但没有返回出来
        # 我们需要重新构建一个全量的 normalized dataframe，或者修改 data_loader 让它返回 merged_df
        # 建议：在 main 里重新加载一次 merged_df (使用 load_merged_dataframe) 并归一化

        # 加载原始数据
        full_df = load_merged_dataframe(DATA_PATH, SENSOR_IDS)
        # 归一化
        # full_df_norm = pd.DataFrame(temp_scaler.transform(full_df), index=full_df.index, columns=full_df.columns)
        full_df_norm = pd.DataFrame(temp_scaler.transform(full_df.values), index=full_df.index, columns=full_df.columns)

        # 2. 准备 Zone Features (同 data_loader)
        zone_ids = coords_df['zone_id'].values.astype(int)
        zone_one_hot = np.eye(3)[zone_ids].T  # (3, N)

        # # 训练数据的开始和结束时间：
        # start_time = datetime(2024, 6, 25, 15, 15, 0)
        # end_time = datetime(2024, 10, 17, 23, 55, 0)
        # time_norm = TimeNormalizer(start_time, end_time)
        #
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #
        # # 查询单点: (1.1m, 1.2m, 3.5m) 在 2024-01-01 12:05 的温度
        # # 初始化时间归一化器
        # time_norm = TimeNormalizer(start_time, end_time)

        # 3. 定义查询点
        example_points = [
            (-5.24, 21.50, 0.63, datetime(2024, 10, 6, 12, 0, 0)),  # 这里的坐标最好找一个存在的传感器坐标试试
            (-2.09, 22.11, 0.42, datetime(2024, 10, 7, 12, 30, 0))
        ]

        # 定义查询切片
        # 注意：None 表示该维度是变量（平面轴），数值表示固定值
        example_plate = [
            # YZ平面：X 固定在 -5.24 (前室大门附近)
            (-5.24, None, None, datetime(2024, 10, 6, 12, 0, 0)),
            # XZ平面：Y 固定在 22.11 (可能在洞窟中部)
            (None, 22.11, None, datetime(2024, 10, 7, 12, 30, 0)),
            # XY平面：Z 固定在 1.00 (距离地面1米的高度)
            (None, None, 1.00, datetime(2024, 10, 7, 12, 30, 0)),
        ]

        # 4. 调用新的推理函数
        results = compare_points_predictions_with_observed(
            points=example_points,
            model=model,
            merged_df_norm=full_df_norm,  # 传入全量数据源
            adj_matrix=adj_matrix,
            zone_features=zone_one_hot,
            coords_scaler=coords_scaler,
            temp_min=temp_min,
            temp_max=temp_max,
            seq_len=SEQ_LEN,
            device=device
        )

        for r in results:
            print(f"Point ({r['x']:.2f}, {r['y']:.2f}, {r['z']:.2f}) @ {r['t']} -> Pred: {r['predicted_temp']:.2f} °C")

        # === 平面切片生成与可视化 ===
        print("\n--- 开始生成平面切片 ---")

        process_plane_slices(
            plane_requests=example_plate,
            model=model,
            merged_df_norm=full_df_norm,
            adj_matrix=adj_matrix,
            zone_features=zone_one_hot,
            coords_scaler=coords_scaler,
            coords_df=coords_df,  # 需要传入 coords_df 来确定边界
            temp_min=temp_min,
            temp_max=temp_max,
            seq_len=SEQ_LEN,
            device=device,
            step=0.2,  # 分辨率：每隔 0.2米 算一个点
            output_dir=SAVE_DIR  # 结果保存文件夹
        )

        print("所有切片生成完毕，存在 ./results_slices 文件夹。")

        # === 7. 保存模型与环境 ===
        model_save_path = os.path.join(SAVE_DIR, f'grotto_{TARGET_GROTTO_ID}_model_checkpoint.pth')
        print(f"\n正在保存 {TARGET_GROTTO_ID} 号窟的模型和配置到: {model_save_path} ...")

        # 辅助函数：将 scaler 的关键属性提取为 list，避免 pickle 版本问题
        def get_scaler_params(scaler):
            return {
                'min_': scaler.min_.tolist(),
                'scale_': scaler.scale_.tolist(),
                'data_min_': scaler.data_min_.tolist(),
                'data_max_': scaler.data_max_.tolist(),
            }

        # 构建检查点字典
        checkpoint = {
            # 1. 模型状态
            'model_state_dict': model.state_dict(),
            # 2. 模型配置参数 (用于重建模型实例)
            'config': {
                'num_nodes': num_nodes,
                'input_dim': INPUT_DIM,
                'out_dim': PRE_LEN,
                'sensor_coords_norm': model.sensor_coords.cpu(),  # 保存归一化后的坐标张量
            },
            # 3. 预处理工具 (非常重要，否则无法反归一化)
            'scalers_params': {
                'coords_scaler': get_scaler_params(coords_scaler),
                'temp_scaler': get_scaler_params(temp_scaler),
                'global_min': float(temp_min), # 存 float
                'global_max': float(temp_max)  # 存 float
            },
            # 4. 图结构与特征
            'graph_data': {
                # adj_matrix 是 numpy 数组，转为 list 避开 numpy 版本差异
                'adj_matrix': adj_matrix.tolist(),
                'zone_ids': coords_df['zone_id'].values.astype(int).tolist(),
                'sensor_ids': SENSOR_IDS # List 是安全的
            },
            # 5. 训练参数 (可选，用于记录)
            'train_info': {
                'seq_len': SEQ_LEN,
                'pre_len': PRE_LEN
            }
        }

        torch.save(checkpoint, model_save_path)
        print("模型保存成功！")

    print(f"✅ Z_PENALTY = {current_z} 的训练和保存已全部完成！\n")
    # 循环结束，自动进入下一个 Z 值的训练
