#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于推理模块
查询和生成温度场
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2024.2.2
time: 2025/9/7 13:27
version: V1.0
"""

from scipy.ndimage import label
from datetime import datetime
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.spatial import cKDTree


class GeometryMasker:
    def __init__(self, shape_file_path, threshold=0.3, slab_thickness=0.15):
        print(f"正在加载几何形状文件 (微厚度版): {shape_file_path} ...")
        df = pd.read_csv(shape_file_path)
        self.valid_points = df[['x', 'y', 'z']].values
        self.threshold = threshold
        self.slab_thickness = slab_thickness
        # 保留全局 3D 树，用于备用或侧视图
        self.tree = cKDTree(self.valid_points)

    def get_distances(self, query_points):
        """全视角微厚度距离计算"""
        xs, ys, zs = query_points[:, 0], query_points[:, 1], query_points[:, 2]

        if np.all(zs == zs[0]):  # --- XY 俯视图 (固定 Z) ---
            fixed_val = zs[0]
            slab_mask = (self.valid_points[:, 2] >= fixed_val - self.slab_thickness) & \
                        (self.valid_points[:, 2] <= fixed_val + self.slab_thickness)
            slab_points = self.valid_points[slab_mask][:, :2]
            query_2d = query_points[:, :2]

        elif np.all(xs == xs[0]):  # --- YZ 侧视图 (固定 X) 🚀 ---
            fixed_val = xs[0]
            slab_mask = (self.valid_points[:, 0] >= fixed_val - self.slab_thickness) & \
                        (self.valid_points[:, 0] <= fixed_val + self.slab_thickness)
            slab_points = self.valid_points[slab_mask][:, 1:]  # 提取 Y, Z 坐标
            query_2d = query_points[:, 1:]

        elif np.all(ys == ys[0]):  # --- XZ 正视图 (固定 Y) 🚀 ---
            fixed_val = ys[0]
            slab_mask = (self.valid_points[:, 1] >= fixed_val - self.slab_thickness) & \
                        (self.valid_points[:, 1] <= fixed_val + self.slab_thickness)
            slab_points = self.valid_points[slab_mask][:, [0, 2]]  # 提取 X, Z 坐标
            query_2d = query_points[:, [0, 2]]

        else:
            return self.tree.query(query_points)[0]

        # 如果薄片内无点，退回 3D；否则进行致密 2D 匹配
        if len(slab_points) == 0:
            print("⚠️ 警告：当前切片厚度内未找到流体节点，退回 3D 距离")
            dists, _ = self.tree.query(query_points)
        else:
            tree_2d = cKDTree(slab_points)
            dists, _ = tree_2d.query(query_2d)

        return dists


# ==============================================================
# 1. 时间归一化工具类
# ==============================================================
class TimeNormalizer:
    """
    负责时间到归一化时间的映射，与训练时保持一致。
    """

    def __init__(self, start_time: datetime, end_time: datetime):
        self.start_time = start_time
        self.end_time = end_time
        self.total_minutes = (end_time - start_time).total_seconds() / 60.0

    def timestamp_to_norm(self, t: datetime) -> float:
        """将时间戳转成 0-1 范围的归一化时间"""
        delta_minutes = (t - self.start_time).total_seconds() / 60.0
        return np.clip(delta_minutes / self.total_minutes, 0.0, 1.0)

    def minutes_offset_to_norm(self, offset_min: float) -> float:
        """输入分钟偏移 -> 归一化时间"""
        return np.clip(offset_min / self.total_minutes, 0.0, 1.0)


# ==============================================================
# 2. 查询单点温度
# ==============================================================
def get_point_temperature(
    model: torch.nn.Module,
    x: float, y: float, z: float, t_raw: object,
    coords_scaler: MinMaxScaler,
    temp_scaler: Optional[MinMaxScaler] = None,
    temp_min: Optional[float] = None,
    temp_max: Optional[float] = None,
    time_normalizer: Optional[TimeNormalizer] = None,
    time_input_type: str = "timestamp",
    device: Optional[torch.device] = None
) -> float:
    """
    查询单个点的温度（真实温度）。
    Args:
        model: 训练好的模型
        x, y, z: 坐标 (m)
        t_raw: datetime / minutes / normalized float
        coords_scaler: 用于归一化空间坐标
        temp_scaler: 可选，用于反归一化
        temp_min, temp_max: 可选，全局最小最大温度（优先使用）
        time_normalizer: 时间归一化器
        time_input_type: 'timestamp' | 'minutes' | 'norm'
        device: torch 设备
    Returns:
        float: 反归一化后的温度
    """
    model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 归一化坐标
    xyz_norm = coords_scaler.transform(np.array([[x, y, z]], dtype=np.float64))[0]

    # 归一化时间
    if time_input_type == "timestamp":
        if not isinstance(t_raw, datetime):
            raise ValueError("t_raw must be datetime when time_input_type='timestamp'")
        t_norm = time_normalizer.timestamp_to_norm(t_raw)
    elif time_input_type == "minutes":
        t_norm = time_normalizer.minutes_offset_to_norm(float(t_raw))
    elif time_input_type == "norm":
        t_norm = float(t_raw)
    else:
        raise ValueError("time_input_type must be one of ['timestamp','minutes','norm']")

    # 拼接输入
    input_arr = np.array([[xyz_norm[0], xyz_norm[1], xyz_norm[2], t_norm]], dtype=np.float32)
    input_tensor = torch.tensor(input_arr, dtype=torch.float32, device=device)

    # 模型推理
    with torch.no_grad():
        pred_norm = model.decoder(input_tensor).detach().cpu().numpy().ravel()[0]

    # 反归一化
    if temp_min is not None and temp_max is not None:
        return float(pred_norm * (temp_max - temp_min) + temp_min)

    if temp_scaler is not None:
        try:
            if getattr(temp_scaler, "n_features_in_", 1) == 1:
                return float(temp_scaler.inverse_transform([[pred_norm]]).ravel()[0])
            else:
                # 多特征 scaler 退化方案
                dmin, dmax = np.min(temp_scaler.data_min_), np.max(temp_scaler.data_max_)
                return float(pred_norm * (dmax - dmin) + dmin)
        except Exception as e:
            print("Warning: inverse_transform failed:", e)

    return float(pred_norm)


# ==============================================================
# 3. 批量查询任意点温度（推荐批量推理用）
# ==============================================================
def batch_query_points(
    model: torch.nn.Module,
    points: List[Tuple[float, float, float, object]],  # [(x,y,z,t_raw), ...]
    coords_scaler: MinMaxScaler,
    temp_scaler: Optional[MinMaxScaler] = None,
    temp_min: Optional[float] = None,
    temp_max: Optional[float] = None,
    time_normalizer: Optional[TimeNormalizer] = None,
    time_input_type: str = "timestamp",
    batch_size: int = 1024,
    device: Optional[torch.device] = None
) -> List[float]:
    """
    批量查询多个空间-时间点温度
    """
    model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if time_input_type in ("timestamp", "minutes") and (time_normalizer is None):
        raise ValueError("time_normalizer is required when time_input_type='timestamp' or 'minutes'")

    # 1. 构建归一化输入
    inputs = np.zeros((len(points), 4), dtype=np.float32)
    for i, (x, y, z, t_raw) in enumerate(points):
        xyz_norm = coords_scaler.transform(np.array([[x, y, z]], dtype=np.float64))[0]
        inputs[i, :3] = xyz_norm
        if time_input_type == "timestamp":
            t_norm = time_normalizer.timestamp_to_norm(t_raw)
        elif time_input_type == "minutes":
            t_norm = time_normalizer.minutes_offset_to_norm(float(t_raw))
        else:
            t_norm = float(t_raw)
        inputs[i, 3] = t_norm

    # 2. 批量前向推理
    preds_norm = np.zeros((len(points),), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(points), batch_size):
            end = min(start + batch_size, len(points))
            batch_tensor = torch.tensor(inputs[start:end], dtype=torch.float32, device=device)
            preds_norm[start:end] = model.decoder(batch_tensor).detach().cpu().numpy().ravel()

    # 3. 反归一化
    if (temp_min is not None) and (temp_max is not None):
        return ((preds_norm * (temp_max - temp_min)) + temp_min).tolist()

    if temp_scaler is not None:
        try:
            if getattr(temp_scaler, "n_features_in_", 1) == 1:
                preds_real = temp_scaler.inverse_transform(preds_norm.reshape(-1, 1)).ravel()
                return preds_real.tolist()
            else:
                dmin, dmax = np.min(temp_scaler.data_min_), np.max(temp_scaler.data_max_)
                preds_real = preds_norm * (dmax - dmin) + dmin
                return preds_real.tolist()
        except Exception as e:
            print("Warning: inverse_transform failed:", e)
            return preds_norm.tolist()

    return preds_norm.tolist()


# ==============================================================
# 5. 平面切片生成与可视化
# ==============================================================
def process_plane_slices(
        plane_requests,  # 用户输入的列表，包含 (x, y, z, t) 其中两个为 None
        model,
        merged_df_norm,  # 全量归一化数据 (用于抓取历史)
        adj_matrix,
        zone_features,
        coords_scaler,
        coords_df,  # 用于确定石窟边界 (min/max)
        temp_min, temp_max,
        seq_len,
        device,
        step=0.2,  # 网格步长 (米)
        output_dir='./results',
        masker=None
):
    """
    处理平面切片请求：生成网格 -> 批量预测 -> 保存CSV -> 画图
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 获取石窟的边界范围 (Bounding Box)
    x_min, x_max = coords_df['x'].min(), coords_df['x'].max()
    y_min, y_max = coords_df['y'].min(), coords_df['y'].max()
    z_min, z_max = coords_df['z'].min(), coords_df['z'].max()

    # 🚀 精细化、不对称的画布扩展 (Padding)
    pad_x_left = 2.0  # 左侧大山厚，保留 2 米
    pad_x_right = 0.15  # 🎯 去除 9 窟：右侧只扩展 0.5 米，强行把 9 窟挡在画板外面！
    pad_y_bottom = 1.5  # 🎯 画面下移：底部不扩展 (0.0 米)，消除与 X 轴的多余灰色留白！
    pad_y_top = 3.0  # 🎯 顶部显示：上方暴力扩展 4.0 米，确保“后脑勺”绝对完整露出！
    pad_z = 2.0  # Z 轴默认扩展 2 米

    for req in plane_requests:
        fixed_x, fixed_y, fixed_z, t_query = req

        # 确定平面类型和变量范围
        grid_points = []
        plane_type = ""
        filename_base = ""

        # --- 情况 A: YZ 平面 (X 固定) ---
        if fixed_x is not None and fixed_y is None and fixed_z is None:
            plane_type = 'YZ'
            print(f"正在生成 X={fixed_x} 的 YZ 平面切片...")
            # 生成网格
            ys = np.arange(y_min - pad_y_bottom, y_max + pad_y_top, step)
            zs = np.arange(z_min - pad_z, z_max + pad_z, step)
            YY, ZZ = np.meshgrid(ys, zs)
            # 展平
            flat_y = YY.ravel()
            flat_z = ZZ.ravel()
            flat_x = np.full_like(flat_y, fixed_x)

            # 构建查询列表
            for i in range(len(flat_x)):
                grid_points.append((flat_x[i], flat_y[i], flat_z[i], t_query))

            filename_base = f"Slice_X_{fixed_x}_{t_query.strftime('%Y%m%d_%H%M')}"
            plot_x_axis, plot_y_axis = YY, ZZ
            x_label, y_label = 'Y (m)', 'Z (m)'

        # --- 情况 B: XZ 平面 (Y 固定) ---
        elif fixed_x is None and fixed_y is not None and fixed_z is None:
            plane_type = 'XZ'
            print(f"正在生成 Y={fixed_y} 的 XZ 平面切片...")
            xs = np.arange(x_min - pad_x_left, x_max + pad_x_right, step)
            zs = np.arange(z_min - pad_z, z_max + pad_z, step)
            XX, ZZ = np.meshgrid(xs, zs)

            flat_x = XX.ravel()
            flat_z = ZZ.ravel()
            flat_y = np.full_like(flat_x, fixed_y)

            for i in range(len(flat_x)):
                grid_points.append((flat_x[i], flat_y[i], flat_z[i], t_query))

            filename_base = f"Slice_Y_{fixed_y}_{t_query.strftime('%Y%m%d_%H%M')}"
            plot_x_axis, plot_y_axis = XX, ZZ
            x_label, y_label = 'X (m)', 'Z (m)'

        # --- 情况 C: XY 平面 (Z 固定) ---
        elif fixed_x is None and fixed_y is None and fixed_z is not None:
            plane_type = 'XY'
            print(f"正在生成 Z={fixed_z} 的 XY 平面切片...")
            xs = np.arange(x_min - pad_x_left, x_max + pad_x_right, step)
            ys = np.arange(y_min - pad_y_bottom, y_max + pad_y_top, step)
            XX, YY = np.meshgrid(xs, ys)

            flat_x = XX.ravel()
            flat_y = YY.ravel()
            flat_z = np.full_like(flat_x, fixed_z)

            for i in range(len(flat_x)):
                grid_points.append((flat_x[i], flat_y[i], flat_z[i], t_query))

            filename_base = f"Slice_Z_{fixed_z}_{t_query.strftime('%Y%m%d_%H%M')}"
            plot_x_axis, plot_y_axis = XX, YY
            x_label, y_label = 'X (m)', 'Y (m)'

        else:
            print(f"跳过无效请求: {req} (必须恰好有两个 None)")
            continue

        # 2. 调用现有的批量推理函数 (复用 main.py 里的逻辑，这里引用它或复制逻辑)
        # 为了代码整洁，建议把 main.py 里的 `compare_points_predictions_with_observed`
        # 的核心推理部分拆出来，或者我们在 inference.py 里再实现一个简单的纯预测版本
        # 这里直接调用下面的 `batch_predict_only` (会在下面定义它)

        preds = batch_predict_only(
            grid_points, model, merged_df_norm, adj_matrix, zone_features,
            coords_scaler, temp_min, temp_max, seq_len, device
        )

        # ================== 核心掩膜：多视角定向拓扑油漆桶 ==================
        if masker is not None:
            query_xyz = np.array([[p[0], p[1], p[2]] for p in grid_points])
            dists = masker.get_distances(query_xyz)

            outline_mask_1d = dists <= masker.threshold
            outline_2d = outline_mask_1d.reshape(plot_x_axis.shape)

            from scipy.ndimage import label
            empty_spaces = (~outline_2d).astype(int)
            labeled_array, num_features = label(empty_spaces)

            solid_mask_2d = outline_2d.copy()

            if plane_type == 'XY':
                # 俯视图逻辑保持不变：只排查左、上边缘，防止右侧通道被误杀
                outside_rock_labels = set(labeled_array[:, 0]) | set(labeled_array[-1, :])
                max_area = 0
                air_label = -1
                for i in range(1, num_features + 1):
                    if i not in outside_rock_labels:
                        area = np.sum(labeled_array == i)
                        if area > max_area:
                            max_area = area
                            air_label = i
                if air_label != -1:
                    solid_mask_2d = solid_mask_2d | (labeled_array == air_label)
            else:
                # 🚀 YZ / XZ 侧视图逻辑：保留所有的独立“空气岛屿”(如诵经道)！
                # 侧视图的空气域被包裹在大山岩石中。凡是不触碰画布边缘的空白，全都是空气！
                edge_labels = set(labeled_array[:, 0]) | set(labeled_array[:, -1]) | \
                              set(labeled_array[0, :]) | set(labeled_array[-1, :])

                for i in range(1, num_features + 1):
                    if i not in edge_labels:
                        solid_mask_2d = solid_mask_2d | (labeled_array == i)

            preds_np = np.array(preds)
            temp_grid = preds_np.reshape(plot_x_axis.shape)
            temp_grid[~solid_mask_2d] = np.nan
            preds = temp_grid.flatten().tolist()
        else:
            temp_grid = np.array(preds).reshape(plot_x_axis.shape)
        # ==============================================================

        # 3. 保存 CSV
        df_res = pd.DataFrame(grid_points, columns=['x', 'y', 'z', 't'])
        df_res['temperature'] = preds
        csv_path = os.path.join(output_dir, f"{filename_base}.csv")
        df_res.to_csv(csv_path, index=False)
        print(f"  -> CSV 已保存: {csv_path}")

        # 4. 绘制热力图
        # 将预测值 reshape 回网格形状
        temp_grid = np.array(preds).reshape(plot_x_axis.shape)

        plt.figure(figsize=(10, 8))
        # 使用 contourf 绘制平滑的填充等高线
        # levels 从 50 改为 200，让颜色渐变像丝绸一样顺滑
        # 这里的 cmap='jet' 可以改成 'turbo' 或 'rainbow' 试试，可能更像 Fluent 的色卡
        cp = plt.contourf(plot_x_axis, plot_y_axis, temp_grid, levels=50, cmap='rainbow')
        # 替换为使用 pcolormesh:
        # shading='nearest' 能最真实地反映掩膜切掉的边界，不会有一丝多余的插值拉伸
        # cp = plt.pcolormesh(plot_x_axis, plot_y_axis, temp_grid, cmap='rainbow', shading='nearest')
        cbar = plt.colorbar(cp)
        cbar.set_label('Temperature (°C)')

        # 设置背景色为灰色，代表墙壁
        plt.gca().set_facecolor('#d3d3d3')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(
            f"{plane_type} Slice Prediction @ {t_query}\n(Fixed: {filename_base.split('_')[1]}={filename_base.split('_')[2]})")
        plt.axis('equal')  # 保证比例一致

        img_path = os.path.join(output_dir, f"{filename_base}.png")
        plt.savefig(img_path, dpi=400)
        plt.close()  # 关闭画布释放内存
        print(f"  -> 图像 已保存: {img_path}")


# 这是一个简化版的推理函数，只返回预测温度列表，不对比真实值
def batch_predict_only(
        points, model, merged_df_norm, adj_matrix, zone_features,
        coords_scaler, temp_min, temp_max, seq_len, device
):
    """
    inference.py 内部使用的纯预测函数
    """
    model.eval()
    preds_list = []
    pre_adj_t = torch.tensor(adj_matrix, dtype=torch.float32, device=device)

    data_values = merged_df_norm.values if hasattr(merged_df_norm, 'values') else merged_df_norm
    time_index = merged_df_norm.index

    # 1. 既然所有点的时间都是一样的 (同一个切片)，直接取第一个点的时间处理即可
    # 这样速度极快，不需要字典分组
    t_query = points[0][3]

    if isinstance(t_query, str):
        t_query_dt = pd.to_datetime(t_query)
    else:
        t_query_dt = t_query

    # 构建历史输入 (只做一次)
    try:
        idx_loc = time_index.get_indexer([t_query_dt], method='pad')[0]
    except:
        return [np.nan] * len(points)

    if idx_loc < seq_len:
        return [np.nan] * len(points)

    temp_window = data_values[idx_loc - seq_len: idx_loc, :].T
    feat_temp = temp_window[np.newaxis, :, :]
    feat_zone = np.tile(zone_features[:, :, np.newaxis], (1, 1, seq_len))
    x_input_np = np.concatenate([feat_temp, feat_zone], axis=0)
    x_input_tensor = torch.tensor(x_input_np, dtype=torch.float32).unsqueeze(0).to(device)

    # 运行 GWN
    with torch.no_grad():
        _, sensor_features = model.gwn(x_input_tensor, pre_adj_t)

        # 批量处理空间点 (可能有几千个，为了显存安全，可以分 Batch)
        # 这里定义一个内部 batch_size
        internal_bs = 2048
        num_points = len(points)

        final_preds = []

        for i in range(0, num_points, internal_bs):
            batch_pts = points[i: i + internal_bs]

            batch_coords_list = []
            for (x, y, z, _) in batch_pts:
                # 纯数值 transform，不带列名
                xyz_norm = coords_scaler.transform(np.array([[x, y, z]]))[0]
                batch_coords_list.append([xyz_norm[0], xyz_norm[1], xyz_norm[2], 0.0])

            query_tensor = torch.tensor(batch_coords_list, dtype=torch.float32).unsqueeze(0).to(device)

            query_xyz = query_tensor[..., :3]
            interpolated_feats = model.interpolator(query_xyz, model.sensor_coords, sensor_features)

            decoder_input = torch.cat([query_tensor, interpolated_feats], dim=-1)
            pred_norm = model.decoder(decoder_input).squeeze(-1)

            vals = pred_norm.cpu().numpy().flatten()

            # 反归一化
            vals_real = vals * (temp_max - temp_min) + temp_min
            final_preds.extend(vals_real)

    return final_preds



# ==============================================================
# 4. 示例调用（在 main.py 末尾引用）
# ==============================================================
if __name__ == "__main__":
    # 假设 main.py 中已有这些对象
    print("此模块提供批量温度推理接口，请在 main.py 中导入使用。")
