#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于核心模型架构
定义了GraphWaveNet、PINN解码器以及将它们组合起来的最终模型。
更完整的空洞卷积堆栈: 新版GraphWaveNet模型在__init__方法中引入了num_blocks和num_layers_per_block参数。这使得模型可以堆叠多组层，每一组都包含扩张率呈指数增长（如1, 2, 4, 8）的卷积层。这是捕捉不同时间尺度依赖的关键。
跳跃连接聚合: 在forward方法中，我们现在会累加所有skip_conv的输出到skip_sum中。最终的end_conv不再是作用于最后一层的输出，而是作用于这个聚合的skip_sum上。这能够整合所有时空层学习到的特征，并能有效缓解训练过程中的梯度消失问题。
更灵活的输入/输出维度: GraphWaveNet的in_dim和out_dim参数使得它能够处理多维特征输入（例如，除了温度，还有湿度等）并预测多步长的未来值。
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2024.2.2
time: 2025/9/9 14:43
version: V1.0
"""

# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class nconv(nn.Module):
    """
    一个简单的归一化图卷积层。
    输入: x (..., num_nodes, in_channels)
    输出: x (..., num_nodes, out_channels)
    """

    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        """
        x: (B, C, N, T)
        A: (N, N)
        返回: (B, C, N, T) -> 对节点维进行邻接加权
        """
        # 确保 A 在同一 device
        if not isinstance(A, torch.Tensor):
            A = torch.tensor(A, dtype=x.dtype, device=x.device)
        else:
            A = A.to(x.device).type(x.dtype)

        # einsum: 'b c n t, n m -> b c m t'，把原本的 n 聚合到输出节点 m
        x = torch.einsum('bcnt,nm->bcmt', x, A)
        return x.contiguous()


class GraphWaveNet(nn.Module):
    """
    完整的GraphWaveNet模型，包含空洞卷积堆栈和跳跃连接。
    核心思想:
    1. 自适应学习邻接矩阵
    2. 使用多层空洞卷积捕捉长时序依赖
    3. 使用跳跃连接聚合不同层的信息
    """

    def __init__(self, num_nodes, num_blocks=2, num_layers_per_block=4, in_dim=1, out_dim=12,
                 residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512):
        super(GraphWaveNet, self).__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.end_channels = end_channels

        # 可学习的自适应邻接矩阵
        self.adp_adj = nn.Parameter(torch.randn(num_nodes, num_nodes))

        # 1. 输入层: 将原始数据维度映射到残差维度
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))

        # 2. 核心时空层堆栈
        self.layers = nn.ModuleList()
        # 扩张率列表，例如 [1, 2, 4, 8]
        dilation_rates = [2 ** i for i in range(num_layers_per_block)]

        receptive_field = 1
        for i in range(num_blocks):
            for dilation in dilation_rates:
                self.layers.append(nn.ModuleList([
                    nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, 2), dilation=dilation),  # 滤波卷积
                    nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, 2), dilation=dilation),  # 门控卷积
                    nn.Conv2d(dilation_channels, residual_channels, kernel_size=(1, 1)),  # 残差卷积
                    nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1)),  # 跳跃连接卷积
                    nconv()  # 图卷积
                ]))
                receptive_field += (2 - 1) * dilation

        self.receptive_field = receptive_field

        # 3. 输出层: 聚合跳跃连接，进行最终预测
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, kernel_size=(1, 1))

        # --- 特征投影层 ---
        # 将 512 维的高维特征降维到 32 维，作为"环境上下文"传给 MLP
        self.feature_dim = 32
        self.feature_proj = nn.Conv2d(end_channels, self.feature_dim, kernel_size=(1, 1))
        # ---------------------

    def forward(self, x, pre_adj):
        """
        x: (B, in_dim, N, T)
        pre_adj: (N, N) numpy or tensor (predefined adjacency)
        返回: y_pred shape -> (B, N, out_dim)  （out_dim 通常为 PRE_LEN）
        """
        # 确保输入维度正确
        assert x.dim() == 4, f"Expected x to be 4D (B, C, N, T), got {x.shape}"

        # 输入通道映射
        x = self.start_conv(x)  # -> (B, residual_channels, N, T)

        # 处理邻接矩阵（自适应 + 预定义）
        adp = F.softmax(F.relu(self.adp_adj), dim=1)
        if not isinstance(pre_adj, torch.Tensor):
            pre_adj = torch.tensor(pre_adj, dtype=adp.dtype, device=adp.device)
        else:
            pre_adj = pre_adj.to(adp.device).type(adp.dtype)
        adj = adp + pre_adj  # (N, N)

        skip_sum = None

        # 核心卷积循环
        for i in range(len(self.layers)):
            filter_conv, gate_conv, residual_conv, skip_conv, gconv = self.layers[i]

            residual = x  # (B, residual_channels, N, T)

            # dilated convs 膨胀卷积 (可能改变时间维长度)
            _f = filter_conv(residual)
            _g = gate_conv(residual)
            filt = torch.tanh(_f)
            gate = torch.sigmoid(_g)
            x_conv = filt * gate  # (B, dilation_channels, N, T')

            # skip 跳跃连接
            skip_out = skip_conv(x_conv)  # (B, skip_channels, N, T')
            if skip_sum is None:
                skip_sum = skip_out
            else:
                # 对齐时间维再相加（取末尾对齐）
                if skip_out.size(-1) != skip_sum.size(-1):
                    min_t = min(skip_out.size(-1), skip_sum.size(-1))
                    skip_sum = skip_sum[..., -min_t:]
                    skip_out = skip_out[..., -min_t:]
                skip_sum = skip_sum + skip_out

            # 图卷积：保持 (B, dilation_channels, N, T')
            x_gconv = gconv(x_conv, adj)

            # 残差连接：先做1x1再对齐时间维
            x_res = residual_conv(x_gconv)  # (B, residual_channels, N, T')
            if x_res.size(-1) != residual.size(-1):
                # align by taking last timestamps
                x = x_res + residual[..., -x_res.size(-1):]
            else:
                x = x_res + residual

        # 最终处理跳跃连接
        if skip_sum is None:
            raise RuntimeError("skip_sum is None: model layers produced no skip connections.")

        # 经过 end_conv_1: (B, end_channels, N, T_out)
        x_features = F.relu(self.end_conv_1(skip_sum))

        # --- 动作 A: 生成最终预测 (给 L_data 用) ---
        # 经过 end_conv_2: (B, out_dim, N, T_out)
        x_out = self.end_conv_2(x_features)  # (B, out_dim, N, T_out)
        # if x_out.size(-1) == 1:
        #     x_out = x_out.squeeze(-1)
        # 强制取最后一个时间步，去掉时间维度
        # Shape 变为: (B, out_dim, N)
        x_out = x_out[..., -1]
        y_pred = x_out.permute(0, 2, 1).contiguous()  # (B, N, out_dim)

        # --- 动作 B: 提取并降维特征 (给 MLP 用) ---
        # 取最后一个时间步的特征作为"当前环境特征" (B, end_channels, N)
        last_step_feat = x_features[..., -1]
        # 为了通过 feature_proj (Conv2d)，扩展回 4D: (B, end_channels, N, 1)
        last_step_feat = last_step_feat.unsqueeze(-1)
        # 投影降维 x_features: (B, end_channels, N, 1) -> (B, feature_dim, N, 1)
        node_features = self.feature_proj(last_step_feat)
        # node_features = self.feature_proj(x_features)
        # 调整形状: (B, N, feature_dim)
        node_features = node_features.squeeze(-1).permute(0, 2, 1).contiguous()  # (B, N, feature_dim)

        return y_pred, node_features

class PINNDecoder(nn.Module):
    """
    PINN解码器，一个简单的多层感知机 (MLP)
    将时空坐标映射为温度值
    """

    def __init__(self, input_dim=36, hidden_dim=64, output_dim=1):
        super(PINNDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class SpatioTemporalPINN(nn.Module):
    """
    将GraphWaveNet和PINN解码器组合起来的最终模型
    """

    # --- in_dim 参数，默认为 1 ---
    def __init__(self, num_nodes, sensor_coords_norm, num_blocks=2,
                 num_layers_per_block=4, in_dim=1, out_dim=12, use_q_net=True):
        """
        sensor_coords_norm: (N, 3) Tensor, 归一化后的传感器坐标，需要在初始化时传入并固定
        """

        super(SpatioTemporalPINN, self).__init__()

        self.use_q_net = use_q_net  # 保存 Q-Net 开关状态

        # GWN
        self.gwn = GraphWaveNet(num_nodes=num_nodes, in_dim=in_dim, out_dim=out_dim)
        # Interpolator RBF插值
        self.interpolator = RBFFeatureInterpolator(sigma=0.1, k=4)
        # self.interpolator = IDWFeatureInterpolator(k=4)  # 找最近的4个传感器插值
        # Decoder (输入 = 4维时空 + 32维GWN特征)
        gwn_feature_dim = 32  # 必须与 GWN 中的 self.feature_dim 一致
        self.decoder = PINNDecoder(input_dim=4 + gwn_feature_dim)
        # 注册传感器坐标为 buffer (不参与梯度更新，但随模型保存)
        # 确保传入的是 Tensor
        if not isinstance(sensor_coords_norm, torch.Tensor):
            sensor_coords_norm = torch.tensor(sensor_coords_norm, dtype=torch.float32)
        self.register_buffer('sensor_coords', sensor_coords_norm)

        # 初始化物理参数，使用 _param 后缀区分
        self.alpha_param = nn.Parameter(torch.tensor(0.01))
        # self.q_param = nn.Parameter(torch.tensor(0.0))
        # 动态上下文感知源项网络
        gwn_feature_dim = 32  # 这是网络里设定的特征维度
        self.q_net = nn.Sequential(
            nn.Linear(3 + gwn_feature_dim, 32),
            nn.LayerNorm(32),  # 层归一化，防止外部动态特征突变冲垮网络
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Tanh()  # 把虚拟热源的强度严格锁死在 [-1.0, 1.0] 之间
        )

    # 使用 property 强制参数为正数
    @property
    def alpha(self):
        return F.softplus(self.alpha_param)  # 始终 > 0

    def forward(self, x_data, pre_adj, pde_coords_t, sensor_features_precalc=None):
        """
        x_data: (B, C, N, T) 传感器历史数据
        pre_adj: (N, N)
        pde_coords_t: (B, N_pde, 4) 采样的虚拟点坐标 (x, y, z, t)
        sensor_features_precalc: (B, N, 32) 如果传入这个，就跳过内部GWN计算
        注意： pde_coords_t 需要有 Batch 维度，这在 trainer 里需要微调一下
        """

        # 获取 Batch Size
        B = x_data.size(0)
        # 处理 pde_coords_t 的维度问题
        # 如果传入的是 (N_pde, 4)，则自动扩展为 (Batch, N_pde, 4)
        if pde_coords_t.dim() == 2:
            pde_coords_t = pde_coords_t.unsqueeze(0).expand(B, -1, -1)

        # 1. GWN 前向传播
        # y_pred_gwn: (B, N, out_dim) -> 用于 L_data
        # sensor_features: (B, N, 32) -> 用于喂给 Decoder
        if sensor_features_precalc is not None:
            # 如果外面传进来了特征，直接用，不再跑 GWN
            y_pred_gwn = None  # 此时不需要预测值，只需要特征
            sensor_features = sensor_features_precalc
        else:
            # 否则自己算 (兼容旧调用方式)
            y_pred_gwn, sensor_features = self.gwn(x_data, pre_adj)
        # y_pred_gwn, sensor_features = self.gwn(x_data, pre_adj)

        # 2. 准备 Decoder 输入
        # pde_coords_t.requires_grad_(True)
        # 准备物理损失计算
        pde_coords_t = pde_coords_t.clone().detach().requires_grad_(True)
        query_xyz = pde_coords_t[:, :, :3]  # (Batch, N_pde, 3)

        # 分离时空坐标
        # query_xyz = pde_coords_t[:, :, :3]  # (B, N_pde, 3)

        # --- 核心融合步骤: 将 GWN 特征插值到 PDE 采样点 ---
        # 这一步将离散的传感器特征变成了连续场的特征  调用 RBF 插值
        interpolated_feats = self.interpolator(query_xyz, self.sensor_coords, sensor_features)  # (B, N_pde, 32)

        # 拼接: (x,y,z,t) + (features)
        decoder_input = torch.cat([pde_coords_t, interpolated_feats], dim=-1)  # (B, N_pde, 36)

        # 3. Decoder 预测
        temp_pred = self.decoder(decoder_input)  # (B, N_pde, 1)
        temp_pred = temp_pred.squeeze(-1)  # (B, N_pde)

        # 4. 计算物理梯度 (自动微分)
        # 注意: 这里的 grad 是对 pde_coords_t 求导
        # interpolated_feats 也是 pde_coords_t 的函数 (因为 IDW 依赖距离)，
        # 这里的导数会自动包含特征空间变化的影响
        grads = torch.autograd.grad(
            outputs=temp_pred,
            inputs=pde_coords_t,
            grad_outputs=torch.ones_like(temp_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        dT_dx, dT_dy, dT_dz, dT_dt = grads[..., 0], grads[..., 1], grads[..., 2], grads[..., 3]

        # 二阶导
        d2T_dx2 = torch.autograd.grad(dT_dx, pde_coords_t, torch.ones_like(dT_dx), create_graph=True)[0][..., 0]
        d2T_dy2 = torch.autograd.grad(dT_dy, pde_coords_t, torch.ones_like(dT_dy), create_graph=True)[0][..., 1]
        d2T_dz2 = torch.autograd.grad(dT_dz, pde_coords_t, torch.ones_like(dT_dz), create_graph=True)[0][..., 2]

        laplacian = d2T_dx2 + d2T_dy2 + d2T_dz2

        # 提取 PDE 采样点的空间坐标 (B, N_pde, 3)
        # pde_coords_t 的形状是 (B, N_pde, 4)，前三个是 x, y, z
        spatial_xyz = pde_coords_t[..., :3]

        # 根据 use_q_net 开关，决定是动态反演，还是强制 Q=0
        if self.use_q_net:
            # 将空间坐标与 GWN 提取的动态特征拼接！
            # interpolated_feats 包含了当前时刻冷空气灌入的"动态情报" (B, N_pde, 32)
            # q_input = torch.cat([spatial_xyz, interpolated_feats], dim=-1)  # 形状变为 (B, N_pde, 35)
            q_input = torch.cat([spatial_xyz, interpolated_feats.detach()], dim=-1)  # 形状变为 (B, N_pde, 35)

            # 通过 q_net 动态计算每个空间点在 *当前时刻* 的强迫源项
            spatial_q = self.q_net(q_input).squeeze(-1)  # (B, N_pde)
        else:
            B_pde, N_pde, _ = pde_coords_t.shape
            spatial_q = torch.zeros((B_pde, N_pde), device=pde_coords_t.device)  # 静态源变体，强行置 0

        # 物理残差 (带入具有空间感知能力的源项 或 0)
        pde_residual = dT_dt - self.alpha * laplacian - spatial_q

        # # 物理残差
        # pde_residual = dT_dt - self.alpha * laplacian - self.q

        return y_pred_gwn, pde_residual.view(-1, 1)

    def predict(self, x_data, pre_adj, query_coords):
        """
        专用于推理的函数：输出预测的温度值 (真实物理量)，而不是残差。

        Args:
            x_data: (Batch, C, N, T) 传感器历史输入
            pre_adj: (N, N) 邻接矩阵
            query_coords: (Batch, N_points, 4) 查询时空坐标 (x, y, z, t)

        Returns:
            temp_pred: (Batch, N_points) 预测温度值 (归一化后的)
        """
        self.eval()  # 切换到评估模式
        with torch.no_grad():
            # 1. 运行 GWN 提取特征
            # sensor_features: (B, N, 32)
            _, sensor_features = self.gwn(x_data, pre_adj)

            # 2. 空间插值
            # 确保 query_coords 是 3 维 (Batch, N_p, 4)
            if query_coords.dim() == 2:
                query_coords = query_coords.unsqueeze(0)

            query_xyz = query_coords[..., :3]  # (B, N_p, 3)
            interpolated_feats = self.interpolator(query_xyz, self.sensor_coords, sensor_features)

            # 3. 拼接输入 (Coordinate + Feature)
            # Input dim: 4 + 32 = 36
            decoder_input = torch.cat([query_coords, interpolated_feats], dim=-1)

            # 4. MLP 解码预测
            temp_pred = self.decoder(decoder_input)  # (B, N_p, 1)

        return temp_pred.squeeze(-1)


# --- 将 IDW 插值改为 RBF 插值 (彻底解决梯度爆炸) ---
class RBFFeatureInterpolator(nn.Module):
    """
    基于高斯径向基函数 (RBF) 的平滑插值层。
    公式: w = exp(-d^2 / (2*sigma^2))
    优势: 即使查询点和传感器重合，导数也是有限值，永远不会 NaN。
    """

    def __init__(self, sigma=0.2, k=4):
        """
        sigma: 控制平滑程度，0.1-0.5 之间通常效果最好。
        k: 选取最近的 k 个邻居
        """
        super(RBFFeatureInterpolator, self).__init__()
        self.k = k
        # 将 sigma 设为可学习参数，让模型自己调整平滑度
        self.sigma = nn.Parameter(torch.tensor(sigma))

    def forward(self, query_coords, sensor_coords, sensor_features):
        """
        query_coords: (B, N_q, 3)
        sensor_coords: (N_s, 3)
        sensor_features: (B, N_s, F)
        """
        B, N_q, _ = query_coords.shape

        # 扩展 sensor_coords 匹配 batch
        sensor_coords_batch = sensor_coords.unsqueeze(0).expand(B, -1, -1)  # (B, N_s, 3)

        # 1. 计算距离矩阵
        dists = torch.cdist(query_coords, sensor_coords_batch)  # (B, N_q, N_s)

        # 2. 找 Top-K 最近邻
        topk_dists, topk_indices = torch.topk(dists, k=self.k, dim=2, largest=False)

        # 3. 计算高斯权重
        # 使用 softplus 保证 sigma 始终大于 0
        # sigma_safe = F.softplus(self.sigma) + 1e-5
        sigma_safe = F.softplus(self.sigma) + 1e-2
        weights = torch.exp(- (topk_dists ** 2) / (2 * sigma_safe ** 2))

        # 归一化权重 (防止全0除)
        # weights_sum = torch.sum(weights, dim=2, keepdim=True) + 1e-8
        weights_sum = torch.sum(weights, dim=2, keepdim=True) + 1e-5
        norm_weights = weights / weights_sum  # (B, N_q, k)

        # 4. Gather 特征
        # 构造 batch 索引
        batch_indices = torch.arange(B, device=sensor_features.device).view(B, 1, 1).expand(-1, N_q, self.k)
        topk_features = sensor_features[batch_indices, topk_indices]  # (B, N_q, k, F)

        # 5. 加权求和
        interp_features = torch.sum(norm_weights.unsqueeze(-1) * topk_features, dim=2)  # (B, N_q, F)

        return interp_features


class IDWFeatureInterpolator(nn.Module):
    """
    可微分的反距离加权插值模块。
    将 GWN 提取的离散 Sensor Features 插值到任意连续的 (x,y,z) 坐标上。
    """

    def __init__(self, k=3):
        super(IDWFeatureInterpolator, self).__init__()
        self.k = k
        self.epsilon = 1e-8

    def forward(self, query_coords, sensor_coords, sensor_features):
        """
        Args:
            query_coords: (B, N_query, 3) - 我们想查询温度的点 (PDE点)
            sensor_coords: (N_sensor, 3) - 传感器的固定位置 (从 Data Loader 传入)
            sensor_features: (B, N_sensor, Feature_Dim) - GWN 提取的特征
        Returns:
            interp_features: (B, N_query, Feature_Dim)
        """
        B, N_q, _ = query_coords.shape
        N_s, _ = sensor_coords.shape

        # 1. 计算距离 (B, N_query, N_sensor)
        # 扩展维度以利用广播
        sensor_coords_batch = sensor_coords.unsqueeze(0).expand(B, -1, -1)  # (B, N_s, 3)

        # 计算欧氏距离 (使用 torch.cdist 高效计算)
        dists = torch.cdist(query_coords, sensor_coords_batch)  # (B, N_q, N_s)

        # 2. 找到最近的 K 个传感器
        # values: (B, N_q, k), indices: (B, N_q, k)
        topk_dists, topk_indices = torch.topk(dists, k=self.k, dim=2, largest=False)

        # 3. 计算权重 (IDW: w = 1 / d)
        weights = 1.0 / (topk_dists + self.epsilon)
        weights_sum = torch.sum(weights, dim=2, keepdim=True)  # (B, N_q, 1)
        norm_weights = weights / weights_sum  # (B, N_q, k)

        # 4. 提取对应的特征
        # sensor_features: (B, N_s, F)
        # 需要根据 topk_indices 从 sensor_features 中 gather 特征
        # 为了 gather，需要把 indices 扩展到 feature 维度
        F_dim = sensor_features.shape[-1]

        # 构造 batch 索引
        # 这是一个稍微复杂的操作，为了从 (B, N_s, F) 中取出 (B, N_q, k, F)
        batch_indices = torch.arange(B, device=sensor_features.device).view(B, 1, 1).expand(-1, N_q, self.k)

        # Gather 特征: topk_features shape (B, N_q, k, F)
        # 注意：PyTorch 的 gather 比较挑剔，这里使用这种索引方式更直观
        topk_features = sensor_features[batch_indices, topk_indices]

        # 5. 加权求和
        # norm_weights: (B, N_q, k) -> (B, N_q, k, 1)
        interp_features = torch.sum(norm_weights.unsqueeze(-1) * topk_features, dim=2)  # (B, N_q, F)

        return interp_features
