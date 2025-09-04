#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coordinate System Utilities

Provides coordinate transformations and GPS processing functions.
"""

import numpy as np


def gps_to_local_coords(gps_positions, origin_gps):
    """将GPS坐标转换为局部坐标系 (米)"""
    R_earth = 6378137.0  # 地球半径
    
    origin_lat, origin_lon, origin_alt = origin_gps
    lat_rad = np.radians(gps_positions[:, 0])
    lon_rad = np.radians(gps_positions[:, 1])
    alt = gps_positions[:, 2]
    
    origin_lat_rad = np.radians(origin_lat)
    origin_lon_rad = np.radians(origin_lon)
    
    # 转换为局部坐标
    x = (lon_rad - origin_lon_rad) * R_earth * np.cos(origin_lat_rad)
    y = (lat_rad - origin_lat_rad) * R_earth
    z = alt - origin_alt
    
    return np.column_stack([x, y, z])


def estimate_velocity_direction(current_idx, positions, timestamps, window_size=5):
    """根据GPS轨迹估计当前点的速度方向"""
    if len(positions) < 2:
        return None
    
    # 定义前后窗口范围
    start_idx = max(0, current_idx - window_size)
    end_idx = min(len(positions), current_idx + window_size + 1)
    
    # 收集有效的位置差分向量
    direction_vectors = []
    weights = []
    
    for i in range(start_idx + 1, end_idx):
        if i < len(positions):
            # 计算位置差分
            pos_diff = np.array(positions[i][:2]) - np.array(positions[i-1][:2])  # 只取经纬度
            
            # 转换经纬度差分为米制
            # 简化计算：使用中点纬度的Earth radius修正
            lat_mid = (positions[i][0] + positions[i-1][0]) / 2
            R_earth = 6378137.0
            lat_mid_rad = np.radians(lat_mid)
            
            # 转换为东向和北向距离（米）
            east_m = pos_diff[1] * R_earth * np.cos(lat_mid_rad) * np.pi / 180
            north_m = pos_diff[0] * R_earth * np.pi / 180
            
            direction_m = np.array([east_m, north_m])
            distance = np.linalg.norm(direction_m)
            
            # 只考虑有明显移动的点
            if distance > 0.5:  # 至少0.5米移动
                # 计算权重：距离当前点越近权重越大
                distance_to_current = abs(i - current_idx)
                weight = 1.0 / (1.0 + distance_to_current)
                
                direction_vectors.append(direction_m)
                weights.append(weight)
    
    if len(direction_vectors) == 0:
        return None
    
    # 计算加权平均方向
    direction_vectors = np.array(direction_vectors)
    weights = np.array(weights)
    
    weighted_direction = np.sum(direction_vectors * weights[:, np.newaxis], axis=0)
    total_weight = np.sum(weights)
    
    if total_weight > 0:
        return weighted_direction / total_weight
    
    return None