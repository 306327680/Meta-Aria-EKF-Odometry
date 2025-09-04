#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensor Calibration Utilities

Provides functions for estimating sensor extrinsics and bias parameters.
"""

import numpy as np
from .data_processing import clean_sensor_data, validate_and_fix_rotation_matrix
from .angles import safe_arctan2


def estimate_global_extrinsics(gps_times, gps_positions, imu_times, imu_accels, imu_gyros):
    """使用全部数据估计IMU到世界坐标系的外参"""
    print("开始全局外参估计...")
    
    # 数据清理
    gps_times_clean, gps_positions_clean = clean_sensor_data(gps_times, gps_positions, "GPS")
    imu_times_clean, imu_accels_clean = clean_sensor_data(imu_times, imu_accels, "IMU加速度")
    imu_times_clean, imu_gyros_clean = clean_sensor_data(imu_times, imu_gyros, "IMU陀螺仪")
    
    # 1. 估计陀螺仪偏置（假设整体平均角速度接近0）
    gyro_bias = np.nanmean(imu_gyros_clean, axis=0)
    
    # 检查偏置是否有效
    if np.any(np.isnan(gyro_bias)) or np.any(np.abs(gyro_bias) > 1.0):  # 偏置不应超过1 rad/s
        print("警告: 陀螺仪偏置异常，使用零偏置")
        gyro_bias = np.zeros(3)
    
    print(f"陀螺仪偏置估计: [{gyro_bias[0]:.4f}, {gyro_bias[1]:.4f}, {gyro_bias[2]:.4f}] rad/s")
    
    # 去除偏置
    imu_gyros_corrected = imu_gyros_clean - gyro_bias
    
    # 2. 使用GPS轨迹估计初始航向角
    initial_yaw = 0.0
    if len(gps_positions_clean) > 10:
        # 使用前20%的轨迹计算平均运动方向
        end_idx = min(len(gps_positions_clean), max(10, len(gps_positions_clean) // 5))
        
        total_displacement = np.zeros(2)
        total_weight = 0
        
        for i in range(1, end_idx):
            pos_diff = gps_positions_clean[i] - gps_positions_clean[i-1]
            
            # 检查位置差是否有效
            if np.any(np.isnan(pos_diff)):
                continue
                
            weight = np.linalg.norm(pos_diff[:2])  # 距离作为权重
            
            if weight > 0.5:  # 只考虑明显运动
                total_displacement += pos_diff[:2] * weight
                total_weight += weight
        
        if total_weight > 0 and not np.any(np.isnan(total_displacement)):
            avg_direction = total_displacement / total_weight
            initial_yaw = safe_arctan2(avg_direction[0], avg_direction[1])  # atan2(East, North)
            print(f"初始航向估计: {np.degrees(initial_yaw):.1f}° (从GPS轨迹)")
        else:
            print("无法从GPS轨迹估计航向角，使用0°")
    
    # 3. 使用低通滤波估计重力方向
    imu_to_world_rotation = np.eye(3)  # 默认值
    
    if len(imu_accels_clean) > 100:
        try:
            # 简单的移动平均滤波
            window_size = min(50, len(imu_accels_clean) // 10)
            filtered_accels = []
            
            for i in range(len(imu_accels_clean)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(imu_accels_clean), i + window_size // 2 + 1)
                window_data = imu_accels_clean[start_idx:end_idx]
                
                # 使用nanmean处理可能的NaN
                filtered_accel = np.nanmean(window_data, axis=0)
                if not np.any(np.isnan(filtered_accel)):
                    filtered_accels.append(filtered_accel)
            
            if len(filtered_accels) == 0:
                print("警告: 滤波后无有效加速度数据，使用单位矩阵")
            else:
                filtered_accels = np.array(filtered_accels)
                
                # 使用中段数据估计重力方向（避免开始和结束的不稳定）
                mid_start = len(filtered_accels) // 4
                mid_end = 3 * len(filtered_accels) // 4
                mid_accels = filtered_accels[mid_start:mid_end]
                
                mean_accel = np.nanmean(mid_accels, axis=0)
                
                if np.any(np.isnan(mean_accel)):
                    print("警告: 平均加速度包含NaN，使用单位矩阵")
                else:
                    gravity_magnitude = np.linalg.norm(mean_accel)
                    
                    print(f"重力向量估计: [{mean_accel[0]:.2f}, {mean_accel[1]:.2f}, {mean_accel[2]:.2f}] m/s²")
                    print(f"重力模长: {gravity_magnitude:.2f} m/s²")
                    
                    if 5.0 < gravity_magnitude < 15.0:  # 合理的重力范围
                        gravity_imu = mean_accel / gravity_magnitude
                        
                        # 理想的重力方向 (ENU坐标系中向下为负Z)
                        gravity_world = np.array([0, 0, -1])  # ENU: 重力向下
                        
                        # 计算旋转矩阵来对齐重力向量
                        v = np.cross(gravity_imu, gravity_world)
                        s = np.linalg.norm(v)
                        c = np.dot(gravity_imu, gravity_world)
                        
                        if s > 1e-6:  # 避免除零
                            vx = np.array([[0, -v[2], v[1]],
                                          [v[2], 0, -v[0]],
                                          [-v[1], v[0], 0]])
                            
                            imu_to_world_rotation = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
                            
                            # 验证和修复旋转矩阵
                            imu_to_world_rotation = validate_and_fix_rotation_matrix(imu_to_world_rotation)
                        else:
                            print("重力向量已对齐，使用单位矩阵")
                    else:
                        print(f"警告: 重力向量异常 ({gravity_magnitude:.2f} m/s²)，使用单位矩阵")
        except Exception as e:
            print(f"重力向量估计出错: {e}，使用单位矩阵")
            imu_to_world_rotation = np.eye(3)
    else:
        print("警告: IMU数据不足，使用单位矩阵")
    
    # 4. 进一步优化外参（可选：基于GPS-IMU一致性）
    # 提取实际测量的重力大小
    measured_gravity = None
    if len(imu_accels_clean) > 100:
        try:
            window_size = min(50, len(imu_accels_clean) // 10)
            filtered_accels = []
            for i in range(len(imu_accels_clean)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(imu_accels_clean), i + window_size // 2 + 1)
                window_data = imu_accels_clean[start_idx:end_idx]
                filtered_accel = np.nanmean(window_data, axis=0)
                if not np.any(np.isnan(filtered_accel)):
                    filtered_accels.append(filtered_accel)
            
            if len(filtered_accels) > 0:
                filtered_accels = np.array(filtered_accels)
                mid_start = len(filtered_accels) // 4
                mid_end = 3 * len(filtered_accels) // 4
                mid_accels = filtered_accels[mid_start:mid_end]
                mean_accel = np.nanmean(mid_accels, axis=0)
                
                if not np.any(np.isnan(mean_accel)):
                    measured_gravity = np.linalg.norm(mean_accel)
        except:
            pass
    
    print("外参估计完成:")
    print(f"  旋转矩阵条件数: {np.linalg.cond(imu_to_world_rotation):.2f}")
    print(f"  是否正交: {np.allclose(imu_to_world_rotation @ imu_to_world_rotation.T, np.eye(3))}")
    if measured_gravity is not None:
        print(f"  测量重力大小: {measured_gravity:.3f} m/s²")
    
    return imu_to_world_rotation, gyro_bias, initial_yaw, measured_gravity