#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPS Data Reader for Project Aria VRS Files

Handles reading GPS position, accuracy, and velocity data with intelligent
speed vector estimation from trajectory when bearing is not available.
"""

import numpy as np
from ..utils.coordinates import estimate_velocity_direction


def read_gps_data(provider, downsample=1):
    """读取GPS数据，包括速度信息"""
    try:
        stream_id = provider.get_stream_id_from_label("gps")
    except Exception:
        raise RuntimeError("未找到GPS流")

    num_samples = provider.get_num_data(stream_id)
    if num_samples == 0:
        raise RuntimeError("GPS流中没有数据")

    timestamps = []
    positions = []
    accuracies = []
    vertical_accuracies = []
    velocities = []

    for idx in range(0, num_samples, downsample):
        try:
            gps_data = provider.get_gps_data_by_index(stream_id, idx)
            
            # 调试：检查GPS数据结构（仅第一个数据点）
            if idx == 0:
                print(f"GPS数据字段检查:")
                all_attrs = [attr for attr in dir(gps_data) if not attr.startswith('_')]
                print(f"  可用属性: {all_attrs}")
                
                # 检查速度相关字段
                velocity_fields = ['speed', 'velocity', 'bearing', 'course', 'vx', 'vy', 'vz']
                for field in velocity_fields:
                    if hasattr(gps_data, field):
                        value = getattr(gps_data, field)
                        print(f"    {field}: {value}")
            
            lat = gps_data.latitude
            lon = gps_data.longitude  
            alt = gps_data.altitude
            acc = gps_data.accuracy
            v_acc = gps_data.verticalAccuracy if hasattr(gps_data, 'verticalAccuracy') else acc  # 回退到总体精度
            timestamp = gps_data.capture_timestamp_ns * 1e-9

            positions.append([lat, lon, alt])
            accuracies.append(acc)
            vertical_accuracies.append(v_acc)
            timestamps.append(timestamp)
            
            # 尝试获取速度信息
            vel = np.zeros(3)  # 默认值
            
            if hasattr(gps_data, 'speed') and hasattr(gps_data, 'bearing'):
                # 从速度和方位角计算速度向量
                speed = gps_data.speed  # m/s
                bearing = gps_data.bearing  # 通常是度数
                
                if not (np.isnan(speed) or np.isnan(bearing)) and speed > 0.1:
                    bearing_rad = np.radians(bearing)
                    vel[0] = speed * np.sin(bearing_rad)  # East
                    vel[1] = speed * np.cos(bearing_rad)  # North
                    vel[2] = 0.0  # 垂直速度通常不可用
                    
                    if idx == 0:  # 调试第一个速度
                        print(f"    速度计算: speed={speed:.2f}m/s, bearing={bearing:.1f}°")
                        print(f"    速度向量: East={vel[0]:.2f}, North={vel[1]:.2f}")
                        
            elif hasattr(gps_data, 'speed') and not hasattr(gps_data, 'bearing'):
                # 只有速度大小，没有方位角，需要从位置差分估计方向
                speed = gps_data.speed
                
                if idx == 0:
                    print(f"    检测到GPS速度: {speed:.2f}m/s (无方位角，稍后从轨迹估计方向)")
                
                if not np.isnan(speed) and speed > 0.1:
                    # 暂时存储速度大小，方向将在后处理中从GPS轨迹估计
                    # 这里先存储为标量形式，后面会转换
                    vel = [speed, 0, 0]  # 临时存储格式：[速度大小, 0, 0]
                    
            elif hasattr(gps_data, 'vx') and hasattr(gps_data, 'vy'):
                # 直接的速度分量
                vel[0] = getattr(gps_data, 'vx', 0.0)
                vel[1] = getattr(gps_data, 'vy', 0.0) 
                vel[2] = getattr(gps_data, 'vz', 0.0)
                
                if idx == 0 and (abs(vel[0]) > 0.1 or abs(vel[1]) > 0.1):
                    print(f"    直接速度: vx={vel[0]:.2f}, vy={vel[1]:.2f}, vz={vel[2]:.2f}")
            
            velocities.append(vel)
            
            # 调试：检查是否有有效速度
            if idx == 0:
                speed_mag = np.linalg.norm(vel[:2])
                print(f"    最终速度向量: [{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}] m/s")
                print(f"    2D速度大小: {speed_mag:.2f} m/s")
            
        except Exception as e:
            print(f"读取GPS数据错误 (索引 {idx}): {e}")
            continue

    velocities = np.array(velocities)
    
    # 后处理：如果速度向量只有大小没有方向，从GPS轨迹估计方向
    if len(velocities) > 0 and velocities.shape[1] == 3:
        needs_direction_estimation = False
        
        for i in range(len(velocities)):
            # 检查是否是临时存储格式 [速度大小, 0, 0]
            if velocities[i, 0] > 0.1 and velocities[i, 1] == 0.0 and velocities[i, 2] == 0.0:
                needs_direction_estimation = True
                break
        
        if needs_direction_estimation:
            print("后处理：从GPS轨迹估计速度方向...")
            positions_array = np.array(positions)
            timestamps_array = np.array(timestamps)
            
            # 计算GPS轨迹的局部运动方向
            for i in range(len(velocities)):
                if velocities[i, 0] > 0.1 and velocities[i, 1] == 0.0:  # 需要估计方向
                    speed_magnitude = velocities[i, 0]
                    
                    # 使用前后几个点的平均运动方向
                    direction_vector = estimate_velocity_direction(i, positions_array, timestamps_array)
                    
                    if direction_vector is not None:
                        # 归一化方向向量并乘以速度大小
                        direction_norm = np.linalg.norm(direction_vector)
                        if direction_norm > 0:
                            unit_direction = direction_vector / direction_norm
                            velocities[i, 0] = speed_magnitude * unit_direction[0]  # East
                            velocities[i, 1] = speed_magnitude * unit_direction[1]  # North
                            velocities[i, 2] = 0.0  # 垂直分量
                            
                            if i == 0:  # 调试第一个估计结果
                                print(f"    第一个速度向量估计结果: East={velocities[i, 0]:.2f}, North={velocities[i, 1]:.2f} m/s")
    
    return (np.array(timestamps),
            np.array(positions),
            np.array(accuracies),
            np.array(vertical_accuracies),
            velocities)