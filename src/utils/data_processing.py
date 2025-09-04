#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Processing Utilities

Provides functions for data validation, interpolation, and synchronization.
"""

import numpy as np
from scipy import interpolate


def validate_timestamps(timestamps, data_name):
    """验证时间戳的有效性"""
    if len(timestamps) == 0:
        raise ValueError(f"{data_name}: 时间戳数组为空")
    
    # 检查单调性
    time_diffs = np.diff(timestamps)
    non_monotonic = np.sum(time_diffs <= 0)
    if non_monotonic > 0:
        print(f"警告: {data_name} 有 {non_monotonic} 个非单调时间戳")
        
    # 检查时间间隔
    if len(time_diffs) > 0:
        avg_dt = np.mean(time_diffs[time_diffs > 0])
        max_dt = np.max(time_diffs)
        min_dt = np.min(time_diffs[time_diffs > 0])
        
        print(f"{data_name} 时间戳统计:")
        print(f"  平均间隔: {avg_dt*1000:.1f}ms")
        print(f"  最大间隔: {max_dt*1000:.1f}ms") 
        print(f"  最小间隔: {min_dt*1000:.1f}ms")
        
        # 检查异常间隔
        large_gaps = time_diffs > avg_dt * 5
        if np.sum(large_gaps) > 0:
            print(f"  发现 {np.sum(large_gaps)} 个大间隔 (>5倍平均值)")
    
    return timestamps


def interpolate_data_advanced(timestamps_target, timestamps_source, data_source, method='linear'):
    """改进的数据插值函数，支持多种插值方法"""
    
    # 验证输入
    if len(timestamps_source) != len(data_source):
        raise ValueError(f"时间戳长度 {len(timestamps_source)} 与数据长度 {len(data_source)} 不匹配")
    
    # 处理单维数据
    if len(data_source.shape) == 1:
        if method == 'linear':
            return np.interp(timestamps_target, timestamps_source, data_source)
        elif method == 'cubic':
            # 三次样条插值（需要至少4个点）
            if len(timestamps_source) >= 4:
                f = interpolate.interp1d(timestamps_source, data_source, kind='cubic', 
                                       bounds_error=False, fill_value='extrapolate')
                return f(timestamps_target)
            else:
                return np.interp(timestamps_target, timestamps_source, data_source)
    
    # 处理多维数据
    result = np.zeros((len(timestamps_target), data_source.shape[1]))
    for i in range(data_source.shape[1]):
        if method == 'linear':
            result[:, i] = np.interp(timestamps_target, timestamps_source, data_source[:, i])
        elif method == 'cubic' and len(timestamps_source) >= 4:
            f = interpolate.interp1d(timestamps_source, data_source[:, i], kind='cubic',
                                   bounds_error=False, fill_value='extrapolate')
            result[:, i] = f(timestamps_target)
        else:
            result[:, i] = np.interp(timestamps_target, timestamps_source, data_source[:, i])
    
    return result


def synchronize_sensors(gps_times, gps_data, imu_times, imu_data, 
                       gps_delay=0.0, imu_delay=0.0):
    """改进的传感器时间同步"""
    
    print("开始传感器时间同步...")
    
    # 验证时间戳
    gps_times_clean = validate_timestamps(gps_times, "GPS")
    imu_times_clean = validate_timestamps(imu_times, "IMU")
    
    # 应用传感器延迟补偿
    gps_times_compensated = gps_times_clean + gps_delay
    imu_times_compensated = imu_times_clean + imu_delay
    
    if gps_delay != 0:
        print(f"应用GPS延迟补偿: {gps_delay*1000:.1f}ms")
    if imu_delay != 0:
        print(f"应用IMU延迟补偿: {imu_delay*1000:.1f}ms")
    
    # 找到公共时间范围（加入缓冲区）
    start_time = max(gps_times_compensated[0], imu_times_compensated[0])
    end_time = min(gps_times_compensated[-1], imu_times_compensated[-1])
    
    # 检查时间重叠
    if start_time >= end_time:
        raise ValueError(f"GPS和IMU数据没有时间重叠: GPS[{gps_times_compensated[0]:.1f}, {gps_times_compensated[-1]:.1f}], IMU[{imu_times_compensated[0]:.1f}, {imu_times_compensated[-1]:.1f}]")
    
    # 创建融合时间戳（以IMU频率为主，但去除异常值）
    imu_dts = np.diff(imu_times_compensated)
    valid_dts = imu_dts[(imu_dts > 0) & (imu_dts < np.percentile(imu_dts, 95))]
    dt_imu = np.median(valid_dts)
    
    # 确保时间范围合理
    max_duration = min(3600, end_time - start_time)  # 最多1小时
    if end_time - start_time > max_duration:
        end_time = start_time + max_duration
        print(f"限制融合时间范围为 {max_duration/60:.1f} 分钟")
    
    fusion_times = np.arange(start_time, end_time, dt_imu)
    
    print(f"同步结果:")
    print(f"  融合时间范围: {start_time:.1f} - {end_time:.1f} 秒 (时长: {end_time-start_time:.1f}秒)")
    print(f"  IMU采样间隔: {dt_imu*1000:.1f} ms")
    print(f"  融合数据点数: {len(fusion_times)}")
    
    return fusion_times, gps_times_compensated, imu_times_compensated


def clean_sensor_data(times, data, data_name):
    """清理传感器数据中的NaN和异常值"""
    print(f"清理{data_name}数据...")
    
    original_count = len(times)
    if original_count == 0:
        return times, data
    
    # 检查时间戳
    time_valid = ~(np.isnan(times) | np.isinf(times))
    
    # 检查数据
    if len(data.shape) == 1:
        data_valid = ~(np.isnan(data) | np.isinf(data))
    else:
        data_valid = ~(np.isnan(data).any(axis=1) | np.isinf(data).any(axis=1))
    
    # 综合有效性
    valid_mask = time_valid & data_valid
    valid_count = np.sum(valid_mask)
    
    if valid_count == 0:
        raise ValueError(f"{data_name}: 所有数据都无效(NaN/Inf)")
    
    # 过滤数据
    clean_times = times[valid_mask]
    clean_data = data[valid_mask]
    
    removed_count = original_count - valid_count
    if removed_count > 0:
        print(f"  移除了 {removed_count}/{original_count} 个无效数据点 ({removed_count/original_count*100:.1f}%)")
        
        # 检查时间间隔是否合理
        if len(clean_times) > 1:
            time_gaps = np.diff(clean_times)
            large_gaps = time_gaps > np.median(time_gaps) * 5
            if np.sum(large_gaps) > 0:
                print(f"  警告: 发现 {np.sum(large_gaps)} 个大时间间隔 (可能影响插值)")
    
    return clean_times, clean_data


def validate_and_fix_rotation_matrix(R):
    """验证和修复旋转矩阵"""
    if np.any(np.isnan(R)) or np.any(np.isinf(R)):
        print("警告: 旋转矩阵包含NaN/Inf，使用单位矩阵")
        return np.eye(3)
    
    # 检查是否接近正交矩阵
    should_be_identity = R @ R.T
    identity_error = np.max(np.abs(should_be_identity - np.eye(3)))
    
    if identity_error > 0.1:
        print(f"警告: 旋转矩阵不正交 (误差: {identity_error:.3f})，进行修正")
        # 使用SVD修正
        U, _, Vt = np.linalg.svd(R)
        R_corrected = U @ Vt
        
        # 确保行列式为+1 (右手坐标系)
        if np.linalg.det(R_corrected) < 0:
            U[:, -1] *= -1
            R_corrected = U @ Vt
            
        return R_corrected
    
    return R