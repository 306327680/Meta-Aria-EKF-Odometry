#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EKF Fusion Pipeline

Orchestrates the complete sensor fusion process from raw data to results.
"""

import numpy as np
from .ekf import EKF
from ..utils.data_processing import clean_sensor_data, synchronize_sensors, interpolate_data_advanced
from ..utils.calibration import estimate_global_extrinsics


def run_ekf_fusion(gps_times, gps_positions, gps_accuracies, gps_velocities, 
                   imu_times, imu_accels, imu_gyros, 
                   mag_times=None, mag_fields=None, 
                   baro_times=None, baro_pressures=None, baro_temps=None,
                   gps_vertical_accuracies=None):
    """运行EKF融合算法"""
    
    print("开始EKF融合算法...")
    
    # 步骤0: 数据清理
    print("清理输入数据...")
    
    # 确保GPS数据的时间戳、位置、精度和速度长度一致
    data_arrays = [gps_times, gps_positions, gps_accuracies, gps_velocities]
    if gps_vertical_accuracies is not None:
        data_arrays.append(gps_vertical_accuracies)
        
    min_gps_len = min(len(arr) for arr in data_arrays)
    gps_times_trim = gps_times[:min_gps_len]
    gps_positions_trim = gps_positions[:min_gps_len]  
    gps_accuracies_trim = gps_accuracies[:min_gps_len]
    gps_velocities_trim = gps_velocities[:min_gps_len]
    gps_vertical_accuracies_trim = gps_vertical_accuracies[:min_gps_len] if gps_vertical_accuracies is not None else None
    
    print(f"GPS数据长度调整: times={len(gps_times)}→{min_gps_len}, pos={len(gps_positions)}→{min_gps_len}, acc={len(gps_accuracies)}→{min_gps_len}, vel={len(gps_velocities)}→{min_gps_len}")
    
    # 清理GPS数据（保持一致性）
    gps_times_clean, gps_positions_clean = clean_sensor_data(gps_times_trim, gps_positions_trim, "GPS位置")
    
    # 使用相同的有效掩码清理精度和速度数据
    valid_gps_mask = ~(np.isnan(gps_times_trim) | np.isinf(gps_times_trim) | 
                       np.isnan(gps_positions_trim).any(axis=1) | np.isinf(gps_positions_trim).any(axis=1))
    
    gps_accuracies_clean = gps_accuracies_trim[valid_gps_mask]
    gps_velocities_clean = gps_velocities_trim[valid_gps_mask]
    gps_vertical_accuracies_clean = gps_vertical_accuracies_trim[valid_gps_mask] if gps_vertical_accuracies_trim is not None else None
    
    print(f"GPS数据清理后: {len(gps_times_clean)} 个有效数据点")
    
    # 步骤1: 全局外参估计
    imu_to_world_rotation, gyro_bias, initial_yaw, measured_gravity = estimate_global_extrinsics(
        gps_times_clean, gps_positions_clean, imu_times, imu_accels, imu_gyros
    )
    
    # 步骤2: 传感器时间同步（使用清理后的数据）
    fusion_times, gps_times_sync, imu_times_sync = synchronize_sensors(
        gps_times_clean, gps_positions_clean, imu_times, imu_accels
    )
    
    # 步骤3: 数据插值
    print("开始数据插值...")
    print(f"调试信息:")
    print(f"  fusion_times长度: {len(fusion_times)}")
    print(f"  gps_times_sync长度: {len(gps_times_sync)}")  
    print(f"  gps_positions_clean长度: {len(gps_positions_clean)}")
    print(f"  gps_accuracies_clean长度: {len(gps_accuracies_clean)}")
    print(f"  imu_times_sync长度: {len(imu_times_sync)}")
    print(f"  imu_accels长度: {len(imu_accels)}")
    
    # 确保使用相同长度的清理后数据
    try:
        interp_gps_pos = interpolate_data_advanced(fusion_times, gps_times_sync, gps_positions_clean, method='linear')
        interp_gps_acc = interpolate_data_advanced(fusion_times, gps_times_sync, gps_accuracies_clean, method='linear')
        
        # 插值垂直精度数据
        interp_gps_vertical_acc = None
        if gps_vertical_accuracies_clean is not None:
            interp_gps_vertical_acc = interpolate_data_advanced(fusion_times, gps_times_sync, gps_vertical_accuracies_clean, method='linear')
        
        # IMU数据使用三次样条插值以保持平滑性
        interp_imu_accel = interpolate_data_advanced(fusion_times, imu_times_sync, imu_accels, method='cubic')
        interp_imu_gyro = interpolate_data_advanced(fusion_times, imu_times_sync, imu_gyros, method='cubic')
        
        print(f"插值后数据长度:")
        print(f"  interp_gps_pos: {interp_gps_pos.shape}")
        print(f"  interp_gps_acc: {interp_gps_acc.shape}")
        print(f"  interp_imu_accel: {interp_imu_accel.shape}")
        print(f"  interp_imu_gyro: {interp_imu_gyro.shape}")
        
        # 检查数据长度一致性
        expected_len = len(fusion_times)
        data_lengths = [len(interp_gps_pos), len(interp_gps_acc), len(interp_imu_accel), len(interp_imu_gyro)]
        if interp_gps_vertical_acc is not None:
            data_lengths.append(len(interp_gps_vertical_acc))
            
        if any(length != expected_len for length in data_lengths):
            print(f"警告: 插值后数据长度不一致，预期长度: {expected_len}")
            print("截断到最短长度...")
            
            min_len = min(data_lengths + [expected_len])
            
            fusion_times = fusion_times[:min_len]
            interp_gps_pos = interp_gps_pos[:min_len]
            interp_gps_acc = interp_gps_acc[:min_len]
            interp_imu_accel = interp_imu_accel[:min_len]
            interp_imu_gyro = interp_imu_gyro[:min_len]
            if interp_gps_vertical_acc is not None:
                interp_gps_vertical_acc = interp_gps_vertical_acc[:min_len]
            
            print(f"调整后统一长度: {min_len}")
        
    except Exception as e:
        print(f"插值过程出错: {e}")
        print("尝试使用原始数据进行插值...")
        
        # 回退到使用原始数据
        try:
            interp_gps_pos = interpolate_data_advanced(fusion_times, gps_times, gps_positions, method='linear')
            interp_gps_acc = interpolate_data_advanced(fusion_times, gps_times, gps_accuracies, method='linear')
            interp_imu_accel = interpolate_data_advanced(fusion_times, imu_times, imu_accels, method='cubic')
            interp_imu_gyro = interpolate_data_advanced(fusion_times, imu_times, imu_gyros, method='cubic')
        except Exception as e2:
            print(f"原始数据插值也失败: {e2}")
            print("使用线性插值作为最后回退...")
            
            interp_gps_pos = interpolate_data_advanced(fusion_times, gps_times, gps_positions, method='linear')
            interp_gps_acc = interpolate_data_advanced(fusion_times, gps_times, gps_accuracies, method='linear') 
            interp_imu_accel = interpolate_data_advanced(fusion_times, imu_times, imu_accels, method='linear')
            interp_imu_gyro = interpolate_data_advanced(fusion_times, imu_times, imu_gyros, method='linear')
    
    # 步骤4: 初始化EKF（使用预估计的外参）
    ekf = EKF(imu_to_world_rotation=imu_to_world_rotation, gyro_bias=gyro_bias)
    
    # 设置重力向量
    if len(gps_positions_clean) > 0:
        origin_lat = gps_positions_clean[0][0]  # GPS纬度
        actual_gravity = ekf.set_gravity_vector(gravity_magnitude=measured_gravity, latitude=origin_lat)
    else:
        actual_gravity = ekf.set_gravity_vector(gravity_magnitude=measured_gravity)
    
    # 插值气压计数据（需要在初始化之前完成）
    interp_baro_pressures = None
    interp_baro_temps = None
    if baro_times is not None and baro_pressures is not None:
        try:
            print("插值气压计数据...")
            interp_baro_pressures = interpolate_data_advanced(fusion_times, baro_times, baro_pressures, method='linear')
            if baro_temps is not None:
                interp_baro_temps = interpolate_data_advanced(fusion_times, baro_times, baro_temps, method='linear')
            print(f"气压计插值成功: {interp_baro_pressures.shape}")
        except Exception as e:
            print(f"气压计插值失败: {e}")
            interp_baro_pressures = None
            interp_baro_temps = None
    
    # 初始化状态（使用改进的初始化方法）
    # 计算初始GPS速度（如果插值数据可用）
    initial_gps_vel = None
    if len(gps_velocities_clean) > 0:
        # 插值GPS速度到fusion_times
        try:
            interp_gps_vel = interpolate_data_advanced(fusion_times, gps_times_clean, gps_velocities_clean, method='linear')
            initial_gps_vel = interp_gps_vel[0]
        except:
            initial_gps_vel = gps_velocities_clean[0]  # 使用原始第一个速度
    
    # 使用GPS位置初始化，但记录气压计基准用于相对约束
    stable_init_pos = interp_gps_pos[0].copy()  # 使用GPS的XYZ位置
    
    # 记录气压计初始基准（用于后续相对高度约束）
    barometer_baseline = None
    if interp_baro_pressures is not None and len(interp_baro_pressures) > 0:
        barometer_baseline = interp_baro_pressures[0]
        print(f"EKF初始化:")
        print(f"  使用GPS位置: [{stable_init_pos[0]:.2f}, {stable_init_pos[1]:.2f}, {stable_init_pos[2]:.2f}]m") 
        print(f"  气压计基准: {barometer_baseline:.2f} kPa (用于相对高度约束)")
    else:
        print(f"EKF初始化:")
        print(f"  使用GPS位置: [{stable_init_pos[0]:.2f}, {stable_init_pos[1]:.2f}, {stable_init_pos[2]:.2f}]m")
        print(f"  气压计不可用")
    
    stable_init_gps_subset = interp_gps_pos[:min(10, len(interp_gps_pos))]
    stable_init_times_subset = fusion_times[:min(10, len(fusion_times))]
    
    print(f"  用于初始化的GPS点数: {len(stable_init_gps_subset)}")
    
    ekf.initialize_state(stable_init_pos, initial_gps_vel, interp_imu_accel[0], 
                        stable_init_gps_subset, stable_init_times_subset)
    
    # 存储结果
    ekf_positions = [ekf.get_position()]
    ekf_velocities = [ekf.get_velocity()]  
    ekf_attitudes = [ekf.get_attitude()]
    
    # 插值GPS速度数据
    print("插值GPS速度数据...")
    interp_gps_vel = None
    if len(gps_velocities_clean) > 0:
        try:
            interp_gps_vel = interpolate_data_advanced(fusion_times, gps_times_clean, gps_velocities_clean, method='linear')
            print(f"GPS速度插值成功: {interp_gps_vel.shape}")
        except Exception as e:
            print(f"GPS速度插值失败: {e}，EKF将仅使用位置观测")
            interp_gps_vel = None
    else:
        print("无GPS速度数据，EKF将仅使用位置观测")
        
    # 插值磁力计数据
    interp_mag_fields = None
    if mag_times is not None and mag_fields is not None:
        try:
            print("插值磁力计数据...")
            interp_mag_fields = interpolate_data_advanced(fusion_times, mag_times, mag_fields, method='linear')
            print(f"磁力计插值成功: {interp_mag_fields.shape}")
        except Exception as e:
            print(f"磁力计插值失败: {e}")
            interp_mag_fields = None
    
    # 气压计数据已在初始化前完成插值

    # 步骤5: 主融合循环
    print("开始EKF融合...")
    for i in range(1, len(fusion_times)):
        t = fusion_times[i]
        dt = t - fusion_times[i-1]
        
        # EKF预测步骤 (使用IMU数据)
        ekf.predict(interp_imu_accel[i], interp_imu_gyro[i], dt, current_time=t)
        
        # EKF更新步骤 (使用GPS数据)
        # 降低GPS更新频率以模拟实际情况
        if i % 10 == 0:  # 每10个IMU周期更新一次GPS
            # 使用GPS位置和速度（如果可用）进行更新
            gps_vel_for_update = interp_gps_vel[i] if interp_gps_vel is not None else None
            gps_vertical_acc_for_update = interp_gps_vertical_acc[i] if interp_gps_vertical_acc is not None else None
            ekf.update_gps(interp_gps_pos[i], interp_gps_acc[i], gps_vel_for_update, 
                          gps_vertical_accuracy=gps_vertical_acc_for_update)
        
        # 磁力计更新 (如果可用，频率较低)
        if interp_mag_fields is not None and i % 20 == 0:  # 每20个IMU周期更新一次磁力计
            ekf.update_magnetometer(interp_mag_fields[i], mag_noise_std=0.1)
        
        # 气压计相对约束 (如果可用，频率较低)
        if interp_baro_pressures is not None and barometer_baseline is not None and i % 15 == 0:
            temp = interp_baro_temps[i] if interp_baro_temps is not None else 20.0
            ekf.update_barometer_relative(interp_baro_pressures[i], barometer_baseline, 
                                        temperature=temp, pressure_noise_std=0.05)
        
        # 保存结果
        ekf_positions.append(ekf.get_position())
        ekf_velocities.append(ekf.get_velocity())
        ekf_attitudes.append(ekf.get_attitude())
    
    print(f"EKF融合完成，处理了 {len(ekf_positions)} 个数据点")
    
    return (fusion_times,
            np.array(ekf_positions),
            np.array(ekf_velocities), 
            np.array(ekf_attitudes),
            interp_gps_pos,
            interp_gps_vel,
            interp_mag_fields,
            interp_baro_pressures,
            interp_baro_temps,
            interp_gps_acc,
            interp_gps_vertical_acc)