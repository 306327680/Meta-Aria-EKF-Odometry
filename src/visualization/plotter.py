#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EKF Fusion Results Visualization

Provides comprehensive plotting functions for EKF vs GPS comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ..utils.angles import unwrap_angles


def calculate_trajectory_length(positions):
    """计算轨迹总长度"""
    if len(positions) < 2:
        return 0.0
    
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return np.sum(distances)


def plot_comparison(fusion_times, ekf_positions, gps_positions, ekf_attitudes, out_prefix, 
                   ekf_velocities=None, gps_velocities=None, magnetometer_data=None, barometer_data=None):
    """绘制EKF前后对比图"""
    
    # 创建主图表 (4x3布局以容纳磁力计和气压计图表)
    fig, axes = plt.subplots(4, 3, figsize=(20, 20))
    fig.suptitle('GPS/IMU EKF Fusion Results with Magnetometer & Barometer', fontsize=16, y=0.995)
    
    # 2D轨迹对比
    axes[0, 0].plot(gps_positions[:, 0], gps_positions[:, 1], 'r-', alpha=0.7, label='GPS Only')
    axes[0, 0].plot(ekf_positions[:, 0], ekf_positions[:, 1], 'b-', linewidth=2, label='EKF Fused')
    axes[0, 0].scatter(gps_positions[0, 0], gps_positions[0, 1], c='green', s=100, marker='s', label='Start')
    axes[0, 0].scatter(gps_positions[-1, 0], gps_positions[-1, 1], c='red', s=100, marker='^', label='End')
    axes[0, 0].set_xlabel('East (m)')
    axes[0, 0].set_ylabel('North (m)')
    axes[0, 0].set_title('2D Trajectory Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')
    
    # 3D轨迹对比
    ax_3d = fig.add_subplot(4, 3, 2, projection='3d')
    ax_3d.plot(gps_positions[:, 0], gps_positions[:, 1], gps_positions[:, 2], 
               'r-', alpha=0.7, label='GPS Only')
    ax_3d.plot(ekf_positions[:, 0], ekf_positions[:, 1], ekf_positions[:, 2], 
               'b-', linewidth=2, label='EKF Fused')
    
    # 添加XY平面投影
    z_min = min(min(gps_positions[:, 2]), min(ekf_positions[:, 2]))
    z_range = max(max(gps_positions[:, 2]), max(ekf_positions[:, 2])) - z_min
    projection_z = z_min - 0.1 * z_range if z_range > 0 else z_min - 1
    
    # 绘制XY平面投影轨迹
    ax_3d.plot(gps_positions[:, 0], gps_positions[:, 1], projection_z, 
               'r-', alpha=0.4, linewidth=1, label='GPS Projection')
    ax_3d.plot(ekf_positions[:, 0], ekf_positions[:, 1], projection_z, 
               'b-', alpha=0.4, linewidth=1, label='EKF Projection')
    
    ax_3d.set_xlabel('East (m)')
    ax_3d.set_ylabel('North (m)')
    ax_3d.set_zlabel('Up (m)')
    ax_3d.set_title('3D Trajectory with XY Projection')
    ax_3d.legend()
    
    # 姿态角时间序列（解决角度绕圈问题）
    roll_deg_unwrapped = unwrap_angles(np.degrees(ekf_attitudes[:, 0]))
    pitch_deg_unwrapped = unwrap_angles(np.degrees(ekf_attitudes[:, 1]))
    yaw_deg_unwrapped = unwrap_angles(np.degrees(ekf_attitudes[:, 2]))
    
    axes[0, 2].plot(fusion_times, roll_deg_unwrapped, 'r-', label='Roll')
    axes[0, 2].plot(fusion_times, pitch_deg_unwrapped, 'g-', label='Pitch') 
    axes[0, 2].plot(fusion_times, yaw_deg_unwrapped, 'b-', label='Yaw')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Angle (degrees, unwrapped)')
    axes[0, 2].set_title('EKF Attitude Estimates (Roll/Pitch/Yaw)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 位置误差时间序列
    position_error = np.linalg.norm(ekf_positions - gps_positions, axis=1)
    axes[1, 0].plot(fusion_times, position_error, 'g-', linewidth=1)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Position Error (m)')
    axes[1, 0].set_title('EKF vs GPS Position Difference')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 各轴位置对比
    axes[1, 1].plot(fusion_times, gps_positions[:, 0], 'r--', alpha=0.7, label='GPS East')
    axes[1, 1].plot(fusion_times, ekf_positions[:, 0], 'b-', label='EKF East')
    axes[1, 1].plot(fusion_times, gps_positions[:, 1], 'r:', alpha=0.7, label='GPS North')  
    axes[1, 1].plot(fusion_times, ekf_positions[:, 1], 'g-', label='EKF North')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Position (m)')
    axes[1, 1].set_title('Position Components vs Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 航向角对比 (从GPS速度计算 vs EKF估计)
    gps_headings = []
    
    # 直接用前后两个点差值计算航向
    for i in range(1, len(gps_positions)):
        pos_diff = gps_positions[i] - gps_positions[i-1]
        heading = np.arctan2(pos_diff[0], pos_diff[1])  # atan2(East, North)
        gps_headings.append(np.degrees(heading))
    
    # 第一个点使用第二个点的航向
    if len(gps_headings) > 0:
        gps_headings = [gps_headings[0]] + gps_headings
    else:
        gps_headings = [0] * len(gps_positions)
    
    # 解决角度绕圈问题
    gps_headings_unwrapped = unwrap_angles(gps_headings)
    ekf_yaw_degrees = np.degrees(ekf_attitudes[:, 2])
    ekf_yaw_unwrapped = unwrap_angles(ekf_yaw_degrees)
    
    # 绘制展开后的航向角
    axes[1, 2].plot(fusion_times, gps_headings_unwrapped, 'r--', alpha=0.7, linewidth=1, 
                   label='GPS Heading (from motion)')
    axes[1, 2].plot(fusion_times, ekf_yaw_unwrapped, 'b-', linewidth=2, label='EKF Yaw')
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Heading (degrees, unwrapped)')
    axes[1, 2].set_title('Heading Comparison (Continuous)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # 添加角度统计信息
    if len(gps_headings_unwrapped) > 0 and len(ekf_yaw_unwrapped) > 0:
        angle_diff = ekf_yaw_unwrapped - gps_headings_unwrapped
        mean_angle_error = np.mean(np.abs(angle_diff))
        print(f"航向角误差统计:")
        print(f"  平均航向角误差: {mean_angle_error:.2f}°")
        print(f"  航向角误差标准差: {np.std(angle_diff):.2f}°")
        print(f"  最大航向角误差: {np.max(np.abs(angle_diff)):.2f}°")
    
    # 速度对比（第三行）
    if ekf_velocities is not None:
        # EKF速度大小时间序列
        ekf_speed = np.linalg.norm(ekf_velocities[:, :2], axis=1)  # 2D速度大小
        axes[2, 0].plot(fusion_times, ekf_speed, 'b-', linewidth=2, label='EKF Speed')
        
        if gps_velocities is not None:
            gps_speed = np.linalg.norm(gps_velocities[:, :2], axis=1)
            axes[2, 0].plot(fusion_times, gps_speed, 'r--', alpha=0.7, label='GPS Speed')
            axes[2, 0].set_title('Speed Comparison')
        else:
            axes[2, 0].set_title('EKF Speed Estimates')
            
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Speed (m/s)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 速度分量对比
        axes[2, 1].plot(fusion_times, ekf_velocities[:, 0], 'b-', linewidth=2, label='EKF Vx (East)')
        axes[2, 1].plot(fusion_times, ekf_velocities[:, 1], 'g-', linewidth=2, label='EKF Vy (North)')
        
        if gps_velocities is not None:
            axes[2, 1].plot(fusion_times, gps_velocities[:, 0], 'r--', alpha=0.7, label='GPS Vx')
            axes[2, 1].plot(fusion_times, gps_velocities[:, 1], 'orange', linestyle='--', alpha=0.7, label='GPS Vy')
            axes[2, 1].set_title('Velocity Components Comparison')
        else:
            axes[2, 1].set_title('EKF Velocity Components')
            
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Velocity (m/s)')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # 第三个图：如果有GPS速度则显示误差，否则显示其他信息
        if gps_velocities is not None:
            # 速度误差
            vel_error = np.linalg.norm(ekf_velocities[:, :2] - gps_velocities[:, :2], axis=1)
            axes[2, 2].plot(fusion_times, vel_error, 'purple', linewidth=1)
            axes[2, 2].set_xlabel('Time (s)')
            axes[2, 2].set_ylabel('Velocity Error (m/s)')
            axes[2, 2].set_title('EKF vs GPS Velocity Difference')
            axes[2, 2].grid(True, alpha=0.3)
            
            print(f"速度误差统计:")
            print(f"  平均速度误差: {np.mean(vel_error):.3f} m/s")
            print(f"  速度误差标准差: {np.std(vel_error):.3f} m/s")
            print(f"  最大速度误差: {np.max(vel_error):.3f} m/s")
        else:
            # 显示EKF速度变化率（加速度）
            ekf_accel = np.zeros_like(ekf_speed)
            if len(fusion_times) > 1:
                dt = np.diff(fusion_times)
                speed_diff = np.diff(ekf_speed)
                ekf_accel[1:] = speed_diff / dt
                
            axes[2, 2].plot(fusion_times, ekf_accel, 'purple', linewidth=1)
            axes[2, 2].set_xlabel('Time (s)')
            axes[2, 2].set_ylabel('Speed Change Rate (m/s²)')
            axes[2, 2].set_title('EKF Speed Acceleration')
            axes[2, 2].grid(True, alpha=0.3)
    else:
        # 如果没有EKF速度数据，显示提示信息
        for col in range(3):
            axes[2, col].text(0.5, 0.5, 'EKF velocity\ndata unavailable', 
                             transform=axes[2, col].transAxes, ha='center', va='center', fontsize=12)
            axes[2, col].set_title('Velocity Data Unavailable')
    
    # 第四行: 磁力计和气压计数据
    # 磁力计数据可视化
    if magnetometer_data is not None:
        mag_norm = np.linalg.norm(magnetometer_data, axis=1)
        axes[3, 0].plot(fusion_times, magnetometer_data[:, 0], 'r-', label='Mag X (μT)', alpha=0.8)
        axes[3, 0].plot(fusion_times, magnetometer_data[:, 1], 'g-', label='Mag Y (μT)', alpha=0.8)
        axes[3, 0].plot(fusion_times, magnetometer_data[:, 2], 'b-', label='Mag Z (μT)', alpha=0.8)
        axes[3, 0].set_xlabel('Time (s)')
        axes[3, 0].set_ylabel('Magnetic Field (μT)')
        axes[3, 0].set_title('Magnetometer Data')
        axes[3, 0].legend()
        axes[3, 0].grid(True, alpha=0.3)
        
        # 磁场强度
        axes[3, 1].plot(fusion_times, mag_norm, 'purple', linewidth=2)
        axes[3, 1].set_xlabel('Time (s)')
        axes[3, 1].set_ylabel('Magnetic Field Magnitude (μT)')
        axes[3, 1].set_title('Total Magnetic Field Strength')
        axes[3, 1].grid(True, alpha=0.3)
        
        print(f"磁力计数据统计:")
        print(f"  磁场强度范围: {np.min(mag_norm):.1f} - {np.max(mag_norm):.1f} μT")
        print(f"  平均磁场强度: {np.mean(mag_norm):.1f} μT")
    else:
        axes[3, 0].text(0.5, 0.5, 'Magnetometer\ndata unavailable', 
                       transform=axes[3, 0].transAxes, ha='center', va='center', fontsize=12)
        axes[3, 0].set_title('Magnetometer Data Unavailable')
        
        axes[3, 1].text(0.5, 0.5, 'Magnetometer\ndata unavailable', 
                       transform=axes[3, 1].transAxes, ha='center', va='center', fontsize=12)
        axes[3, 1].set_title('Magnetometer Data Unavailable')
    
    # 气压计和EKF高度数据 (双y轴，缩放到同一尺度)
    if barometer_data is not None:
        pressure_data, _ = barometer_data  # 忽略计算的高度，使用EKF的Z轴
        ekf_altitude = ekf_positions[:, 2]  # EKF估计的Z轴位置
        
        # 对气压数据进行滑动滤波
        def moving_average_filter(data, window_size=200):
            """滑动平均滤波器"""
            if len(data) < window_size:
                return data
            filtered = np.convolve(data, np.ones(window_size)/window_size, mode='same')
            # 处理边界效应
            filtered[:window_size//2] = data[:window_size//2]
            filtered[-window_size//2:] = data[-window_size//2:]
            return filtered
        
        pressure_filtered = moving_average_filter(pressure_data, window_size=200)
        
        # 缩放到同一尺度 (使用滤波后的数据计算缩放参数)
        pressure_range = np.max(pressure_filtered) - np.min(pressure_filtered)
        altitude_range = np.max(ekf_altitude) - np.min(ekf_altitude)
        
        # 选择一个合适的缩放因子，使两个数据在视觉上有相似的变化幅度
        if pressure_range > 0 and altitude_range > 0:
            scale_factor = altitude_range / pressure_range
            
            # 正确的缩放方法：保持中心点，只缩放变化幅度
            pressure_center = np.mean(pressure_filtered)
            altitude_center = np.mean(ekf_altitude)
            
            # 缩放原始数据和滤波数据
            pressure_raw_scaled = (pressure_data - pressure_center) * scale_factor + altitude_center
            pressure_filtered_scaled = (pressure_filtered - pressure_center) * scale_factor + altitude_center
        else:
            pressure_raw_scaled = pressure_data
            pressure_filtered_scaled = pressure_filtered
            scale_factor = 1.0
        
        # 创建双y轴
        ax_pressure = axes[3, 2]
        ax_altitude = ax_pressure.twinx()
        
        # 绘制缩放后的数据：原始（散点）+ 滤波（线条）
        line1 = ax_pressure.scatter(fusion_times, pressure_raw_scaled, c='lightblue', s=10, alpha=0.4, label='Pressure (raw)')
        line2 = ax_pressure.plot(fusion_times, pressure_filtered_scaled, 'b-', linewidth=2, label='Pressure (filtered)')
        line3 = ax_altitude.plot(fusion_times, ekf_altitude, 'r-', linewidth=2, label='EKF Z-axis')
        
        # 添加GPS高度观测
        gps_altitude = gps_positions[:, 2]  # GPS Z坐标
        line4 = ax_altitude.plot(fusion_times, gps_altitude, 'g--', linewidth=2, alpha=0.8, label='GPS Altitude')
        
        # 设置左轴 (气压，缩放后)
        ax_pressure.set_xlabel('Time (s)')
        ax_pressure.set_ylabel('Pressure (scaled to Z-axis range)', color='b')
        ax_pressure.tick_params(axis='y', labelcolor='b')
        
        # 设置右轴 (EKF和GPS高度)
        ax_altitude.set_ylabel('Altitude (m)', color='k')
        ax_altitude.tick_params(axis='y', labelcolor='k')
        
        # 设置标题和图例
        ax_pressure.set_title('Barometer Pressure vs EKF Z-axis vs GPS Altitude')
        ax_pressure.grid(True, alpha=0.3)
        
        # 所有图例都放右边
        lines1, labels1 = ax_pressure.get_legend_handles_labels()
        lines2, labels2 = ax_altitude.get_legend_handles_labels()
        
        # 合并所有图例并放在右上角
        ax_altitude.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        
        # 添加实际数值范围到图例
        pressure_info = f"Pressure: {np.min(pressure_data):.3f}-{np.max(pressure_data):.3f} kPa"
        filter_info = f"Filter: 200-point moving average"
        scale_info = f"Scale factor: {scale_factor:.1f}"
        altitude_info = f"EKF Z-axis: {np.min(ekf_altitude):.1f}-{np.max(ekf_altitude):.1f} m"
        gps_info = f"GPS Altitude: {np.min(gps_altitude):.1f}-{np.max(gps_altitude):.1f} m"
        ax_pressure.text(0.02, 0.78, f"{pressure_info}\n{filter_info}\n{scale_info}\n{altitude_info}\n{gps_info}", 
                        transform=ax_pressure.transAxes, fontsize=8, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        print(f"气压计与EKF/GPS高度对比统计:")
        print(f"  原始气压范围: {np.min(pressure_data):.3f} - {np.max(pressure_data):.3f} kPa")
        print(f"  滤波气压范围: {np.min(pressure_filtered):.3f} - {np.max(pressure_filtered):.3f} kPa") 
        print(f"  EKF Z轴范围: {np.min(ekf_altitude):.1f} - {np.max(ekf_altitude):.1f} m")
        print(f"  GPS高度范围: {np.min(gps_altitude):.1f} - {np.max(gps_altitude):.1f} m")
        print(f"  缩放因子: {scale_factor:.1f}")
        print(f"  滤波效果: 噪声降低 {(np.std(pressure_data) - np.std(pressure_filtered))/np.std(pressure_data)*100:.1f}%")
        print(f"  平均气压: {np.mean(pressure_filtered):.3f} kPa (滤波后)")
        print(f"  平均EKF高度: {np.mean(ekf_altitude):.1f} m")
        print(f"  平均GPS高度: {np.mean(gps_altitude):.1f} m")
        print(f"  EKF-GPS高度差: 平均={np.mean(ekf_altitude - gps_altitude):.2f} m, 标准差={np.std(ekf_altitude - gps_altitude):.2f} m")
    else:
        axes[3, 2].text(0.5, 0.5, 'Barometer\ndata unavailable', 
                       transform=axes[3, 2].transAxes, ha='center', va='center', fontsize=12)
        axes[3, 2].set_title('Barometer Data Unavailable')
    
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_ekf_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 统计信息
    mean_error = np.mean(position_error)
    std_error = np.std(position_error)
    max_error = np.max(position_error)
    
    print(f"\nEKF融合结果统计:")
    print(f"  平均位置误差: {mean_error:.2f} m")
    print(f"  位置误差标准差: {std_error:.2f} m") 
    print(f"  最大位置误差: {max_error:.2f} m")
    print(f"  轨迹总长度 (GPS): {calculate_trajectory_length(gps_positions):.1f} m")
    print(f"  轨迹总长度 (EKF): {calculate_trajectory_length(ekf_positions):.1f} m")