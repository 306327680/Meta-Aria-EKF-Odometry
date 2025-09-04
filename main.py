#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Aria GPS/IMU EKF Fusion - Main Application

Multi-sensor fusion odometry system for Project Aria VRS files.

Usage:
    python main.py path/to/file.vrs [options]
    
Features:
- GPS/IMU sensor fusion with Extended Kalman Filter
- Optional magnetometer and barometer integration  
- Comprehensive visualization and analysis
- Robust error handling and data validation
"""

import argparse
import sys
import numpy as np
import warnings

# Suppress common warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Project Aria tools
from projectaria_tools.core import data_provider

# Local modules
from src.sensors import (
    read_gps_data, read_imu_data, 
    read_magnetometer_data, read_barometer_data
)
from src.fusion import run_ekf_fusion
from src.utils.coordinates import gps_to_local_coords
from src.visualization import plot_comparison


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Project Aria GPS/IMU EKF Fusion for Odometry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py data.vrs
  python main.py data.vrs --downsample 2 --out-prefix result
  
Sensor Support:
  - GPS: Position and velocity (required)
  - IMU: Accelerometer and gyroscope (required)  
  - Magnetometer: Heading corrections (optional)
  - Barometer: Altitude corrections (optional)
        """
    )
    
    parser.add_argument("vrs_path", type=str, help="Path to the Project Aria VRS file")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Downsample factor for data (default: 1)")
    parser.add_argument("--out-prefix", type=str, default="ekf_fusion", 
                        help="Output file prefix (default: ekf_fusion)")
    
    args = parser.parse_args()
    
    print(f"正在处理 VRS 文件: {args.vrs_path}")
    
    # 打开VRS文件
    try:
        provider = data_provider.create_vrs_data_provider(args.vrs_path)
    except Exception as e:
        print(f"无法打开VRS文件: {e}")
        sys.exit(1)
    
    # 读取GPS数据
    try:
        print("读取GPS数据...")
        gps_times, gps_positions, gps_accuracies, gps_vertical_accuracies, gps_velocities = read_gps_data(provider, args.downsample)
        print(f"GPS数据点数: {len(gps_times)}")
        print(f"GPS速度数据范围: {np.min(np.linalg.norm(gps_velocities, axis=1)):.2f} - {np.max(np.linalg.norm(gps_velocities, axis=1)):.2f} m/s")
    except Exception as e:
        print(f"读取GPS数据失败: {e}")
        sys.exit(1)
    
    # 读取IMU数据  
    try:
        print("读取IMU数据...")
        imu_times, imu_accels, imu_gyros = read_imu_data(provider, args.downsample)
        print(f"IMU数据点数: {len(imu_times)}")
    except Exception as e:
        print(f"读取IMU数据失败: {e}")
        sys.exit(1)
    
    # 读取磁力计数据（可选）
    mag_times, mag_fields = None, None
    try:
        print("尝试读取磁力计数据...")
        mag_times, mag_fields = read_magnetometer_data(provider, args.downsample)
        if mag_times is not None:
            print(f"磁力计数据点数: {len(mag_times)}")
        else:
            print("磁力计数据不可用")
    except Exception as e:
        print(f"读取磁力计数据失败: {e}")
    
    # 读取气压计数据（可选）
    baro_times, baro_pressures, baro_temps = None, None, None
    try:
        print("尝试读取气压计数据...")
        baro_times, baro_pressures, baro_temps = read_barometer_data(provider, args.downsample)
        if baro_times is not None:
            print(f"气压计数据点数: {len(baro_times)}")
        else:
            print("气压计数据不可用")
    except Exception as e:
        print(f"读取气压计数据失败: {e}")
    
    # 转换GPS到局部坐标系
    print("转换GPS坐标...")
    
    # 检查起点GPS高度稳定性，跳过异常点
    stable_start_idx = 0
    for i in range(1, min(15, len(gps_positions))):
        height_change = abs(gps_positions[i][2] - gps_positions[i-1][2])
        if height_change > 30:  # 单次变化超过30米认为异常
            print(f"警告: GPS点{i-1}→{i}高度变化{height_change:.1f}m异常，继续寻找稳定起点")
            continue
        else:
            # 找到相对稳定的起点
            stable_start_idx = max(0, i - 2)  # 稍微往前取一点作为安全边际
            break
    
    if stable_start_idx > 0:
        print(f"使用GPS点{stable_start_idx}作为稳定起点 (跳过前{stable_start_idx}个不稳定点)")
        gps_positions = gps_positions[stable_start_idx:]
        gps_times = gps_times[stable_start_idx:]
        gps_accuracies = gps_accuracies[stable_start_idx:]
        gps_vertical_accuracies = gps_vertical_accuracies[stable_start_idx:]
        gps_velocities = gps_velocities[stable_start_idx:]
        
        print(f"调整后GPS数据点数: {len(gps_times)}")
    
    origin_gps = gps_positions[0]  # 使用稳定的GPS点作为原点
    gps_local = gps_to_local_coords(gps_positions, origin_gps)
    
    print(f"坐标原点: 纬度={origin_gps[0]:.6f}°, 经度={origin_gps[1]:.6f}°, 海拔={origin_gps[2]:.1f}m")
    
    # 转换GPS速度到局部坐标系
    print("转换GPS速度...")
    # GPS速度通常已经在ENU坐标系，只需要检查和转换
    gps_velocities_local = gps_velocities.copy()  # 速度不需要坐标转换
    
    # 运行EKF融合
    print("运行EKF融合...")
    try:
        (fusion_times, ekf_positions, ekf_velocities, 
         ekf_attitudes, interp_gps, interp_gps_vel,
         interp_mag_fields, interp_baro_pressures, interp_baro_temps,
         interp_gps_acc, interp_gps_vertical_acc) = run_ekf_fusion(
            gps_times, gps_local, gps_accuracies, gps_velocities_local,
            imu_times, imu_accels, imu_gyros,
            mag_times, mag_fields, 
            baro_times, baro_pressures, baro_temps,
            gps_vertical_accuracies
        )
        print(f"EKF融合完成，生成 {len(fusion_times)} 个数据点")
    except Exception as e:
        print(f"EKF融合失败: {e}")
        sys.exit(1)
    
    # 准备传感器数据用于可视化
    magnetometer_data = interp_mag_fields
    barometer_data = None
    
    if interp_baro_pressures is not None:
        # 使用气压高度公式计算海拔高度
        # h = 44330 * (1 - (P/P0)^(1/5.255))
        # 其中 P0 = 101.325 kPa (海平面标准大气压)
        sea_level_pressure = 101.325  # kPa
        altitudes = 44330 * (1 - (interp_baro_pressures / sea_level_pressure) ** (1/5.255))
        barometer_data = (interp_baro_pressures, altitudes)
        
        print(f"气压计数据处理:")
        print(f"  气压范围: {np.min(interp_baro_pressures):.1f} - {np.max(interp_baro_pressures):.1f} kPa")
        print(f"  计算高度范围: {np.min(altitudes):.1f} - {np.max(altitudes):.1f} m")
    
    # 绘制对比图
    print("生成对比图...")
    plot_comparison(fusion_times, ekf_positions, interp_gps, ekf_attitudes, args.out_prefix, 
                   ekf_velocities=ekf_velocities, gps_velocities=interp_gps_vel,
                   magnetometer_data=magnetometer_data, barometer_data=barometer_data)
    
    print(f"\n完成！生成文件:")
    print(f"  {args.out_prefix}_ekf_comparison.png")
    print("\n提示: EKF融合结合了GPS的绝对位置和IMU的高频运动信息，")
    print("      可以提供更平滑、更高频率的轨迹估计。")


if __name__ == "__main__":
    main()