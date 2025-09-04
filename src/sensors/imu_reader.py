#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMU Data Reader for Project Aria VRS Files

Handles reading accelerometer and gyroscope data from IMU sensors.
"""

import numpy as np


def read_imu_data(provider, downsample=1):
    """读取IMU数据"""
    try:
        # 尝试不同的IMU流标签
        stream_labels = ["imu-right", "imu-left", "imu", "accelerometer", "gyroscope"]
        stream_id = None
        
        for label in stream_labels:
            try:
                stream_id = provider.get_stream_id_from_label(label)
                print(f"找到IMU流: {label}")
                break
            except:
                continue
                
        if stream_id is None:
            raise RuntimeError("未找到IMU数据流")
            
    except Exception as e:
        raise RuntimeError(f"无法找到IMU流: {e}")

    num_samples = provider.get_num_data(stream_id)
    if num_samples == 0:
        raise RuntimeError("IMU流中没有数据")

    timestamps = []
    accelerations = []
    angular_velocities = []

    for idx in range(0, num_samples, downsample):
        try:
            imu_data = provider.get_imu_data_by_index(stream_id, idx)
            
            # 调试：检查第一个IMU数据的结构
            if idx == 0:
                print(f"IMU数据类型: {type(imu_data)}")
                print(f"IMU数据属性: {[attr for attr in dir(imu_data) if not attr.startswith('_')]}")
            
            # 提取数据
            accel = np.array([imu_data.accel_msec2[0], imu_data.accel_msec2[1], imu_data.accel_msec2[2]])
            gyro = np.array([imu_data.gyro_radsec[0], imu_data.gyro_radsec[1], imu_data.gyro_radsec[2]])
            timestamp = imu_data.capture_timestamp_ns * 1e-9
            
            accelerations.append(accel)
            angular_velocities.append(gyro)
            timestamps.append(timestamp)
            
        except Exception as e:
            print(f"读取IMU数据错误 (索引 {idx}): {e}")
            continue

    return (np.array(timestamps), 
            np.array(accelerations), 
            np.array(angular_velocities))