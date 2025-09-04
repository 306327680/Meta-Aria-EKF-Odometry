#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Magnetometer Data Reader for Project Aria VRS Files

Handles reading magnetic field data for heading corrections.
"""

import numpy as np


def read_magnetometer_data(provider, downsample=1):
    """读取磁力计数据"""
    try:
        # 使用正确的Project Aria磁力计标签
        stream_id = provider.get_stream_id_from_label("mag0")
        print(f"找到磁力计流: mag0")
    except Exception as e:
        print(f"无法找到磁力计流 (mag0): {e}")
        return None, None

    num_samples = provider.get_num_data(stream_id)
    if num_samples == 0:
        print("磁力计流中没有数据")
        return None, None

    timestamps = []
    magnetic_fields = []

    for idx in range(0, num_samples, downsample):
        try:
            mag_data = provider.get_magnetometer_data_by_index(stream_id, idx)
            
            # 调试：检查第一个磁力计数据的结构
            if idx == 0:
                print(f"磁力计数据类型: {type(mag_data)}")
                print(f"磁力计数据属性: {[attr for attr in dir(mag_data) if not attr.startswith('_')]}")
            
            # 提取磁力计数据 (转换为微特斯拉 μT)
            mag_field = np.array([
                mag_data.mag_tesla[0] * 1e6,  # 转换为 μT
                mag_data.mag_tesla[1] * 1e6,
                mag_data.mag_tesla[2] * 1e6
            ])
            timestamp = mag_data.capture_timestamp_ns * 1e-9
            
            magnetic_fields.append(mag_field)
            timestamps.append(timestamp)
            
        except Exception as e:
            print(f"读取磁力计数据错误 (索引 {idx}): {e}")
            continue

    if len(timestamps) == 0:
        return None, None
        
    return np.array(timestamps), np.array(magnetic_fields)