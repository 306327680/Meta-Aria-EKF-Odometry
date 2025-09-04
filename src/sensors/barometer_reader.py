#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Barometer Data Reader for Project Aria VRS Files

Handles reading atmospheric pressure data for altitude corrections.
"""

import numpy as np


def read_barometer_data(provider, downsample=1):
    """读取气压计数据"""
    try:
        # 使用正确的Project Aria气压计标签
        stream_id = provider.get_stream_id_from_label("baro0")
        print(f"找到气压计流: baro0")
    except Exception as e:
        print(f"无法找到气压计流 (baro0): {e}")
        return None, None, None

    num_samples = provider.get_num_data(stream_id)
    if num_samples == 0:
        print("气压计流中没有数据")
        return None, None, None

    timestamps = []
    pressures = []
    temperatures = []

    for idx in range(0, num_samples, downsample):
        try:
            baro_data = provider.get_barometer_data_by_index(stream_id, idx)
            
            # 调试：检查第一个气压计数据的结构
            if idx == 0:
                print(f"气压计数据类型: {type(baro_data)}")
                print(f"气压计数据属性: {[attr for attr in dir(baro_data) if not attr.startswith('_')]}")
            
            # 提取气压和温度数据 (按照Project Aria格式)
            pressure = baro_data.pressure * 1e-3  # 转换为 kPa
            temperature = baro_data.temperature   # 温度 (摄氏度)
            timestamp = baro_data.capture_timestamp_ns * 1e-9
            
            pressures.append(pressure)
            temperatures.append(temperature)
            timestamps.append(timestamp)
            
        except Exception as e:
            print(f"读取气压计数据错误 (索引 {idx}): {e}")
            continue

    if len(timestamps) == 0:
        return None, None, None
        
    return np.array(timestamps), np.array(pressures), np.array(temperatures)