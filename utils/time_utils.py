# utils/time_utils.py
from datetime import datetime, time, timedelta
from typing import Tuple, List, Optional, Dict
import pandas as pd
import numpy as np

def parse_time_string(time_str: str) -> time:
    """解析时间字符串"""
    try:
        return datetime.strptime(time_str, "%H:%M").time()
    except ValueError:
        try:
            return datetime.strptime(time_str, "%H:%M:%S").time()
        except:
            return time(0, 0)

def format_time_string(time_obj: time) -> str:
    """格式化时间对象为字符串"""
    return time_obj.strftime("%H:%M")

def calculate_time_duration(start_time: str, end_time: str) -> float:
    """计算时间段长度（小时）"""
    start = parse_time_string(start_time)
    end = parse_time_string(end_time)
    
    start_dt = datetime.combine(datetime.today(), start)
    end_dt = datetime.combine(datetime.today(), end)
    
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)
    
    duration = (end_dt - start_dt).total_seconds() / 3600.0
    return duration

def split_time_range(start_time: str, end_time: str, 
                    num_segments: int = 4) -> List[Tuple[str, str]]:
    """分割时间范围"""
    duration = calculate_time_duration(start_time, end_time)
    segment_duration = duration / num_segments
    
    segments = []
    current_time = parse_time_string(start_time)
    
    for i in range(num_segments):
        segment_start = current_time
        
        # 计算结束时间
        segment_end_dt = datetime.combine(datetime.today(), current_time) + \
                        timedelta(hours=segment_duration)
        segment_end = segment_end_dt.time()
        
        segments.append((
            format_time_string(segment_start),
            format_time_string(segment_end)
        ))
        
        current_time = segment_end
    
    return segments

def get_current_time_segment(current_time: datetime, 
                           time_segments: List[Tuple[str, str]]) -> int:
    """获取当前时间所在的时间段索引"""
    current_time_str = current_time.strftime("%H:%M")
    
    for i, (start, end) in enumerate(time_segments):
        if start <= current_time_str < end:
            return i
    
    # 如果不在任何时间段内，返回最后一个
    return len(time_segments) - 1

def calculate_time_to_close(current_time: datetime, 
                          close_time: str) -> float:
    """计算距离关店时间还有多少小时"""
    close = parse_time_string(close_time)
    
    current_dt = datetime.combine(current_time.date(), current_time.time())
    close_dt = datetime.combine(current_time.date(), close)
    
    if close_dt < current_dt:
        close_dt += timedelta(days=1)
    
    time_to_close = (close_dt - current_dt).total_seconds() / 3600.0
    return max(0, time_to_close)

def is_within_time_range(check_time: datetime, 
                        start_time: str, 
                        end_time: str) -> bool:
    """检查时间是否在指定范围内"""
    check_time_str = check_time.strftime("%H:%M")
    start = parse_time_string(start_time)
    end = parse_time_string(end_time)
    
    start_str = format_time_string(start)
    end_str = format_time_string(end)
    
    if start_str <= end_str:
        return start_str <= check_time_str < end_str
    else:
        # 跨天范围
        return check_time_str >= start_str or check_time_str < end_str

def calculate_time_decay_factor(current_time: datetime,
                               start_time: str,
                               end_time: str,
                               decay_rate: float = 2.0) -> float:
    """计算时间衰减因子"""
    # 计算剩余时间比例
    total_duration = calculate_time_duration(start_time, end_time)
    time_to_close = calculate_time_to_close(current_time, end_time)
    
    if total_duration == 0:
        return 1.0
    
    remaining_ratio = time_to_close / total_duration
    
    # 指数衰减
    decay_factor = 1.0 + (1.0 - remaining_ratio) ** decay_rate
    
    return decay_factor

def get_time_features(timestamp: datetime) -> Dict[str, float]:
    """获取时间特征"""
    return {
        'hour': timestamp.hour,
        'minute': timestamp.minute,
        'day_of_week': timestamp.weekday(),
        'day_of_month': timestamp.day,
        'month': timestamp.month,
        'quarter': (timestamp.month - 1) // 3 + 1,
        'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
        'hour_sin': np.sin(2 * np.pi * timestamp.hour / 24),
        'hour_cos': np.cos(2 * np.pi * timestamp.hour / 24),
        'day_sin': np.sin(2 * np.pi * timestamp.weekday() / 7),
        'day_cos': np.cos(2 * np.pi * timestamp.weekday() / 7),
        'month_sin': np.sin(2 * np.pi * timestamp.month / 12),
        'month_cos': np.cos(2 * np.pi * timestamp.month / 12)
    }

def adjust_for_timezone(timestamp: datetime, 
                       timezone_offset: int = 8) -> datetime:
    """调整时区（默认东八区）"""
    return timestamp + timedelta(hours=timezone_offset)

def get_business_hours_pattern(store_type: str = "supermarket") -> Dict[str, Tuple[str, str]]:
    """获取营业时间模式"""
    patterns = {
        "supermarket": {
            "weekday": ("08:00", "22:00"),
            "weekend": ("08:00", "22:00"),
            "holiday": ("09:00", "21:00")
        },
        "convenience": {
            "weekday": ("07:00", "23:00"),
            "weekend": ("07:00", "23:00"),
            "holiday": ("08:00", "22:00")
        },
        "mall": {
            "weekday": ("10:00", "22:00"),
            "weekend": ("10:00", "22:00"),
            "holiday": ("10:00", "22:00")
        }
    }
    
    return patterns.get(store_type, patterns["supermarket"])

def calculate_peak_hours(store_type: str = "supermarket") -> List[Tuple[str, str]]:
    """计算高峰时段"""
    patterns = {
        "supermarket": [
            ("08:00", "10:00"),   # 早上高峰
            ("12:00", "14:00"),   # 午间高峰
            ("17:00", "19:00"),   # 晚间高峰
            ("20:00", "22:00")    # 清仓时段
        ],
        "convenience": [
            ("07:00", "09:00"),   # 早上通勤
            ("11:00", "13:00"),   # 午餐时间
            ("17:00", "19:00"),   # 下班时间
            ("21:00", "23:00")    # 夜间
        ]
    }
    
    return patterns.get(store_type, patterns["supermarket"])