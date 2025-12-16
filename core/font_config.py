# core/font_config.py - 独立的字体配置模块
import matplotlib
import platform
import os
import sys

def setup_chinese_font():
    """设置中文字体，解决中文乱码问题"""
    
    # 先重置为默认配置
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    
    # 获取系统信息
    system = platform.system()
    
    # 根据系统设置字体路径
    font_paths = []
    if system == 'Windows':
        font_paths = [
            r"C:\Windows\Fonts\msyh.ttc",  # 微软雅黑
            r"C:\Windows\Fonts\msjh.ttc",  # 微软正黑
            r"C:\Windows\Fonts\simhei.ttf",  # 黑体
            r"C:\Windows\Fonts\simsun.ttc",  # 宋体
        ]
    elif system == 'Darwin':  # macOS
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",  # 苹方
            "/System/Library/Fonts/Hiragino Sans GB.ttc",  # 冬青黑体
            "/Library/Fonts/Arial Unicode.ttf",  # Arial Unicode
        ]
    else:  # Linux
        font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # 文泉驿微米黑
            "/usr/share/fonts/truetype/arphic/uming.ttc",  # 文鼎明体
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # DejaVu Sans
        ]
    
    # 尝试加载字体
    font_added = False
    font_name = None
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                # 添加字体到matplotlib
                font_prop = matplotlib.font_manager.FontProperties(fname=font_path)
                font_name = font_prop.get_name()
                matplotlib.font_manager.fontManager.addfont(font_path)
                print(f"成功加载字体: {font_name} from {font_path}")
                font_added = True
                break
            except Exception as e:
                print(f"加载字体 {font_path} 失败: {e}")
                continue
    
    if font_added and font_name:
        # 设置字体
        matplotlib.rcParams['font.sans-serif'] = [font_name]
        matplotlib.rcParams['axes.unicode_minus'] = False
        matplotlib.rcParams['font.size'] = 12
        print(f"已设置中文字体: {font_name}")
    else:
        # 如果找不到中文字体，使用英文字体
        print("警告: 未找到中文字体，使用英文字体")
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        matplotlib.rcParams['axes.unicode_minus'] = False
        font_name = 'DejaVu Sans'
    
    return font_name