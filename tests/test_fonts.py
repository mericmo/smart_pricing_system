import matplotlib
import matplotlib.pyplot as plt
import os
import shutil

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
def fix_matplotlib_fonts():
    """修复matplotlib字体问题"""
    print("=== 开始修复matplotlib中文乱码问题 ===\n")

    # 1. 显示当前设置
    print("1. 当前matplotlib配置:")
    print(f"   版本: {matplotlib.__version__}")
    print(f"   字体设置: {matplotlib.rcParams.get('font.sans-serif', '未设置')}")
    print(f"   负号设置: {matplotlib.rcParams.get('axes.unicode_minus', '未设置')}")

    # 2. 清除缓存
    print("\n2. 清除matplotlib缓存...")
    cache_dir = matplotlib.get_cachedir()
    print(f"   缓存目录: {cache_dir}")

    try:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print("   ✓ 已清除缓存")
        else:
            print("   ℹ 缓存目录不存在")
    except Exception as e:
        print(f"   ✗ 清除缓存失败: {e}")

    # 3. 重建字体缓存
    print("\n3. 重建字体缓存...")
    try:
        matplotlib.font_manager._rebuild()
        print("   ✓ 重建成功")
    except Exception as e:
        print(f"   ✗ 重建失败: {e}")

    # 4. 设置Windows字体
    print("\n4. 设置中文字体...")
    # Windows中文系统字体
    windows_fonts = [
        'Microsoft YaHei',  # 微软雅黑
        'SimHei',  # 黑体
        'SimSun',  # 宋体
        'KaiTi',  # 楷体
        'FangSong',  # 仿宋
    ]

    # 设置字体
    matplotlib.rcParams['font.sans-serif'] = windows_fonts
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.size'] = 12

    print(f"   ✓ 设置字体: {windows_fonts[0]}")
    print(f"   ✓ 设置负号显示: {matplotlib.rcParams['axes.unicode_minus']}")

    # 5. 创建测试图表
    print("\n5. 创建测试图表...")
    try:
        plt.figure(figsize=(8, 4))
        plt.text(0.1, 0.8, "中文测试: 智能定价系统", fontsize=14, fontweight='bold')
        plt.text(0.1, 0.6, "商品编码: 8006144", fontsize=12)
        plt.text(0.1, 0.5, "模型评估: R² = -3.508", fontsize=12)
        plt.text(0.1, 0.4, "预测销量 vs 实际销量", fontsize=12)
        plt.text(0.1, 0.3, "门店: 205625", fontsize=12)
        plt.text(0.1, 0.2, "注意: R²为负表示模型效果差", fontsize=11, color='red')
        plt.title("字体修复测试 - 中文和负号显示", fontsize=16, fontweight='bold')
        plt.axis('off')

        test_file = "font_repair_test.png"
        plt.savefig(test_file, dpi=150, bbox_inches='tight')
        plt.close()

        if os.path.exists(test_file):
            print(f"   ✓ 测试图表已保存: {test_file}")
            print(f"   文件大小: {os.path.getsize(test_file)} 字节")
        else:
            print("   ✗ 保存测试图表失败")
    except Exception as e:
        print(f"   ✗ 创建测试图表失败: {e}")

    # 6. 验证设置
    print("\n6. 验证最终设置:")
    print(f"   当前字体: {matplotlib.rcParams.get('font.sans-serif')}")
    print(f"   负号显示: {matplotlib.rcParams.get('axes.unicode_minus')}")

    print("\n=== 修复完成 ===")
    print("\n接下来:")
    print("1. 打开 'font_repair_test.png' 文件")
    print("2. 检查中文是否正常显示")
    print("3. 如果中文正常，重新运行你的定价策略生成器")

    return test_file if os.path.exists(test_file) else None


if __name__ == "__main__":
    fix_matplotlib_fonts()
    input("\n按 Enter 键退出...")