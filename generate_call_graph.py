#!/usr/bin/env python3
"""
生成调用关系图的Python脚本
使用 graphviz 或 matplotlib 生成可视化图表
"""

import os
import sys
from typing import Dict, List, Set, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib 未安装，将使用文本格式输出")

try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    print("提示: 安装 graphviz 可以获得更好的可视化效果: pip install graphviz")


class CallGraphGenerator:
    """调用关系图生成器"""
    
    def __init__(self):
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[Tuple[str, str, str]] = []  # (from, to, label)
        self.modules: Set[str] = set()
        
    def add_node(self, node_id: str, label: str, module: str = "", node_type: str = "function"):
        """添加节点"""
        self.nodes[node_id] = {
            'label': label,
            'module': module,
            'type': node_type
        }
        if module:
            self.modules.add(module)
    
    def add_edge(self, from_node: str, to_node: str, label: str = ""):
        """添加边"""
        self.edges.append((from_node, to_node, label))
    
    def build_from_main(self):
        """从main.py构建调用关系"""
        
        # 入口节点
        self.add_node("main", "main()", "main.py", "entry")
        
        # 数据加载
        self.add_node("load_data", "load_data_files()", "main.py", "function")
        self.add_node("create_sample", "create_sample_data()", "main.py", "function")
        self.add_edge("main", "load_data", "文件存在")
        self.add_edge("main", "create_sample", "文件不存在")
        
        # 系统初始化
        self.add_node("config_mgr", "ConfigManager()", "core.config", "class")
        self.add_node("generator_init", "EnhancedPricingStrategyGenerator.__init__()", 
                     "core.pricing_strategy_generator", "class")
        self.add_edge("main", "config_mgr")
        self.add_edge("main", "generator_init")
        
        # 初始化时创建的子组件
        self.add_node("data_proc", "TransactionDataProcessor", "data.data_processor", "class")
        self.add_node("feature_eng", "PricingFeatureEngineer", "data.feature_engineer", "class")
        self.add_node("demand_pred", "EnhancedDemandPredictor", "models.demand_predictor", "class")
        self.add_node("visualizer", "SimplifiedModelVisualizer", "core.model_evaluator", "class")
        self.add_edge("generator_init", "data_proc", "创建")
        self.add_edge("generator_init", "feature_eng", "创建")
        self.add_edge("generator_init", "demand_pred", "创建")
        self.add_edge("generator_init", "visualizer", "创建")
        
        # 生成策略主流程
        self.add_node("generate_strategy", "generate_pricing_strategy()", 
                     "core.pricing_strategy_generator", "method")
        self.add_edge("main", "generate_strategy")
        
        # 获取商品信息
        self.add_node("get_product_info", "_get_product_info()", 
                     "core.pricing_strategy_generator", "method")
        self.add_node("filter_product", "filter_by_product()", "data.data_processor", "method")
        self.add_node("get_summary", "get_product_summary()", "data.data_processor", "method")
        self.add_edge("generate_strategy", "get_product_info")
        self.add_edge("get_product_info", "filter_product")
        self.add_edge("get_product_info", "get_summary")
        
        # 准备特征
        self.add_node("prepare_features", "_prepare_features()", 
                     "core.pricing_strategy_generator", "method")
        self.add_node("create_features", "create_features()", "data.feature_engineer", "method")
        self.add_edge("generate_strategy", "prepare_features")
        self.add_edge("prepare_features", "create_features")
        self.add_edge("prepare_features", "get_summary")
        
        # 训练模型
        self.add_node("train_predictor", "_train_demand_predictor()", 
                     "core.pricing_strategy_generator", "method")
        self.add_node("prepare_training", "prepare_training_data_from_transactions()", 
                     "models.demand_predictor", "method")
        self.add_node("train_model", "train()", "models.demand_predictor", "method")
        self.add_node("predict_train", "predict_train_set()", "models.demand_predictor", "method")
        self.add_node("calc_metrics", "_calculate_performance_metrics()", 
                     "core.pricing_strategy_generator", "method")
        self.add_node("create_report", "create_comprehensive_report()", 
                     "core.model_evaluator", "method")
        self.add_edge("generate_strategy", "train_predictor")
        self.add_edge("train_predictor", "prepare_training")
        self.add_edge("train_predictor", "train_model")
        self.add_edge("train_predictor", "predict_train")
        self.add_edge("train_predictor", "calc_metrics")
        self.add_edge("train_predictor", "create_report")
        self.add_edge("prepare_training", "filter_product")
        self.add_edge("prepare_training", "get_summary")
        
        # 定价优化
        self.add_node("optimizer", "PricingOptimizer", "models.pricing_optimizer", "class")
        self.add_node("optimize_pricing", "optimize_staged_pricing()", 
                     "models.pricing_optimizer", "method")
        self.add_node("dp_optimization", "_dynamic_programming_optimization()", 
                     "models.pricing_optimizer", "method")
        self.add_node("predict_demand", "predict_demand()", "models.demand_predictor", "method")
        self.add_edge("generate_strategy", "optimizer", "创建")
        self.add_edge("generate_strategy", "optimize_pricing")
        self.add_edge("optimize_pricing", "dp_optimization")
        self.add_edge("dp_optimization", "predict_demand")
        
        # 评估方案
        self.add_node("evaluate_schedule", "evaluate_pricing_schedule()", 
                     "models.pricing_optimizer", "method")
        self.add_edge("generate_strategy", "evaluate_schedule")
        
        # 计算置信度
        self.add_node("calc_confidence", "_calculate_confidence_score()", 
                     "core.pricing_strategy_generator", "method")
        self.add_edge("generate_strategy", "calc_confidence")
        
        # 生成可视化
        self.add_node("generate_viz", "_generate_strategy_visualizations_with_pil()", 
                     "core.model_evaluator", "method")
        self.add_edge("generate_strategy", "generate_viz")
        
        # 创建策略对象
        self.add_node("strategy_obj", "EnhancedPricingStrategy", 
                     "core.pricing_strategy_generator", "dataclass")
        self.add_edge("generate_strategy", "strategy_obj", "创建")
        
        # 保存策略
        self.add_node("save_strategy", "save_strategy()", 
                     "core.pricing_strategy_generator", "method")
        self.add_edge("main", "save_strategy")
        self.add_edge("save_strategy", "strategy_obj")
    
    def export_graphviz(self, filename: str = "call_graph"):
        """导出为 Graphviz DOT 格式"""
        if not HAS_GRAPHVIZ:
            print("Graphviz 未安装，跳过 DOT 格式导出")
            return
        
        dot = Digraph(comment='智能定价系统调用关系图', format='png')
        dot.attr(rankdir='TB', size='12,16')
        dot.attr('node', shape='box', style='rounded,filled')
        
        # 按模块分组
        module_colors = {
            'main.py': '#e1f5ff',
            'core.pricing_strategy_generator': '#fff4e1',
            'data.data_processor': '#ffe1f5',
            'data.feature_engineer': '#e1ffe1',
            'models.demand_predictor': '#f5e1ff',
            'models.pricing_optimizer': '#ffe1e1',
            'core.model_evaluator': '#e1e1ff',
            'core.config': '#f5f5e1'
        }
        
        # 添加节点
        for node_id, node_info in self.nodes.items():
            module = node_info['module']
            color = module_colors.get(module, '#ffffff')
            label = f"{node_info['label']}\n[{module}]"
            dot.node(node_id, label, fillcolor=color)
        
        # 添加边
        for from_node, to_node, label in self.edges:
            if label:
                dot.edge(from_node, to_node, label=label)
            else:
                dot.edge(from_node, to_node)
        
        # 保存
        output_path = dot.render(filename, cleanup=True)
        print(f"Graphviz 图表已保存到: {output_path}")
        return output_path
    
    def export_text(self, filename: str = "call_graph.txt"):
        """导出为文本格式"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("智能定价系统调用关系图\n")
            f.write("=" * 60 + "\n\n")
            
            # 按模块分组显示
            modules_dict = {}
            for node_id, node_info in self.nodes.items():
                module = node_info['module']
                if module not in modules_dict:
                    modules_dict[module] = []
                modules_dict[module].append((node_id, node_info))
            
            for module, nodes in sorted(modules_dict.items()):
                f.write(f"\n模块: {module}\n")
                f.write("-" * 60 + "\n")
                for node_id, node_info in nodes:
                    f.write(f"  {node_info['label']} ({node_info['type']})\n")
            
            f.write("\n\n调用关系:\n")
            f.write("=" * 60 + "\n")
            for from_node, to_node, label in self.edges:
                from_label = self.nodes[from_node]['label']
                to_label = self.nodes[to_node]['label']
                if label:
                    f.write(f"{from_label} --[{label}]--> {to_label}\n")
                else:
                    f.write(f"{from_label} --> {to_label}\n")
        
        print(f"文本格式图表已保存到: {filename}")
    
    def export_mermaid(self, filename: str = "call_graph_mermaid.md"):
        """导出为 Mermaid 格式"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("```mermaid\n")
            f.write("graph TB\n")
            
            # 添加节点（简化标签）
            for node_id, node_info in self.nodes.items():
                label = node_id.replace('_', ' ')
                f.write(f"    {node_id}[\"{node_info['label']}\"]\n")
            
            # 添加边
            for from_node, to_node, label in self.edges:
                if label:
                    f.write(f"    {from_node} -->|{label}| {to_node}\n")
                else:
                    f.write(f"    {from_node} --> {to_node}\n")
            
            f.write("```\n")
        
        print(f"Mermaid 格式图表已保存到: {filename}")


def main():
    """主函数"""
    generator = CallGraphGenerator()
    generator.build_from_main()
    
    # 导出多种格式
    generator.export_text("call_graph.txt")
    generator.export_mermaid("call_graph_mermaid.md")
    
    if HAS_GRAPHVIZ:
        generator.export_graphviz("call_graph")
    
    print("\n调用关系图生成完成！")
    print("生成的文件:")
    print("  - call_graph.txt (文本格式)")
    print("  - call_graph_mermaid.md (Mermaid格式)")
    if HAS_GRAPHVIZ:
        print("  - call_graph.png (Graphviz PNG格式)")


if __name__ == "__main__":
    main()


