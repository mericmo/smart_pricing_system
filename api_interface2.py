# api_interface.py
from flask import Flask, request, jsonify, current_app
from typing import Dict, Any
import json
import numpy as np
from discount_optimizer import DiscountOptimizer
from algorithm import SalesPredictor
from feature_store import FeatureStore
from data_preprocessor import DataPreprocessor
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局对象
discount_optimizer = None
model_predictor = None
feature_store = None
data_preprocessor = None

def initialize_services():
    """初始化服务"""
    global discount_optimizer, model_predictor, feature_store, data_preprocessor
    
    try:
        logger.info("正在初始化服务组件...")
        
        # 初始化各组件
        feature_store = FeatureStore()
        data_preprocessor = DataPreprocessor()
        model_predictor = SalesPredictor(forecast_horizon=7)
        
        # 初始化折扣优化器
        discount_optimizer = DiscountOptimizer(
            model_predictor=model_predictor,
            feature_store=feature_store,
            data_preprocessor=data_preprocessor
        )
        
        logger.info("服务组件初始化完成")
        return True
        
    except Exception as e:
        logger.error(f"初始化服务失败: {str(e)}")
        return False

def _run_sensitivity_analysis(discount_plan, parameter_name):
    """
    运行敏感性分析
    
    参数:
    - discount_plan: 折扣方案
    - parameter_name: 分析参数名称
    
    返回:
    - 敏感性分析结果
    """
    try:
        if parameter_name == 'price_elasticity':
            # 价格弹性敏感性分析
            elasticities = [-1.0, -1.5, -2.0, -2.5, -3.0]
            results = []
            
            for elasticity in elasticities:
                # 模拟不同弹性下的销量变化
                total_sales = sum(item['expected_sales'] for item in discount_plan)
                total_profit = sum(item['expected_profit'] for item in discount_plan)
                
                # 简化计算：假设销量随弹性线性变化
                elasticity_factor = elasticity / -2.0  # 以-2.0为基准
                adjusted_sales = total_sales * elasticity_factor
                adjusted_profit = total_profit * elasticity_factor
                
                results.append({
                    'elasticity': elasticity,
                    'expected_sales': max(0, adjusted_sales),
                    'expected_profit': max(0, adjusted_profit),
                    'sales_change_percent': (elasticity_factor - 1) * 100
                })
            
            return {
                'parameter': 'price_elasticity',
                'base_value': -2.0,
                'variations': elasticities,
                'results': results
            }
            
        elif parameter_name == 'sales_rate':
            # 销售速率敏感性分析
            rate_factors = [0.5, 0.75, 1.0, 1.25, 1.5]
            results = []
            
            for factor in rate_factors:
                total_sales = sum(item['expected_sales'] for item in discount_plan)
                total_profit = sum(item['expected_profit'] for item in discount_plan)
                
                adjusted_sales = total_sales * factor
                adjusted_profit = total_profit * factor
                
                results.append({
                    'rate_factor': factor,
                    'expected_sales': max(0, adjusted_sales),
                    'expected_profit': max(0, adjusted_profit),
                    'sales_change_percent': (factor - 1) * 100
                })
            
            return {
                'parameter': 'sales_rate',
                'base_value': 1.0,
                'variations': rate_factors,
                'results': results
            }
            
        else:
            return {
                'parameter': parameter_name,
                'error': '不支持该参数的分析'
            }
            
    except Exception as e:
        logger.error(f"敏感性分析失败: {str(e)}")
        return {
            'parameter': parameter_name,
            'error': str(e)
        }

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    try:
        # 检查服务状态
        services_ready = all([
            discount_optimizer is not None,
            model_predictor is not None,
            feature_store is not None,
            data_preprocessor is not None
        ])
        
        status = 'healthy' if services_ready else 'unhealthy'
        message = '所有服务正常运行' if services_ready else '部分服务未初始化'
        
        return jsonify({
            'status': status,
            'service': 'discount_optimizer',
            'message': message,
            'timestamp': json.dumps(str, default=str)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/discount-plan', methods=['POST'])
def generate_discount_plan():
    """
    生成折扣方案API接口
    
    请求体示例:
    {
        "product_code": "4834512",
        "current_inventory": 100,
        "promotion_window": ["20:00", "22:00"],
        "min_gross_margin": 0.1,
        "allow_staggered_pricing": true,
        "current_date": "2025-11-01",
        "base_price": 7.99,
        "cost_price": 5.59,
        "historical_sales_data": [...]  # 可选
    }
    """
    try:
        # 检查服务是否初始化
        if discount_optimizer is None:
            return jsonify({
                'success': False,
                'error': '折扣优化服务未初始化',
                'message': '请先初始化服务'
            }), 503
        
        # 获取请求数据
        request_data = request.get_json()
        
        if not request_data:
            return jsonify({
                'success': False,
                'error': '请求体不能为空'
            }), 400
        
        # 验证必要参数
        required_fields = ['product_code', 'current_inventory']
        for field in required_fields:
            if field not in request_data:
                return jsonify({
                    'success': False,
                    'error': f'缺少必要参数: {field}'
                }), 400
        
        # 记录请求
        logger.info(f"生成折扣方案请求 - 商品: {request_data['product_code']}, 库存: {request_data['current_inventory']}")
        
        # 生成折扣方案
        discount_plan = discount_optimizer.generate_discount_plan(request_data)
        
        return jsonify({
            'success': True,
            'data': discount_plan,
            'message': '折扣方案生成成功'
        })
        
    except ValueError as e:
        logger.warning(f"参数验证失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': '参数验证失败'
        }), 400
    except Exception as e:
        logger.error(f"生成折扣方案失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': '生成折扣方案失败'
        }), 500

@app.route('/api/batch-discount-plan', methods=['POST'])
def generate_batch_discount_plan():
    """
    批量生成折扣方案API接口
    """
    try:
        if discount_optimizer is None:
            return jsonify({
                'success': False,
                'error': '折扣优化服务未初始化'
            }), 503
        
        request_data = request.get_json()
        
        if not request_data or 'products' not in request_data:
            return jsonify({
                'success': False,
                'error': '请求体必须包含products数组'
            }), 400
        
        logger.info(f"批量生成折扣方案请求 - 商品数量: {len(request_data['products'])}")
        
        results = []
        for i, product_data in enumerate(request_data['products']):
            try:
                logger.debug(f"处理第{i+1}个商品: {product_data.get('product_code', 'unknown')}")
                plan = discount_optimizer.generate_discount_plan(product_data)
                results.append({
                    'product_code': product_data.get('product_code'),
                    'success': True,
                    'plan': plan
                })
            except Exception as e:
                logger.error(f"处理商品失败: {product_data.get('product_code', 'unknown')}, 错误: {str(e)}")
                results.append({
                    'product_code': product_data.get('product_code', 'unknown'),
                    'success': False,
                    'error': str(e)
                })
        
        success_count = sum(1 for r in results if r['success'])
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(results),
            'success_count': success_count,
            'failure_count': len(results) - success_count
        })
        
    except Exception as e:
        logger.error(f"批量生成折扣方案失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/plan-simulation', methods=['POST'])
def simulate_discount_plan():
    """
    模拟折扣方案效果API接口
    
    请求体示例:
    {
        "discount_plan": [...],  # 折扣方案数据
        "simulation_params": {
            "current_inventory": 100,
            "base_price": 10.0,
            "cost_price": 7.0,
            "scenarios": ["optimistic", "pessimistic", "normal"]
        }
    }
    """
    try:
        if discount_optimizer is None:
            return jsonify({
                'success': False,
                'error': '折扣优化服务未初始化'
            }), 503
        
        request_data = request.get_json()
        
        # 获取方案和模拟参数
        discount_plan = request_data.get('discount_plan')
        simulation_params = request_data.get('simulation_params', {})
        
        if not discount_plan:
            return jsonify({
                'success': False,
                'error': '缺少折扣方案数据'
            }), 400
        
        logger.info("模拟折扣方案效果")
        
        # 计算基础案例指标
        current_inventory = simulation_params.get('current_inventory', 100)
        base_price = simulation_params.get('base_price', 10.0)
        cost_price = simulation_params.get('cost_price', 7.0)
        
        # 使用折扣优化器的计算方法
        plan_metrics = discount_optimizer._calculate_plan_metrics(
            discount_plan, current_inventory, base_price, cost_price
        )
        
        # 运行敏感性分析 - 使用内部函数，而不是self
        sensitivity_analysis = {
            'price_elasticity': _run_sensitivity_analysis(discount_plan, 'price_elasticity'),
            'sales_rate': _run_sensitivity_analysis(discount_plan, 'sales_rate')
        }
        
        # 场景分析
        scenarios = simulation_params.get('scenarios', ['optimistic', 'pessimistic', 'normal'])
        scenario_results = {}
        
        for scenario in scenarios:
            if scenario == 'optimistic':
                # 乐观场景：销售速率提高25%
                adjusted_plan = discount_plan.copy()
                for item in adjusted_plan:
                    item['expected_sales'] = int(item['expected_sales'] * 1.25)
                    item['expected_revenue'] = item['expected_sales'] * item.get('final_price', base_price)
                    item['expected_profit'] = item['expected_sales'] * (item.get('final_price', base_price) - cost_price)
                scenario_results[scenario] = discount_optimizer._calculate_plan_metrics(
                    adjusted_plan, current_inventory, base_price, cost_price
                )
                
            elif scenario == 'pessimistic':
                # 悲观场景：销售速率降低25%
                adjusted_plan = discount_plan.copy()
                for item in adjusted_plan:
                    item['expected_sales'] = int(item['expected_sales'] * 0.75)
                    item['expected_revenue'] = item['expected_sales'] * item.get('final_price', base_price)
                    item['expected_profit'] = item['expected_sales'] * (item.get('final_price', base_price) - cost_price)
                scenario_results[scenario] = discount_optimizer._calculate_plan_metrics(
                    adjusted_plan, current_inventory, base_price, cost_price
                )
                
            elif scenario == 'normal':
                # 正常场景：使用原方案
                scenario_results[scenario] = plan_metrics
        
        simulation_result = {
            'base_case': plan_metrics,
            'sensitivity_analysis': sensitivity_analysis,
            'scenario_analysis': scenario_results,
            'risk_assessment': {
                'worst_case_profit': scenario_results.get('pessimistic', {}).get('total_expected_profit', 0),
                'best_case_profit': scenario_results.get('optimistic', {}).get('total_expected_profit', 0),
                'profit_range': scenario_results.get('optimistic', {}).get('total_expected_profit', 0) - 
                               scenario_results.get('pessimistic', {}).get('total_expected_profit', 0),
                'clearance_confidence': min(1.0, plan_metrics.get('clearance_rate', 0) * 1.2)  # 置信度估计
            }
        }
        
        return jsonify({
            'success': True,
            'simulation_result': simulation_result,
            'message': '模拟分析完成'
        })
        
    except Exception as e:
        logger.error(f"模拟折扣方案失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': '模拟分析失败'
        }), 500

@app.route('/api/optimize-discount', methods=['POST'])
def optimize_discount_directly():
    """
    直接优化折扣方案API接口
    提供更灵活的优化参数
    """
    try:
        if discount_optimizer is None:
            return jsonify({
                'success': False,
                'error': '折扣优化服务未初始化'
            }), 503
        
        request_data = request.get_json()
        
        if not request_data:
            return jsonify({
                'success': False,
                'error': '请求体不能为空'
            }), 400
        
        # 提取优化参数
        optimization_params = {
            'product_code': request_data.get('product_code'),
            'current_inventory': request_data.get('current_inventory'),
            'promotion_window': request_data.get('promotion_window', ['20:00', '22:00']),
            'min_gross_margin': request_data.get('min_gross_margin', 0.1),
            'allow_staggered_pricing': request_data.get('allow_staggered_pricing', True),
            'current_date': request_data.get('current_date'),
            'base_price': request_data.get('base_price'),
            'cost_price': request_data.get('cost_price'),
            'historical_sales_data': request_data.get('historical_sales_data'),
            
            # 优化算法参数
            'time_slots': request_data.get('time_slots', 4),
            'customer_perception_weight': request_data.get('customer_perception_weight', 0.3),
            'max_discount_change': request_data.get('max_discount_change', 0.2),
            'discount_options': request_data.get('discount_options', [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]),
            'price_elasticity': request_data.get('price_elasticity', -2.0)
        }
        
        # 更新折扣优化器的参数
        if 'time_slots' in request_data:
            discount_optimizer.default_params['time_slots'] = request_data['time_slots']
        if 'customer_perception_weight' in request_data:
            discount_optimizer.default_params['customer_perception_weight'] = request_data['customer_perception_weight']
        if 'max_discount_change' in request_data:
            discount_optimizer.default_params['max_discount_change'] = request_data['max_discount_change']
        
        logger.info(f"直接优化折扣方案 - 参数: {optimization_params}")
        
        # 生成折扣方案
        discount_plan = discount_optimizer.generate_discount_plan(optimization_params)
        
        return jsonify({
            'success': True,
            'data': discount_plan,
            'optimization_params': optimization_params,
            'message': '优化完成'
        })
        
    except Exception as e:
        logger.error(f"直接优化折扣方案失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': '优化失败'
        }), 500

@app.route('/api/system-info', methods=['GET'])
def get_system_info():
    """获取系统信息"""
    try:
        system_info = {
            'service_name': 'Smart Discount Optimization System',
            'version': '1.0.0',
            'status': 'running' if discount_optimizer else 'initializing',
            'components': {
                'discount_optimizer': discount_optimizer is not None,
                'model_predictor': model_predictor is not None,
                'feature_store': feature_store is not None,
                'data_preprocessor': data_preprocessor is not None
            },
            'endpoints': [
                {'path': '/health', 'method': 'GET', 'description': '健康检查'},
                {'path': '/api/discount-plan', 'method': 'POST', 'description': '生成折扣方案'},
                {'path': '/api/batch-discount-plan', 'method': 'POST', 'description': '批量生成折扣方案'},
                {'path': '/api/plan-simulation', 'method': 'POST', 'description': '模拟折扣方案效果'},
                {'path': '/api/optimize-discount', 'method': 'POST', 'description': '直接优化折扣方案'},
                {'path': '/api/system-info', 'method': 'GET', 'description': '获取系统信息'}
            ],
            'default_parameters': discount_optimizer.default_params if discount_optimizer else {}
        }
        
        return jsonify({
            'success': True,
            'system_info': system_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def run_api_server(host='0.0.0.0', port=5000, debug=False):
    """启动API服务器"""
    try:
        # 初始化服务
        if not initialize_services():
            logger.error("服务初始化失败，无法启动API服务器")
            return
        
        logger.info(f"启动API服务器: http://{host}:{port}")
        logger.info("可用端点:")
        logger.info("  GET  /health                    - 健康检查")
        logger.info("  POST /api/discount-plan        - 生成折扣方案")
        logger.info("  POST /api/batch-discount-plan  - 批量生成折扣方案")
        logger.info("  POST /api/plan-simulation      - 模拟折扣方案效果")
        logger.info("  POST /api/optimize-discount    - 直接优化折扣方案")
        logger.info("  GET  /api/system-info          - 获取系统信息")
        
        app.run(host=host, port=port, debug=debug)
        
    except Exception as e:
        logger.error(f"启动API服务器失败: {str(e)}")
        raise

if __name__ == '__main__':
    run_api_server(debug=True)