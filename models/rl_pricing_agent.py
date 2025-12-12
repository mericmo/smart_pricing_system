# models/rl_pricing_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'done'))

class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.beta = 0.4  # 重要性采样权重指数
        self.beta_increment = 0.001
        
    def push(self, state, action, reward, next_state, done, error: float = None):
        """存储经验"""
        if error is None:
            error = abs(reward)  # 如果没有提供误差，使用奖励绝对值
        
        priority = (error + 1e-5) ** self.alpha
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(state, action, next_state, reward, done))
        else:
            self.buffer[self.position] = Transition(state, action, next_state, reward, done)
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        """采样批次"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        # 计算采样概率
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # 计算重要性采样权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        # 增加beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 获取批次数据
        batch = [self.buffer[idx] for idx in indices]
        
        states = torch.FloatTensor(np.array([t.state for t in batch]))
        actions = torch.LongTensor(np.array([t.action for t in batch]))
        rewards = torch.FloatTensor(np.array([t.reward for t in batch]))
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch]))
        dones = torch.FloatTensor(np.array([t.done for t in batch]))
        weights = torch.FloatTensor(weights)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices: List[int], errors: np.ndarray):
        """更新优先级"""
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha
    
    def __len__(self):
        return len(self.buffer)

class NoisyLinear(nn.Module):
    """带噪声的线性层，用于探索"""
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # 可训练参数
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        
        # 噪声参数
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """重置参数"""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """重置噪声"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """缩放噪声"""
        noise = torch.randn(size)
        noise = noise.sign().mul(noise.abs().sqrt())
        return noise
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)

class DuelingDQN(nn.Module):
    """Dueling DQN网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, 
                 noisy: bool = True):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.noisy = noisy
        
        # 特征提取层
        if noisy:
            self.feature = nn.Sequential(
                NoisyLinear(state_dim, hidden_dim),
                nn.ReLU(),
                NoisyLinear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        else:
            self.feature = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        
        # 优势流
        if noisy:
            self.advantage = nn.Sequential(
                NoisyLinear(hidden_dim, hidden_dim),
                nn.ReLU(),
                NoisyLinear(hidden_dim, action_dim)
            )
        else:
            self.advantage = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        
        # 价值流
        if noisy:
            self.value = nn.Sequential(
                NoisyLinear(hidden_dim, hidden_dim),
                nn.ReLU(),
                NoisyLinear(hidden_dim, 1)
            )
        else:
            self.value = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features = self.feature(x)
        advantage = self.advantage(features)
        value = self.value(features)
        
        # 合并优势流和价值流
        return value + advantage - advantage.mean(dim=1, keepdim=True)
    
    def reset_noise(self):
        """重置噪声"""
        if self.noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

class PricingEnvironment:
    """定价环境（强化学习）"""
    
    def __init__(self, config, demand_predictor, product_info: Dict,
                 initial_stock: int, promotion_hours: Tuple[int, int]):
        """
        初始化定价环境
        
        Args:
            config: 配置管理器
            demand_predictor: 需求预测器
            product_info: 商品信息
            initial_stock: 初始库存
            promotion_hours: 促销时段 (开始小时, 结束小时)
        """
        self.config = config
        self.demand_predictor = demand_predictor
        self.product_info = product_info
        self.initial_stock = initial_stock
        self.promotion_hours = promotion_hours
        
        # 状态空间维度
        self.state_dim = 8  # [库存比例, 时间比例, 当前折扣, 累计利润比例, 小时, 星期几, 价格弹性, 促销敏感度]
        
        # 动作空间
        self.action_dim = 7  # 7种折扣调整动作
        self.discount_actions = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # 7种折扣率
        
        # 环境参数
        self.max_steps = 8  # 2小时，每15分钟一个step
        self.time_step = 900  # 15分钟（秒）
        
        # 奖励参数
        self.profit_weight = 1.0
        self.clearance_bonus = 10.0
        self.waste_penalty = 5.0
        self.discount_change_penalty = 0.1
        
        # 重置环境
        self.reset()
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_step = 0
        self.current_stock = self.initial_stock
        self.cumulative_profit = 0
        self.current_discount = 1.0  # 初始不打折
        self.sales_history = []
        self.price_history = []
        
        # 计算初始时间特征
        current_hour = self.promotion_hours[0]
        self.current_hour = current_hour
        self.day_of_week = 0  # 假设是周一
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        # 时间比例
        time_ratio = self.current_step / self.max_steps
        
        # 库存比例
        stock_ratio = self.current_stock / max(self.initial_stock, 1)
        
        # 利润比例（归一化）
        max_possible_profit = (self.product_info['original_price'] - 
                             self.product_info['cost_price']) * self.initial_stock
        profit_ratio = self.cumulative_profit / max(max_possible_profit, 1)
        
        # 当前小时（标准化到0-1）
        hour_normalized = self.current_hour / 24.0
        
        # 星期几（标准化到0-1）
        day_normalized = self.day_of_week / 7.0
        
        # 价格弹性（从商品信息中获取）
        price_elasticity = self.product_info.get('price_elasticity', 1.2)
        elasticity_normalized = (price_elasticity - 0.5) / 2.5  # 假设弹性在0.5-3.0之间
        
        # 促销敏感度
        promotion_sensitivity = self.product_info.get('promotion_sensitivity', 1.0)
        sensitivity_normalized = (promotion_sensitivity - 0.5) / 2.0  # 假设在0.5-2.5之间
        
        state = np.array([
            stock_ratio,
            time_ratio,
            self.current_discount,
            profit_ratio,
            hour_normalized,
            day_normalized,
            elasticity_normalized,
            sensitivity_normalized
        ], dtype=np.float32)
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        # 获取折扣
        new_discount = self.discount_actions[action]
        
        # 计算折扣变化惩罚
        discount_change = abs(new_discount - self.current_discount)
        
        # 预测销售
        time_to_close = 1 - (self.current_step / self.max_steps)
        
        # 准备特征
        features = {
            'hist_avg_sales': self.product_info.get('hist_avg_sales', 10),
            'price_elasticity': self.product_info.get('price_elasticity', 1.2),
            'promotion_sensitivity': self.product_info.get('promotion_sensitivity', 1.0)
        }
        
        # 使用需求预测器预测销售
        if hasattr(self.demand_predictor, 'predict_demand'):
            predicted_sales = self.demand_predictor.predict_demand(
                features=features,
                discount_rate=new_discount,
                time_to_close=time_to_close,
                current_stock=self.current_stock,
                base_demand=self.product_info.get('base_demand_rate', 5)
            )
        else:
            # 如果需求预测器不可用，使用启发式模型
            predicted_sales = self._heuristic_demand_prediction(
                new_discount, time_to_close, self.current_stock
            )
        
        # 添加随机波动
        predicted_sales *= np.random.uniform(0.8, 1.2)
        
        # 实际销售量不能超过库存
        actual_sales = min(int(predicted_sales), self.current_stock)
        
        # 更新库存和利润
        price = self.product_info['original_price'] * new_discount
        cost = self.product_info['cost_price']
        profit = (price - cost) * actual_sales
        
        self.current_stock -= actual_sales
        self.cumulative_profit += profit
        self.current_discount = new_discount
        self.current_step += 1
        
        # 更新小时
        self.current_hour = (self.promotion_hours[0] + 
                           self.current_step * 0.25) % 24
        
        # 记录历史
        self.sales_history.append(actual_sales)
        self.price_history.append(price)
        
        # 计算奖励
        reward = self._calculate_reward(profit, actual_sales, discount_change)
        
        # 检查是否结束
        done = (self.current_step >= self.max_steps or 
                self.current_stock <= 0 or
                new_discount <= self.config.optimization_config.constraints.get('min_discount', 0.4))
        
        # 额外信息
        info = {
            'sales': actual_sales,
            'discount': new_discount,
            'price': price,
            'profit': profit,
            'stock': self.current_stock,
            'step': self.current_step,
            'time_ratio': self.current_step / self.max_steps
        }
        
        return self._get_state(), reward, done, info
    
    def _heuristic_demand_prediction(self, discount: float, 
                                    time_to_close: float,
                                    current_stock: int) -> float:
        """启发式需求预测"""
        base_demand = self.product_info.get('base_demand_rate', 5.0)
        price_elasticity = self.product_info.get('price_elasticity', 1.2)
        
        # 价格效应
        price_factor = (1.0 / discount) ** price_elasticity
        
        # 时间压力效应
        time_factor = 1.0 + (1.0 - time_to_close) * 2.0
        
        # 库存压力效应
        stock_factor = min(1.0 + current_stock / 50.0, 2.0)
        
        # 促销敏感度
        promo_sensitivity = self.product_info.get('promotion_sensitivity', 1.0)
        promo_factor = 1.0 + (1.0 - discount) * promo_sensitivity
        
        predicted = base_demand * price_factor * time_factor * stock_factor * promo_factor
        
        return predicted
    
    def _calculate_reward(self, profit: float, sales: int, 
                         discount_change: float) -> float:
        """计算奖励"""
        # 基础利润奖励
        reward = profit * self.profit_weight
        
        # 售罄奖励
        if self.current_stock <= 0:
            reward += self.clearance_bonus
        
        # 浪费惩罚（如果最后还有库存）
        if self.current_step >= self.max_steps and self.current_stock > 0:
            waste_loss = self.current_stock * self.product_info['cost_price'] * 0.5
            reward -= min(waste_loss, self.waste_penalty)
        
        # 折扣变化惩罚（鼓励稳定定价）
        reward -= discount_change * self.discount_change_penalty
        
        # 标准化奖励
        max_profit = (self.product_info['original_price'] - 
                     self.product_info['cost_price']) * self.initial_stock
        reward = reward / max(max_profit, 1)
        
        return reward
    
    def render(self, mode: str = 'human'):
        """渲染环境状态"""
        if mode == 'human':
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Stock: {self.current_stock}/{self.initial_stock}")
            print(f"Discount: {self.current_discount:.2f}")
            print(f"Cumulative Profit: {self.cumulative_profit:.2f}")
            print(f"Current Hour: {self.current_hour:.2f}")
            print("-" * 40)

class PricingRLAgent:
    """强化学习定价智能体"""
    
    def __init__(self, config, state_dim: int, action_dim: int):
        """
        初始化强化学习智能体
        
        Args:
            config: 配置管理器
            state_dim: 状态维度
            action_dim: 动作维度
        """
        self.config = config.rl_config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 网络
        self.policy_net = DuelingDQN(
            state_dim, action_dim, 
            hidden_dim=self.config.hidden_dim,
            noisy=True
        ).to(self.device)
        
        self.target_net = DuelingDQN(
            state_dim, action_dim,
            hidden_dim=self.config.hidden_dim,
            noisy=True
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 优化器
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=self.config.learning_rate
        )
        
        # 经验回放
        self.memory = PrioritizedReplayBuffer(
            capacity=self.config.memory_size,
            alpha=0.6
        )
        
        # 训练参数
        self.steps_done = 0
        self.episode_rewards = []
        self.loss_history = []
        
        # 日志
        self.logger = logging.getLogger(__name__)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """选择动作"""
        self.steps_done += 1
        
        if training:
            # 在训练时，网络已经通过噪声进行探索
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            
            return q_values.max(1)[1].item()
        else:
            # 在评估时，使用贪婪策略（重置噪声）
            self.policy_net.eval()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            
            self.policy_net.train()
            self.policy_net.reset_noise()
            
            return q_values.max(1)[1].item()
    
    def optimize_model(self):
        """优化模型"""
        if len(self.memory) < self.config.batch_size:
            return None
        
        # 采样
        states, actions, rewards, next_states, dones, weights, indices = \
            self.memory.sample(self.config.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # 计算当前Q值
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.config.gamma * next_q_values
        
        # 计算TD误差
        td_errors = (current_q_values.squeeze() - target_q_values).abs().cpu().numpy()
        
        # 计算损失
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values, reduction='none')
        loss = (loss * weights).mean()
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        # 重置噪声
        self.policy_net.reset_noise()
        
        # 更新优先级
        self.memory.update_priorities(indices, td_errors)
        
        return loss.item()
    
    def train_episode(self, env: PricingEnvironment) -> Dict[str, Any]:
        """训练一个episode"""
        state = env.reset()
        total_reward = 0
        done = False
        episode_info = []
        
        while not done:
            # 选择动作
            action = self.select_action(state, training=True)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            self.memory.push(state, action, reward, next_state, done)
            
            # 优化模型
            loss = self.optimize_model()
            if loss is not None:
                self.loss_history.append(loss)
            
            # 更新目标网络
            if self.steps_done % self.config.target_update_frequency == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # 更新状态
            state = next_state
            total_reward += reward
            episode_info.append(info)
        
        self.episode_rewards.append(total_reward)
        
        # 记录训练信息
        episode_stats = {
            'episode_reward': total_reward,
            'total_sales': sum(info['sales'] for info in episode_info),
            'total_profit': sum(info['profit'] for info in episode_info),
            'final_stock': episode_info[-1]['stock'] if episode_info else 0,
            'average_discount': np.mean([info['discount'] for info in episode_info]),
            'steps': len(episode_info)
        }
        
        return episode_stats
    
    def evaluate(self, env: PricingEnvironment, num_episodes: int = 10) -> Dict[str, Any]:
        """评估智能体"""
        self.policy_net.eval()
        
        episode_stats = []
        
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            episode_info = []
            
            while not done:
                action = self.select_action(state, training=False)
                next_state, reward, done, info = env.step(action)
                
                state = next_state
                episode_reward += reward
                episode_info.append(info)
            
            stats = {
                'reward': episode_reward,
                'sales': sum(info['sales'] for info in episode_info),
                'profit': sum(info['profit'] for info in episode_info),
                'final_stock': episode_info[-1]['stock'] if episode_info else 0,
                'discount_pattern': [info['discount'] for info in episode_info]
            }
            
            episode_stats.append(stats)
        
        self.policy_net.train()
        
        # 计算平均统计量
        avg_stats = {
            'avg_reward': np.mean([s['reward'] for s in episode_stats]),
            'avg_sales': np.mean([s['sales'] for s in episode_stats]),
            'avg_profit': np.mean([s['profit'] for s in episode_stats]),
            'avg_final_stock': np.mean([s['final_stock'] for s in episode_stats]),
            'clearance_rate': np.mean([1 if s['final_stock'] == 0 else 0 
                                      for s in episode_stats]),
            'discount_patterns': [s['discount_pattern'] for s in episode_stats]
        }
        
        return avg_stats
    
    def generate_pricing_strategy(self, env: PricingEnvironment) -> List[Dict]:
        """生成定价策略"""
        self.policy_net.eval()
        
        state = env.reset()
        done = False
        strategy = []
        
        while not done:
            action = self.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            # 记录策略
            time_slot = {
                'step': env.current_step,
                'discount': info['discount'],
                'price': info['price'],
                'expected_sales': info['sales'],
                'stock': info['stock'],
                'profit': info['profit']
            }
            
            strategy.append(time_slot)
            state = next_state
        
        self.policy_net.train()
        
        return strategy
    
    def save_model(self, path: str):
        """保存模型"""
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episode_rewards': self.episode_rewards,
            'loss_history': self.loss_history
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.steps_done = checkpoint.get('steps_done', 0)
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.loss_history = checkpoint.get('loss_history', [])
        
        self.logger.info(f"模型已从 {path} 加载")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        if not self.episode_rewards:
            return {}
        
        recent_rewards = self.episode_rewards[-100:]  # 最近100个episode
        
        return {
            'total_episodes': len(self.episode_rewards),
            'avg_reward_last_100': np.mean(recent_rewards) if recent_rewards else 0,
            'max_reward': np.max(self.episode_rewards) if self.episode_rewards else 0,
            'min_reward': np.min(self.episode_rewards) if self.episode_rewards else 0,
            'avg_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0,
            'memory_size': len(self.memory),
            'exploration_rate': self._get_exploration_rate()
        }
    
    def _get_exploration_rate(self) -> float:
        """获取探索率"""
        # 由于使用噪声网络，探索是内置的
        # 这里返回一个固定值表示使用了噪声探索
        return 1.0

class RLPricingTrainer:
    """强化学习定价训练器"""
    
    def __init__(self, config, demand_predictor):
        """
        初始化训练器
        
        Args:
            config: 配置管理器
            demand_predictor: 需求预测器
        """
        self.config = config
        self.demand_predictor = demand_predictor
        self.agent = None
        self.env = None
        self.logger = logging.getLogger(__name__)
    
    def create_environment(self, product_info: Dict, 
                          initial_stock: int,
                          promotion_hours: Tuple[int, int]) -> PricingEnvironment:
        """创建环境"""
        self.env = PricingEnvironment(
            config=self.config,
            demand_predictor=self.demand_predictor,
            product_info=product_info,
            initial_stock=initial_stock,
            promotion_hours=promotion_hours
        )
        
        return self.env
    
    def create_agent(self) -> PricingRLAgent:
        """创建智能体"""
        if self.env is None:
            raise ValueError("请先创建环境")
        
        self.agent = PricingRLAgent(
            config=self.config,
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim
        )
        
        return self.agent
    
    def train(self, num_episodes: int = None, 
              callback: callable = None) -> Dict[str, Any]:
        """训练智能体"""
        if self.env is None or self.agent is None:
            raise ValueError("请先创建环境和智能体")
        
        if num_episodes is None:
            num_episodes = self.config.rl_config.training_episodes
        
        self.logger.info(f"开始训练，总episodes: {num_episodes}")
        
        training_stats = {
            'episode_rewards': [],
            'episode_sales': [],
            'episode_profits': [],
            'clearance_rates': []
        }
        
        for episode in range(num_episodes):
            # 训练一个episode
            stats = self.agent.train_episode(self.env)
            
            # 记录统计信息
            training_stats['episode_rewards'].append(stats['episode_reward'])
            training_stats['episode_sales'].append(stats['total_sales'])
            training_stats['episode_profits'].append(stats['total_profit'])
            training_stats['clearance_rates'].append(1 if stats['final_stock'] == 0 else 0)
            
            # 定期评估
            if episode % self.config.rl_config.evaluation_interval == 0:
                eval_stats = self.agent.evaluate(self.env, num_episodes=5)
                
                self.logger.info(
                    f"Episode {episode}: "
                    f"Reward={stats['episode_reward']:.2f}, "
                    f"Sales={stats['total_sales']}, "
                    f"Profit={stats['total_profit']:.2f}, "
                    f"Stock={stats['final_stock']}, "
                    f"Avg Discount={stats['average_discount']:.2f}"
                )
                
                self.logger.info(
                    f"Evaluation: "
                    f"Avg Reward={eval_stats['avg_reward']:.2f}, "
                    f"Clearance Rate={eval_stats['clearance_rate']:.2f}"
                )
            
            # 回调函数
            if callback:
                callback(episode, stats)
        
        # 最终评估
        final_eval = self.agent.evaluate(self.env, num_episodes=20)
        
        training_stats['final_evaluation'] = final_eval
        training_stats['agent_stats'] = self.agent.get_training_stats()
        
        self.logger.info(f"训练完成，最终评估: {final_eval}")
        
        return training_stats
    
    def generate_optimal_strategy(self, product_info: Dict,
                                 initial_stock: int,
                                 promotion_hours: Tuple[int, int]) -> Dict[str, Any]:
        """生成最优策略"""
        # 创建环境
        env = self.create_environment(
            product_info=product_info,
            initial_stock=initial_stock,
            promotion_hours=promotion_hours
        )
        
        # 如果智能体不存在，创建并加载预训练模型
        if self.agent is None:
            self.agent = self.create_agent()
            # 这里可以加载预训练模型
            # self.agent.load_model("pretrained_model.pth")
        
        # 生成策略
        strategy = self.agent.generate_pricing_strategy(env)
        
        # 计算统计信息
        total_sales = sum(s['expected_sales'] for s in strategy)
        total_profit = sum(s['profit'] for s in strategy)
        clearance_rate = 1 - strategy[-1]['stock'] / initial_stock if initial_stock > 0 else 1
        
        # 格式化策略
        formatted_strategy = []
        for i, step in enumerate(strategy):
            time_slot = {
                'time_slot': i + 1,
                'discount': round(step['discount'], 3),
                'discount_percentage': f"{round((1-step['discount'])*100, 1)}%",
                'price': round(step['price'], 2),
                'expected_sales': step['expected_sales'],
                'expected_profit': round(step['profit'], 2),
                'remaining_stock': step['stock']
            }
            formatted_strategy.append(time_slot)
        
        result = {
            'strategy': formatted_strategy,
            'summary': {
                'total_expected_sales': total_sales,
                'total_expected_profit': round(total_profit, 2),
                'clearance_rate': round(clearance_rate, 3),
                'average_discount': round(np.mean([s['discount'] for s in strategy]), 3),
                'strategy_type': 'reinforcement_learning'
            }
        }
        
        return result