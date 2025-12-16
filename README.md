# Smart Pricing System (草稿 -> 可生产化路线)

目标：在指定时间窗口内（例如 20:00–22:00）对“日清品”生成阶梯式动态折扣方案，确保售罄的同时尽可能最大化毛利并满足业务约束（最小毛利率、最大折扣等）。

快速开始（MVP）:
1. 新增依赖：请查看 requirements.txt 并安装。
2. 将 core/forecast.py, core/optimizer.py, utils/simulator.py 添加到仓库后，运行回测脚本（待添加）进行本地模拟。
3. 推荐工作流：先在历史数据上回测，再在单店 A/B 测试，最后逐步 rollout。

主要模块：
- forecast: 基线销量与价格弹性模型
- optimizer: 离线/在线折扣求解器
- api: （待实现）暴露策略生成与销量上报接口
- simulator: 用于回测与健壮性测试

下一步：
- 补全 main.py（拆分为启动脚本与 task scheduler）
- 添加单元测试与 CI，编写入库/变更审批流程。