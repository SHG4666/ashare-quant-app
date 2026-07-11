# A股量化研究台

一个以盘后研究为主的 Streamlit 应用，覆盖行情校验、策略回测、全市场自动选股、自选股扫描、次日交易计划和复盘记录。

> 仅用于研究学习，不构成投资建议。

## 在线部署

本项目适合使用 **GitHub + Streamlit Community Cloud** 部署：

1. 代码保存在 GitHub 仓库。
2. Streamlit Community Cloud 从 GitHub 自动构建应用。
3. 部署后获得固定的 `https://xxx.streamlit.app` 访问地址。
4. 后续推送到 GitHub 的代码会自动触发重新部署。

GitHub Pages 只能运行静态 HTML/CSS/JavaScript，不能直接运行本项目的 Python、Streamlit 和行情处理逻辑。

在 Streamlit Community Cloud 创建应用时使用：

- Repository：本项目的 GitHub 仓库
- Branch：`main`
- Main file path：`app.py`
- Python version：`3.11`

## 云端数据说明

完整的 Sequoia-X SQLite 行情库只保留在本机，不上传 GitHub。仓库内的 `cloud_data/` 包含：

- 当前自选股票池的三年历史行情种子
- 轻量全市场候选快照

云端运行时：

- 默认股票和当前股票池优先使用种子行情，打开和扫描速度稳定。
- 未包含在种子数据中的股票继续尝试在线行情源。
- 全市场自动选股在没有本地数据库时使用候选快照。
- 页面会显示数据截止日期；种子数据变旧时不会伪装成实时数据。

更新云端种子数据：

```bash
python scripts/build_cloud_seed.py
```

该命令需要本机 Sequoia-X 数据库以及网络行情访问。

## 本地运行

本地推荐 Python 3.11；云端使用 Streamlit 当前支持的 Python 版本：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

默认打开：

```text
http://localhost:8501
```

如需指定本地数据路径：

```bash
export ASHARE_SEQUOIA_DB=/path/to/sequoia_v2.db
export ASHARE_MODULE_PATH=/path/to/Ashare.py
```

## 主要功能

### 单股研究

- 前复权、不复权和后复权价格口径
- 双均线、RSI、MACD、布林带策略
- 手续费、滑点、止损和止盈
- 收益、回撤、夏普、交易记录和月度/年度统计
- 策略对比、参数优化和等权组合回测

### 股票池与计划

- 自选股添加、更新和批量移除
- 全市场趋势、动量、量能、突破和波动评分
- 自选股策略扫描
- 正常市场价格校准
- 仓位、风险预算、止损价和目标价计划

### 复盘

- 执行偏差记录
- 纪律、执行和复盘评分
- 复盘日志 CSV 导出

## 自选股格式

默认股票池保存在 `data_cache/watchlist.txt`：

```text
600522 中天科技
002747 埃斯顿
```

云端容器的本地文件系统不是长期数据库。应用重新部署或容器重建后，自选股和复盘记录会恢复为 GitHub 仓库中的版本。重要修改应同步回仓库。

## 测试

```bash
pytest -q
```

## 项目结构

```text
.
├── app.py
├── ashare_quant/
├── cloud_data/
│   ├── market_candidates.csv
│   ├── market_snapshot.json
│   └── watchlist_history/
├── data_cache/
│   └── watchlist.txt
├── scripts/
│   └── build_cloud_seed.py
├── tests/
├── requirements.txt
├── packages.txt
└── .streamlit/config.toml
```
