# Spatial-Interpolation

三维空间插值 —— 多种插值方法的对比实验。

## 数据

`Data/` 下为钻孔采样数据：

| 文件 | 说明 |
| --- | --- |
| `data_1.xlsx` ~ `data_10.xlsx` | 分组数据，前 9 列（`usecols=[0..8]`）为特征，第 10 列（`usecols=[9]`）为待插值的目标值 |
| `Data.xlsx` / `address.xlsx` | 汇总数据与坐标 |

脚本会把这 10 组数据拼成 `[10, 13, 9]` 的三维结构，再展平为 `[130, 9]` 送入各模型，
训练 / 测试通过显式索引划分。

## 方法

| 文件 | 方法 |
| --- | --- |
| `Kriging3D.py` | 三维普通克里金（`pykrige.OrdinaryKriging3D`，高斯变异函数） |
| `SVR.py` | 支持向量回归 |
| `3dCNN.py` | 3D CNN |
| `3dCNN_attention.py` | 3D CNN + 注意力 |
| `3dCNN_Parameters.py` | 3D CNN 参数对比实验 |
| `patch.py` / `point.py` | 滑窗切片与点位处理 |
| `draw.py` / `draw_y.py` / `draw_Box-plot.py` / `redraw_loss.py` | 结果成图、箱线图、损失曲线 |
| `predict.py` / `test.py` | 推理与测试 |
| `X_scalar` / `y_scalar` / `X_all_scalar` / `y_all_scalar` | 归一化器中间文件 |

## 评估

各方法统一用 **MSE / MAE / R²** 对比，测试集从 `result/X_test.npy`、`result/y_test.npy` 读取，
预测结果写回 `result/`（如 `Kriging_y_pred.txt`）。

## 环境

```
numpy
pandas
scikit-learn
pykrige
openpyxl
tensorflow / keras
matplotlib
```

## 使用

脚本内均为相对路径，请在仓库根目录执行：

```bash
python Kriging3D.py
python SVR.py
python 3dCNN.py
```

> `result/` 目录（测试集与预测结果）未入库，需先由 `patch.py` 等脚本生成。

---

> 本 README 依据仓库现有代码整理，细节如与实现不符请以代码为准。
