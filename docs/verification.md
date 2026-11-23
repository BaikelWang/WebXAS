# 验证范围与公开实验

八个方法页面分别解释吸收谱预处理、预边与边位、线性组合拟合、复数 Fourier 变换、结构拟合、小波、系列趋势及 PCA/MCR。打开本地服务的 [/methods](http://127.0.0.1:8011/methods)，展开各方法的实验示例，可查看原始来源、处理说明、适用范围及可再发布的图件和 CSV。

| 方法 | RC2 冻结验证 | 证据边界 |
|---|---|---|
| G01 差谱 | 对实测谱作独立逐点差分 | 差谱振幅受归一化影响 |
| G02 边位、预边 | 与实验扫描的独立导数定义、作者算法及已知面积控制相对照 | 积分依赖背景、窗口和符号约定 |
| G03 参考谱 LCF | 配方质量分数的预设验收未通过 | 公开系数仅供探索，不能据此报告混合物定量回收 |
| G04 Fourier | 在实测 χ(k) 上与直接复数 DFT 比较 | 未相移校正的峰位不能直接换算键长 |
| G05 结构拟合 | 双温 Cu 回归与温度匹配的衍射参照；图库含 ZnSe 同引擎复算 | 配位数依赖 S₀²、路径与窗口 |
| G06 小波 | 实测 Cu 谱上进行独立离散核计算 | 热区不唯一识别散射元素 |
| G07 系列 | 对照 Ge 原始表格；图库另展示 Fe 三种已知配方 | 样品序列中的坐标取决于实验设计 |
| G08 分解 | 对 Au 保留样本及独立 MCR 求解器作比较 | 成分的尺度、符号、排列和旋转存在不确定性 |

### 图件与输入的出处

WebXAS 绘制的 G01、G03、G07、G08 使用 Sebastian Praetz、Christopher Schlesiger、Damian Alexander Motz、Lena Heinrich、Eva Maria Heppke、Stephen Klimke 和 Moritz Jahns 的 Fe 胶带数据。处理后的结果保留作者署名与许可。[原始数据](https://doi.org/10.5281/zenodo.14959531) · [研究论文](https://doi.org/10.1371/journal.pone.0323678) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

G02 使用 Thales Chamarelli Felipe、Alexandre Henrique Ferraz、Douglas Galante、Simone Bernardini 和 Christian Millo 的实测 Fe 荧光扫描，经 WebXAS 预处理和背景拟合后绘图。[原始数据](https://doi.org/10.5281/zenodo.22554443) · [研究论文](https://doi.org/10.1038/s41598-026-75019-9) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

G04、G05、G06 提供数据出处和处理边界。ZnSe 验证脚本按固定提交取回实测谱并校验 SHA-256。[Cu 实验记录](https://www.xaslib.xrayabsorption.org/spectrum/91/) · [ZnSe 谱与 Larch 示例](https://github.com/xraypy/xraylarch/blob/860d8a690c81eefb0e61dee4ca3703ef4b67e93d/examples/xafsdata/znse_zn_xafs.001)

### 从候选包同步

在公开仓库的干净检出中执行 `python tools/sync_candidate.py /path/to/candidate/package/app 0.5.2rc2`。该脚本逐个导入 `tools/release-files.txt` 中审核过的源码路径，并适配安装后的数据目录。审阅新增及已删除文件、公共知识库、图库署名、数据许可、实际 wheel 文件清单及差异。测试夹具由 `tools/fetch_test_inputs.py` 按上游提交与 SHA-256 重建。

在 Python 3.12 环境安装 `.[test]` 后执行 `python tools/fetch_test_inputs.py` 与 `python -m pytest -q package/app/tests`。严格排版回归要求系统的 Times New Roman 四种字形，Arial 或 Liberation Sans 用于无衬线模板。Windows 专用用例覆盖深路径 worker 和共享读取时的心跳更新。源码回归和隔离目录中的 wheel 测试使用独立入口。

[RC2 数值摘要](rc2-scientific-results.json) 保留各项对照的公开输入哈希、冻结结果哈希、单位及误差。G03 胶带样品同时列出称量的质量分数、Fe 原子系数、换算的质量分数和百分点误差。图表中的 LCF 展示窗口与质量分数验收窗口各自单独记录。
