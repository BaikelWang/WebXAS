# WebXAS

WebXAS 是面向 X 射线吸收谱（XAS）的本地分析工具，支持谱数据导入、归一化与背景处理、XANES 对齐与比较、基于明确 FEFF 路径和参数约束的 EXAFS 拟合，以及图件、工程和报告导出。分析内核随源码发布，依赖由 Python 包管理器安装。

## 本地运行

使用 Python 3.12。以下命令从项目根目录执行，首次安装需连接 Python 软件包索引。

```bash
# Linux / macOS
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m uvicorn xasweb.app:app --host 127.0.0.1 --port 8011
```

```powershell
# Windows PowerShell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m uvicorn xasweb.app:app --host 127.0.0.1 --port 8011
```

打开 [快速初筛](http://127.0.0.1:8011/)、[自动工作台](http://127.0.0.1:8011/auto)、[专业工作台](http://127.0.0.1:8011/pro)、[知识库](http://127.0.0.1:8011/knowledge) 或 [API 文档](http://127.0.0.1:8011/docs)。项目和任务默认保存在本地 `.runtime/`，可用 `XASWEB_PROJECT_ROOT` 和 `XASWEB_JOB_ROOT` 指定独立目录。

自动工作台的“合成示例”在运行时生成 Cu K 边数学谱，可用于验证导入、处理和分析流程。专业工作台的“打开合成演示”载入三条同引擎生成的 χ(k) 和单散射、多重散射理论路径，用于复现联合拟合。理论路径取自 ParSeq-XAS 固定提交 `b60f1ac63f20d1ee792223f2ec2e06aadb0cd124`，原始来源、提交和哈希保留在示例工程中。[示例路径来源](https://github.com/kklmn/ParSeq-XAS/tree/b60f1ac63f20d1ee792223f2ec2e06aadb0cd124) · [ParSeq-XAS 示例许可](xasweb/data/PARSEQ-XAS-LICENSE.txt)

公开知识库包含三篇入门指南和一条 Larch 官方文档链接。分析使用 Larch 2026.3.1，专业拟合可导入已有的 FEFF 理论路径。根据结构生成新路径时，FEFF 可执行文件的可用性取决于所安装的 Larch 及平台。[Larch 官方文档](https://xraypy.github.io/xraylarch/)

## 分析结果的适用范围

- 归一化和背景处理的结果取决于所选区间、E₀、有效 k 范围及处理参数。
- Fourier 变换所得相位未校正的 R 空间峰位需要结合理论路径和拟合解释。
- 拟合结果依赖给定的结构模型、路径、拟合窗口及噪声处理。结合残差、参数相关性和不确定度进行判断。
- 合成示例用于验证软件行为和同引擎计算。实验精度评估需要来源明确的参考谱、可复核的分析条件和不确定度。

## 公开验证与安装包自检

[方法、实验图件与科学验证](docs/verification.md) 汇总八类方法的证据范围，并标明 G03 定量验收结果。运行服务后可在[方法与实验示例](http://127.0.0.1:8011/methods)查看公开图件及其数据出处。严格 Times New Roman 模板需要系统已安装该字体的常规、粗体、斜体和粗斜体字形。科学图件导出使用系统字体。

在新建的 Python 3.12 虚拟环境中，从仓库根目录构建 wheel，并执行独立安装测试。命令适用于已激活环境的 PowerShell 或 Bash。

```bash
python -m pip wheel --no-deps --wheel-dir dist .
python -c "import glob, subprocess, sys; wheels = glob.glob('dist/*.whl'); assert len(wheels) == 1; subprocess.check_call([sys.executable, '-m', 'pip', 'install', wheels[0] + '[test]'])"
python tools/wheel_smoke.py
```

测试脚本从上游固定提交取回 ZnSe 实测谱、校验原始字节，在临时目录启动服务，导入实际谱，并将专业工程导出为 ZIP 后重新打开。用户也可在专业工作台执行相同步骤。默认运行目录是启动位置的 `.runtime`，可通过 `XASWEB_RUNTIME_ROOT` 指定。实验数据和理论路径各自保留上游归属与适用条件。

归档前可按[工程操作与科学复核步骤](docs/workflows.md)核对输入、处理状态、联合模型及导出记录。

[RC3 工程和实验验收](docs/rc3-results.md) 给出 Fe 称量配方、Cu 箔低温两壳、KEK 振幅传递与 NIMS 氧化物对照的结果和限制。
