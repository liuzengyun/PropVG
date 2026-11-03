# 测试与评估脚本

项目自有的测试脚本统一放在本目录。训练入口仍为`tools/train.py`，数据转换、分析及可视化工具仍放在`tools/`；`propvg/apis/test.py`是训练中调用的评估库，不是独立脚本，位置不变。第三方代码和历史实验备份不移动。

所有命令从项目根目录执行，使用原来的`propvg`环境。这里只调整路径和导入方式，不改变模型、损失、数据或评估算法。

## 正式模型评估

```bash
python tools/test_script/eval_aerialvg.py /实验目录/保存的配置.py \
  --checkpoint /实验目录/top1_best.pth --split test

python tools/test_script/test.py /配置文件.py --load-from /模型.pth

bash tools/test_script/dist_test.sh /配置文件.py 4 --load-from /模型.pth
```

`multi_test.sh`保留原来多组历史模型测试命令，只同步更新入口路径。执行前自行检查其中的模型路径和显卡设置，不应把它作为单元测试自动执行。

## CPU回归测试

全部回归测试：

```bash
PYTHONWARNINGS=ignore python -m unittest discover -s tools/test_script -p 'test_*.py'
```

单个模块也可以直接执行：

```bash
PYTHONWARNINGS=ignore python tools/test_script/test_relation_foreground_policy.py
PYTHONWARNINGS=ignore python tools/test_script/test_bcem.py
```

`_bootstrap.py`只补充本目录、`tools/`及项目根目录的导入路径，不更改当前工作目录或随机数状态。回归测试之间的共用fixture，以及对训练、分析、可视化工具的引用仍可正常解析。

迁移前发现`test_crs_text_pooler.py`有5个测试方法失败（包含7条失败记录），对应原CRS整数padding mask的历史行为与该测试预期不同。本次目录整理保留这些测试及原CRS行为，不通过修改算法或放宽断言掩盖问题。
