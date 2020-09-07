# Wait4GPU

一个在有足够的空闲gpu时执行深度学习脚本的小工具

## 安装

```shell script
python setup.py install
```

## 使用

作为Python脚本的启动器

```shell script
python3 -m Wait4GPU.wait4gpu [--num-required NUM_REQUIRED] [--no-python] training_script
```

在代码中使用

```python
from Wait4GPU.wait4gpu import wait_until_idle
import os
available = wait_until_idle(num_req=2)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(available)
```