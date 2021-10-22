# Wait4GPU

A simple utility to execute your deep learning scripts when there are enough idle gpus.

~~抢卡的~~ 一个在有足够的空闲gpu时执行深度学习训练的小工具

## Install

```shell script
python setup.py install
```

## Usage

### As a command wrapper

```shell script
python3 -m Wait4GPU [--num-required NUM_REQUIRED] [--no-python] training_script
```

You may refer to `Wait4GPU/__main__.py` for more detailed usage.

### Integrate with your code

```python
from Wait4GPU.wait4gpu import wait_until_idle
import os

available = wait_until_idle(num_req=2)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(available)
```