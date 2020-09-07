# Wait4GPU

[中文](./readme_cn.md)

A simple utility to execute your deep learning scripts when there are enough idle gpus.

## Install

```shell script
python setup.py install
```

## Usage

As a command wrapper

```shell script
python3 -m Wait4GPU.wait4gpu [--num-required NUM_REQUIRED] [--no-python] training_script
```

Integrate with your code

```python
from Wait4GPU.wait4gpu import wait_until_idle
import os
available = wait_until_idle(num_req=2)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(available)
```