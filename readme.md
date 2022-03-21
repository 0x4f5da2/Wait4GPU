# Wait4GPU

A simple utility to execute your deep learning scripts when there are enough idle gpus.

一个在有足够的空闲gpu时执行深度学习训练的小工具

## Install

```shell script
python setup.py install
```

## Usage

### As a command wrapper

```shell script
python3 -m Wait4GPU [--num-required NUM_REQUIRED] [--no-python] training_script
```
**Example:**
```shell script
# train `atss_r50_fpn_1x_coco.py` with 1 gpu using mmdetection
python3 -m Wait4GPU tools/train.py configs/atss/atss_r50_fpn_1x_coco.py

# train `atss_r50_fpn_1x_coco.py` with 2 gpus using mmdetection
python3 -m Wait4GPU --num-required 2 --no-python ./tools/dist_train.sh configs/atss/atss_r50_fpn_1x_coco.py 2
```

You may refer to `Wait4GPU/__main__.py` for more detailed usage.

### Integrate with your code

```python
from Wait4GPU.wait4gpu import wait_until_idle
import os

available = wait_until_idle(num_req=2)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(available)
```
