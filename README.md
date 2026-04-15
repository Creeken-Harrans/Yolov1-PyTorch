# YOLOv1-PyTorch

一个基于 PyTorch 的 YOLOv1 目标检测实现，面向 Pascal VOC 数据集，包含：

- 训练脚本
- 随机样例可视化推理脚本
- VOC07 test 上的 mAP 评估逻辑
- 数据集读取与 YOLOv1 target 构造
- 损失函数实现
- 基于 `ResNet34` 的检测模型

这个仓库不是逐层复刻论文版 `Darknet`，而是一个更现代、也更容易直接跑起来的 YOLOv1 变体实现。当前代码默认使用 `ResNet34 pretrained on ImageNet` 作为 backbone，并在其后接 YOLOv1 风格的检测头与损失。

## 1. 项目概览

### 1.1 这个仓库做了什么

仓库实现了一个完整的 YOLOv1 训练/推理闭环：

1. 从 VOC XML 标注中读取目标框与类别。
2. 将标注编码为 `S x S x (5B + C)` 的 YOLO target。
3. 使用 `ResNet34 + detection head` 进行训练。
4. 在推理阶段将预测从网格输出解码为真实框。
5. 经过置信度过滤和 NMS 后进行可视化。
6. 在 VOC 评估风格下计算每个类别的 AP 和总体 mAP。

### 1.2 当前默认实验设置

默认配置写在 [config/voc.yaml](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/config/voc.yaml)：

- 输入尺寸：`448`
- 网格大小：`S = 7`
- 每个网格预测框数：`B = 2`
- 类别数：`C = 20`
- 训练集：`VOC2007 trainval + VOC2012 trainval`
- 测试集：`VOC2007 test`
- 主干网络：`ResNet34`
- 检测头：默认使用卷积预测头
- 框分支：默认使用 `sigmoid`

### 1.3 与原始 YOLOv1 论文的主要差异

当前实现和论文版 YOLOv1 存在这些关键区别：

- 主干网络不是论文中的 Darknet，而是 `torchvision.models.resnet34`。
- 在 YOLO detection head 中加入了 `BatchNorm2d`。
- 默认支持两种预测头：
  - `use_conv: True` 时使用 `1x1 conv` 输出预测。
  - `use_conv: False` 时使用全连接层，更接近论文表达。
- 默认开启 `use_sigmoid: True`，对框相关预测做 sigmoid，代码注释中说明这会让训练更稳定、收敛更快。
- 学习率和调度策略也不是完全按论文原始设置照搬，而是根据作者实验做了调整。

如果你追求“更像论文”的结构，可以在配置里尝试：

- `use_conv: False`
- `use_sigmoid: False`

但这不代表一定会有更好的训练稳定性或最终指标。

## 2. 仓库结构

仓库核心文件如下：

```text
Yolov1-PyTorch
├── config/
│   └── voc.yaml
├── dataset/
│   └── voc.py
├── loss/
│   └── yolov1_loss.py
├── models/
│   └── yolo.py
├── tools/
│   ├── train.py
│   └── infer.py
├── utils/
│   └── visualization_utils.py
├── data/
│   ├── VOC2007-test/
│   ├── VOC2012/
│   ├── VOCdevkit/
│   └── *.tar
└── requirements.txt
```

各目录职责：

- [tools/train.py](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/tools/train.py)：训练入口。
- [tools/infer.py](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/tools/infer.py)：样例推理与 mAP 评估入口。
- [dataset/voc.py](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/dataset/voc.py)：VOC 数据读取、增强、归一化、target 构建。
- [models/yolo.py](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/models/yolo.py)：YOLOv1 模型定义。
- [loss/yolov1_loss.py](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/loss/yolov1_loss.py)：YOLOv1 损失函数。
- [utils/visualization_utils.py](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/utils/visualization_utils.py)：预测框、网格图、类别热区可视化。
- [config/voc.yaml](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/config/voc.yaml)：默认训练与推理配置。

## 3. 环境要求

### 3.1 Python 与依赖

建议使用：

- Python `3.10` 或相近版本
- PyTorch `2.3.1`
- torchvision `0.18.1`

仓库当前依赖见 [requirements.txt](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/requirements.txt)：

```txt
einops==0.8.0
numpy==2.0.1
opencv_python==4.10.0.84
Pillow==10.4.0
PyYAML==6.0.1
torch==2.3.1
torchvision==0.18.1
tqdm==4.66.4
albumentations==1.4.13
```

### 3.2 创建环境

```bash
cd /home/Creeken/Paper/yolo-v5/Yolov1-PyTorch
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

如果你使用 conda，也可以：

```bash
conda create -n yolov1 python=3.10 -y
conda activate yolov1
pip install -r requirements.txt
```

### 3.3 设备支持

训练脚本中设备选择逻辑如下：

- 优先使用 `CUDA`
- 若没有 CUDA 且 `torch.backends.mps.is_available()`，则切换到 `MPS`
- 否则使用 `CPU`

也就是说：

- NVIDIA GPU 用户会自动走 CUDA。
- Apple Silicon 用户若启用 MPS，会自动走 MPS。
- 其他环境默认 CPU。

## 4. 数据集准备

### 4.1 当前代码默认使用的数据集

默认配置使用：

- `VOC2007 trainval`
- `VOC2012 trainval`
- `VOC2007 test`

也就是经典 YOLOv1 在 VOC 上的常见训练/测试组合。

### 4.2 推荐的数据目录结构

最直接的方式，是把数据整理成下面这样：

```text
data/
├── VOC2007/
│   ├── Annotations/
│   ├── ImageSets/
│   └── JPEGImages/
├── VOC2007-test/
│   ├── Annotations/
│   ├── ImageSets/
│   └── JPEGImages/
└── VOC2012/
    ├── Annotations/
    ├── ImageSets/
    └── JPEGImages/
```

### 4.3 当前仓库对路径的兼容逻辑

训练脚本 [tools/train.py](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/tools/train.py) 里有一个非常实用的细节：

- 它先尝试读取配置中的路径，如 `data/VOC2007`
- 如果这个路径不存在，会自动 fallback 到：
  - `data/VOCdevkit/VOC2007`
  - `data/VOCdevkit/VOC2012`
  - 具体规则是 `data/VOCdevkit/<basename(path)>`

这意味着当前仓库中即使 `VOC2007` 被放在 `data/VOCdevkit/VOC2007` 下，也仍然可以工作。

但需要注意：

- `VOC2007-test` 默认不会走 `VOCdevkit` fallback，因为 VOC test 目录当前配置是 `data/VOC2007-test`。
- 因此建议你明确保留 `data/VOC2007-test` 这个目录。

### 4.4 当前工作树里已经看到的数据目录

仓库当前 `data/` 下已经存在这些目录：

- `data/VOC2007-test`
- `data/VOC2012`
- `data/VOCdevkit/VOC2007`

这与训练脚本的路径兼容逻辑是匹配的。

### 4.5 VOC 数据下载建议

如果你要重新准备数据，可从 Pascal VOC 官方站点下载：

- VOC2007 train/val
- VOC2007 test
- VOC2012 train/val

解压后放到上面的目录结构中即可。

### 4.6 标注格式要求

当前实现依赖 VOC 原生 XML 标注，读取逻辑在 [dataset/voc.py](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/dataset/voc.py) 中，代码会读取：

- `<size>/<width>`
- `<size>/<height>`
- `<object>/<name>`
- `<object>/<difficult>`
- `<object>/<bndbox>/<xmin,ymin,xmax,ymax>`

如果 XML 缺字段，代码会直接抛错。

### 4.7 `difficult` 标签的处理方式

这个点很重要：

- 训练阶段：`difficult == 1` 的目标会被忽略，不参与训练 target 构建。
- 测试阶段：仍会读取 `difficult` 目标，但评估时会在 AP 统计中按 VOC 逻辑忽略其对 recall 的影响。

这是一个合理且接近 VOC 习惯的处理方式。

## 5. 快速开始

在仓库根目录执行：

```bash
cd /home/Creeken/Paper/yolo-v5/Yolov1-PyTorch
pip install -r requirements.txt
```

### 5.1 开始训练

```bash
python tools/train.py --config config/voc.yaml
```

或者：

```bash
python -m tools.train --config config/voc.yaml
```

### 5.2 生成随机样例推理结果

```bash
python tools/infer.py --config config/voc.yaml
```

默认会从测试集里随机抽样 5 张图，保存预测图和类别网格图。

### 5.3 执行 mAP 评估

```bash
python tools/infer.py --config config/voc.yaml --evaluate True
```

这会在执行样例推理的同时额外做一次全量测试集评估，并在终端打印：

- 每个类别的 AP
- 最终 mAP

## 6. 配置文件详解

默认配置文件是 [config/voc.yaml](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/config/voc.yaml)。

内容如下：

```yaml
dataset_params:
  train_im_sets: ['data/VOC2007', 'data/VOC2012']
  test_im_sets: ['data/VOC2007-test']
  num_classes : 20
  im_size : 448

model_params:
  im_channels : 3
  backbone_channels : 512
  conv_spatial_size : 7
  yolo_conv_channels : 1024
  leaky_relu_slope : 0.1
  fc_dim : 4096
  fc_dropout : 0.5
  S : 7
  B : 2
  use_sigmoid : True
  use_conv : True

train_params:
  task_name: 'voc'
  seed: 1111
  acc_steps: 1
  log_steps: 100
  num_epochs: 135
  batch_size: 64
  lr_steps: [50, 75, 100, 125]
  lr: 0.001
  infer_conf_threshold : 0.2
  eval_conf_threshold : 0.001
  nms_threshold : 0.5
  ckpt_name: 'yolo_voc2007.pth'
```

下面逐项解释。

### 6.1 `dataset_params`

#### `train_im_sets`

训练集目录列表。当前是：

- `data/VOC2007`
- `data/VOC2012`

脚本会把两者拼接在一起读取。

#### `test_im_sets`

测试集目录列表。当前是：

- `data/VOC2007-test`

#### `num_classes`

类别数。VOC 为 `20`。

#### `im_size`

输入图片 resize 后的边长。当前为 `448`，与 YOLOv1 论文设定一致。

### 6.2 `model_params`

#### `im_channels`

输入通道数，RGB 图像为 `3`。

#### `backbone_channels`

backbone 输出通道数。当前 `ResNet34 layer4` 输出是 `512`，因此配置为 `512`。

如果你更换 backbone，这个值通常需要一起改。

#### `conv_spatial_size`

仅在 `use_conv: False`、使用全连接输出头时才真正重要。它表示进入全连接层前 feature map 的空间尺寸。

当前配置是 `7`。

#### `yolo_conv_channels`

YOLO 检测头中的中间通道数。默认 `1024`。

#### `leaky_relu_slope`

LeakyReLU 负半轴斜率，默认 `0.1`。

#### `fc_dim`

全连接预测头的隐藏层维度，仅 `use_conv: False` 时使用。

#### `fc_dropout`

全连接预测头中的 dropout 概率，仅 `use_conv: False` 时使用。

#### `S`

YOLO 网格数。当前为 `7`，即 `7x7` 网格。

#### `B`

每个网格预测框数。当前为 `2`。

#### `use_sigmoid`

是否对框相关预测使用 sigmoid。当前默认 `True`。

这会影响：

- 损失计算中的 box 相关值
- 推理解码时的 box 相关值

#### `use_conv`

是否使用卷积输出头。当前默认 `True`。

- `True`：输出头为 `1x1 conv`
- `False`：输出头为 `Flatten + Linear + Dropout + Linear`

### 6.3 `train_params`

#### `task_name`

训练输出目录。当前是 `voc`。

checkpoint 会保存在：

```text
voc/yolo_voc2007.pth
```

#### `seed`

随机种子。用于：

- `torch`
- `numpy`
- `random`
- 若是 CUDA，还会设置 `torch.cuda.manual_seed_all`

#### `acc_steps`

梯度累积步数。适合显存不足时使用。

比如：

- 原本你想要 `batch_size=64`
- 但显存只够 `batch_size=16`
- 可以设置：
  - `batch_size=16`
  - `acc_steps=4`

这样等效每 4 个 mini-batch 更新一次参数。

#### `log_steps`

每多少 step 打印一次 loss。

#### `num_epochs`

训练轮数，当前 `135`。

#### `batch_size`

训练 batch size，当前 `64`。

#### `lr_steps`

学习率衰减 epoch，当前：

- `50`
- `75`
- `100`
- `125`

#### `lr`

初始学习率，当前 `1e-3`。

#### `infer_conf_threshold`

样例推理可视化时的置信度阈值，当前 `0.2`。

#### `eval_conf_threshold`

评估 mAP 时的置信度阈值，当前 `0.001`。

评估阈值更低是合理的，因为 mAP 计算通常希望尽量保留候选框。

#### `nms_threshold`

NMS IoU 阈值，当前 `0.5`。

#### `ckpt_name`

checkpoint 文件名，当前 `yolo_voc2007.pth`。

## 7. 训练流程详解

训练脚本位于 [tools/train.py](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/tools/train.py)。

### 7.1 训练命令

```bash
python tools/train.py --config config/voc.yaml
```

### 7.2 训练阶段做了什么

训练主流程大致如下：

1. 读取 YAML 配置。
2. 解析数据集路径，并做相对路径转绝对路径。
3. 对训练/测试数据路径做 fallback 解析。
4. 根据设备自动调整 batch size。
5. 构建 `VOCDataset` 和 `DataLoader`。
6. 构建 `YOLOV1` 模型。
7. 如 checkpoint 已存在，则自动加载继续训练。
8. 创建 `SGD + MultiStepLR`。
9. 逐 epoch 训练，计算 YOLOv1 loss。
10. 每个 epoch 保存一次 checkpoint。

### 7.3 自动恢复训练

如果下面这个文件存在：

```text
<task_name>/<ckpt_name>
```

例如：

```text
voc/yolo_voc2007.pth
```

训练脚本会自动加载它，而不是从头开始。

所以如果你希望彻底重新训练，需要：

- 删除旧 checkpoint
- 或修改 `task_name`
- 或修改 `ckpt_name`

### 7.4 CUDA 显存不足时的自动 batch 调整

训练脚本内置了一段很实用的逻辑：

- 当设备是 `CUDA` 时，会读取 GPU 总显存大小。
- 如果显存较小，会自动把 `batch_size` 从配置值下调。
- 同时把 `acc_steps` 成倍增大，尽量维持等效批大小。

规则如下：

- 显存 `<= 8 GiB` 且原始 `batch_size > 2`，改为 `2`
- 显存 `<= 12 GiB` 且原始 `batch_size > 4`，改为 `4`
- 显存 `<= 16 GiB` 且原始 `batch_size > 8`，改为 `8`

这段逻辑只对 `CUDA` 生效：

- 不对 `MPS` 生效
- 不对 `CPU` 生效

### 7.5 优化器与调度器

当前训练使用：

- 优化器：`torch.optim.SGD`
- momentum：`0.9`
- weight_decay：`5e-4`
- scheduler：`MultiStepLR(gamma=0.5)`

这是非常典型的检测任务设置。

### 7.6 日志输出

训练过程中会打印：

- 当前配置
- 是否切换到 MPS
- 数据集类别映射
- 数据总量
- loss 日志
- 每个 epoch 结束提示

如果 loss 出现 `NaN`，脚本会直接退出。

## 8. 推理与可视化

推理脚本位于 [tools/infer.py](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/tools/infer.py)。

### 8.1 推理命令

```bash
python tools/infer.py --config config/voc.yaml
```

### 8.2 推理阶段做了什么

随机样例推理流程如下：

1. 加载配置。
2. 构建测试集 `VOCDataset(split='test')`。
3. 加载训练好的 checkpoint。
4. 从测试集里随机抽 `5` 张图片。
5. 前向得到 YOLO 网格输出。
6. 解码为 `x1, y1, x2, y2`。
7. 进行置信度过滤。
8. 按类别执行 NMS。
9. 生成两类可视化结果：
   - 预测框结果图
   - 网格类别分布图

### 8.3 推理输出保存在哪里

这里要特别说明一下当前代码的真实行为：

推理结果不是保存在 `task_name/` 下，而是直接保存在仓库根目录的：

```text
samples/preds/
samples/grid_cls/
```

具体文件示例：

```text
samples/preds/0_pred.jpeg
samples/grid_cls/0_grid_map.jpeg
```

这是当前实现的真实路径，不是配置控制的。

### 8.4 两种输出图分别代表什么

#### `samples/preds/*.jpeg`

这是标准检测可视化图：

- 原图
- 预测框
- 类别标签
- 置信度分数

#### `samples/grid_cls/*.jpeg`

这是辅助理解 YOLO 网格预测的图：

- 将图片切成 `7x7` 网格
- 每个格子用颜色表示预测类别
- 叠加类别名字

它更偏“解释模型行为”，而不是常规检测可视化。

## 9. mAP 评估

### 9.1 评估命令

```bash
python tools/infer.py --config config/voc.yaml --evaluate True
```

### 9.2 评估流程

评估函数 `evaluate_map(args)` 会：

1. 遍历整个测试集。
2. 将预测结果按类别整理。
3. 将 GT 按类别整理。
4. 记录 `difficult` 标志。
5. 对每个类别计算 AP。
6. 输出总体 mAP。

### 9.3 当前 AP 的计算方式

`compute_map` 支持两种 AP 计算方式：

- `area`
- `interp`

当前实际调用使用的是：

```python
compute_map(preds, gts, method='area', difficult=difficults)
```

也就是面积积分法。

### 9.4 评估输出形式

终端会打印：

- `Class Wise Average Precisions`
- 每个类别的 AP
- `Mean Average Precision`

### 9.5 为什么评估阈值比推理阈值低

配置中：

- `infer_conf_threshold = 0.2`
- `eval_conf_threshold = 0.001`

这是合理的：

- 可视化时希望图上更干净，所以阈值高。
- 评估时希望尽可能保留候选框，然后让 PR 曲线和 AP 统计更完整，所以阈值低。

## 10. 数据集实现细节

数据集类位于 [dataset/voc.py](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/dataset/voc.py)。

### 10.1 类别列表

当前 VOC 20 类如下：

```text
person
bird
cat
cow
dog
horse
sheep
aeroplane
bicycle
boat
bus
car
motorbike
train
bottle
chair
diningtable
pottedplant
sofa
tvmonitor
```

代码中会先 `sorted(classes)`，因此内部类别索引顺序是字典序后的结果，而不是上面手写列表的原始顺序。

### 10.2 数据增强

训练集增强包含：

- `HorizontalFlip(p=0.5)`
- `Affine(scale=(0.8, 1.2), translate_percent=(-0.2, 0.2), always_apply=True)`
- `ColorJitter(...)`
- `Resize(448, 448)`

测试集只有：

- `Resize(448, 448)`

### 10.3 图像归一化

图像先缩放到 `[0, 1]`，再按 ImageNet 均值方差归一化：

- mean: `[0.485, 0.456, 0.406]`
- std: `[0.229, 0.224, 0.225]`

这与使用 ImageNet 预训练的 `ResNet34` 是对齐的。

### 10.4 YOLO target 的构造方式

每张图最终会被编码成：

```text
S x S x (5B + C)
```

当前默认就是：

```text
7 x 7 x 30
```

每个网格单元包含：

- `x_offset`
- `y_offset`
- `sqrt(w)`
- `sqrt(h)`
- `conf`
- `class one-hot`

并且会对 `B` 个框重复写入同一个 GT 框参数。

### 10.5 返回值格式

数据集 `__getitem__` 返回：

```python
im_tensor, targets, image_path
```

其中 `targets` 包含：

- `bboxes`: 归一化到 `[0, 1]` 的 `x1y1x2y2`
- `labels`: 类别索引
- `yolo_targets`: `S x S x (5B + C)` 的训练 target
- `difficult`: VOC difficult 标志

## 11. 模型实现细节

模型定义在 [models/yolo.py](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/models/yolo.py)。

### 11.1 Backbone

当前 backbone 是：

```python
torchvision.models.resnet34(
    weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1
)
```

使用的是 ImageNet 预训练权重。

保留的层包括：

- `conv1`
- `bn1`
- `relu`
- `maxpool`
- `layer1`
- `layer2`
- `layer3`
- `layer4`

### 11.2 Detection Head

backbone 后面接了 4 个卷积块：

- Conv
- BatchNorm
- LeakyReLU

其中第二个卷积使用 `stride=2`，把空间尺寸进一步压到 `7x7`。

### 11.3 输出头

有两种模式：

#### 模式一：卷积输出头

当 `use_conv=True`：

- 使用一个 `1x1 conv`
- 输出通道数为 `5B + C`
- 再 `permute` 成 `(N, S, S, 5B + C)`

#### 模式二：全连接输出头

当 `use_conv=False`：

- Flatten
- Linear
- LeakyReLU
- Dropout
- Linear

更接近经典 YOLOv1 的 FC 形式。

## 12. 损失函数实现细节

损失定义在 [loss/yolov1_loss.py](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/loss/yolov1_loss.py)。

### 12.1 损失组成

总损失由以下部分组成：

- 分类损失
- 负责框的 objectness 损失
- 负责框的坐标损失
- 非负责框与无目标格子的 no-object 损失

### 12.2 超参数

当前损失内部使用：

- `lambda_coord = 5`
- `lambda_noobj = 0.5`

这与经典 YOLOv1 设置一致。

### 12.3 负责框的选择方式

对于每个有目标的网格：

1. 先把 `B` 个预测框都解码成 `x1y1x2y2`
2. 再和 target 框计算 IoU
3. 选择 IoU 最大的那个框作为 responsible predictor

这个逻辑是 YOLOv1 的核心之一。

### 12.4 宽高的预测方式

损失和 target 都在学习：

- `sqrt(w)`
- `sqrt(h)`

推理时会通过平方重新恢复真实宽高。

## 13. 推理解码细节

预测解码主要在 [tools/infer.py](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/tools/infer.py) 的 `convert_yolo_pred_x1y1x2y2` 中完成。

### 13.1 解码过程

对于每个网格、每个框：

1. 得到网格内中心点偏移 `x_offset, y_offset`
2. 结合网格左上角位置恢复全图归一化中心点
3. 将预测的 `sqrt(w), sqrt(h)` 平方恢复成宽高
4. 转换成 `x1, y1, x2, y2`

### 13.2 类别分数

当前实现里：

- 先在类别维度取最大值得到 `class_score`
- 再把 `box_confidence * class_score` 作为最终框分数

### 13.3 NMS 方式

NMS 使用：

```python
torchvision.ops.nms
```

并且是“按类别分别做 NMS”，不会跨类别互相抑制。

## 14. 输出文件与目录

### 14.1 训练输出

训练会在 `task_name` 指定目录下生成 checkpoint，例如：

```text
voc/yolo_voc2007.pth
```

### 14.2 推理输出

推理可视化结果固定写到：

```text
samples/preds/
samples/grid_cls/
```

### 14.3 什么时候会自动创建目录

代码会在运行时自动创建：

- `voc/` 或你设置的 `task_name`
- `samples/`
- `samples/preds/`
- `samples/grid_cls/`

## 15. 如何迁移到自己的数据集

如果你要用自己的数据集，而不是 VOC，建议按下面步骤修改。

### 15.1 修改配置

先改 [config/voc.yaml](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/config/voc.yaml) 或新建一个配置文件：

- `train_im_sets`
- `test_im_sets`
- `num_classes`
- 可能还需要：
  - `im_size`
  - `task_name`
  - `ckpt_name`

### 15.2 修改数据加载逻辑

重点改 [dataset/voc.py](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/dataset/voc.py)：

- `load_images_and_anns`
- 类别列表 `classes`
- 如果标注格式不是 VOC XML，需要重写解析逻辑

### 15.3 保持返回接口不变

无论你改成什么数据集，最好仍然让数据集类返回：

```python
im_tensor, targets, image_path
```

且 `targets` 至少保留：

- `yolo_targets`
- `bboxes`
- `labels`
- `difficult`

如果没有 `difficult` 概念，也建议返回全 0，方便复用现有评估逻辑。

### 15.4 修改类别数量

你必须保证这几处一致：

- 配置里的 `num_classes`
- 数据集里的类别列表长度
- 模型输出维度中的 `C`

否则训练或推理阶段会直接 shape 对不上。

## 16. 常见问题

### 16.1 为什么训练一开始就下载权重

因为模型默认使用 `torchvision` 的 `ResNet34_Weights.IMAGENET1K_V1`。如果本地没有缓存，第一次运行会自动下载 ImageNet 预训练权重。

### 16.2 为什么 batch size 变小了

因为训练脚本会在 CUDA 环境下按显存自动缩小 batch size，并同步增大 `acc_steps`，这是代码里的显存保护逻辑。

### 16.3 为什么评估时候很慢

因为评估会：

- 遍历整个测试集
- 每张图都做解码
- 按类别做 NMS
- 汇总所有类别的 AP

这本来就比单张可视化推理慢很多。

### 16.4 为什么输出路径不是 `task_name/samples`

因为当前代码把可视化结果直接写死到了仓库根目录下的 `samples/`，而不是 `task_name/`。这是当前实现细节，不是 README 约定。

### 16.5 命令行里 `--infer_samples False` 是否可靠

当前脚本参数使用的是：

```python
type=bool
```

在 Python `argparse` 中，这种写法对字符串布尔值并不稳妥。也就是说：

- `--evaluate True` 通常能达到预期
- 但 `--infer_samples False` 这种形式不一定会像你直觉那样可靠地变成 `False`

因此当前最稳妥的用法是：

- 只做默认样例推理：`python tools/infer.py --config config/voc.yaml`
- 做样例推理并顺带评估：`python tools/infer.py --config config/voc.yaml --evaluate True`

如果你后续要长期维护这个项目，建议把这两个参数改成 `action='store_true'` / `action='store_false'` 风格。

### 16.6 用自己的 backbone 需要改哪里

至少需要同时检查这些点：

- [models/yolo.py](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/models/yolo.py) 中的 `self.features`
- 配置中的 `backbone_channels`
- 配置中的 `conv_spatial_size`
- 检测头输入输出 shape 是否仍与 `S x S` 一致

## 17. 示例工作流

一个典型使用流程如下：

### 17.1 第一步：安装依赖

```bash
cd /home/Creeken/Paper/yolo-v5/Yolov1-PyTorch
pip install -r requirements.txt
```

### 17.2 第二步：确认数据目录

确保至少有：

```text
data/VOC2007-test
data/VOC2012
data/VOCdevkit/VOC2007
```

或者你自己整理成：

```text
data/VOC2007
data/VOC2007-test
data/VOC2012
```

### 17.3 第三步：开始训练

```bash
python tools/train.py --config config/voc.yaml
```

### 17.4 第四步：查看 checkpoint

训练后检查：

```text
voc/yolo_voc2007.pth
```

### 17.5 第五步：跑随机样例可视化

```bash
python tools/infer.py --config config/voc.yaml
```

查看：

```text
samples/preds/
samples/grid_cls/
```

### 17.6 第六步：做测试集评估

```bash
python tools/infer.py --config config/voc.yaml --evaluate True
```

## 18. 代码阅读建议

如果你准备深入理解这个仓库，推荐按下面顺序读代码：

1. [config/voc.yaml](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/config/voc.yaml)
2. [dataset/voc.py](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/dataset/voc.py)
3. [models/yolo.py](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/models/yolo.py)
4. [loss/yolov1_loss.py](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/loss/yolov1_loss.py)
5. [tools/train.py](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/tools/train.py)
6. [tools/infer.py](/home/Creeken/Paper/yolo-v5/Yolov1-PyTorch/tools/infer.py)

这是最接近实际运行路径的阅读顺序。

## 19. 可视化示例

原仓库历史 README 中包含了示例图与视频链接，这些资源如果你需要保留展示，可以继续使用：

- YOLOv1 Explanation and Implementation 视频：
  - `https://youtu.be/TPD9AfY7AHo`

示例图片链接也可以继续补回到这里，作为训练效果展示素材。

## 20. 引用

如果这个仓库对你的工作有帮助，可引用原始 YOLOv1 论文：

```bibtex
@article{DBLP:journals/corr/RedmonDGF15,
  author       = {Joseph Redmon and
                  Santosh Kumar Divvala and
                  Ross B. Girshick and
                  Ali Farhadi},
  title        = {You Only Look Once: Unified, Real-Time Object Detection},
  journal      = {CoRR},
  volume       = {abs/1506.02640},
  year         = {2015},
  url          = {http://arxiv.org/abs/1506.02640},
  eprinttype   = {arXiv},
  eprint       = {1506.02640}
}
```

## 21. 总结

如果你想要一个：

- 可以直接训练 VOC 的 YOLOv1 项目
- 结构清晰、代码不绕的教学型实现
- 同时具备训练、可视化和 mAP 评估能力的基线仓库

这个项目是很合适的起点。

它的核心特点可以概括为：

- YOLOv1 思路保留得比较完整
- 工程上更偏 PyTorch 现代写法
- 通过 `ResNet34 + conv head + sigmoid boxes` 提高了可训练性
- 适合作为学习 YOLOv1、二次改造 backbone、迁移自定义数据集的基础模板
