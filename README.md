# BetaSuite OpenVINO 版

基于 [BetaSuite 0.2.4](https://github.com/solarorb93/BetaSuite) 的本地图片和视频批处理工具。可用像素化、模糊或色块遮挡指定区域，也可反向遮挡画面并保留匹配区域。

## 安装

使用 Python 3.10。在项目根目录安装依赖。处理视频还需要把 `ffmpeg` 与 `ffprobe` 加入 PATH。

```bash
python -m pip install -r requirements.txt
```

## 准备模型和素材

在项目根目录创建 `model`、`uncensored_pics` 和 `uncensored_vids`。图片和视频分别放入两个输入目录。把模型权重放在 `model` 下：

| 文件 | 用途 |
| --- | --- |
| `320n.onnx` | 主检测模型。[NudeNet 权重发布页](https://github.com/notAI-tech/NudeNet/releases/tag/v3.4-weights) |
| `PFLD_112_1_opt_sim.onnx` | 可选。98 点人脸定位，用于绘制眼部黑条。请使用同名、输入为 `1×3×112×112`、输出为 `1×196` 的兼容权重。没有该模型时可在配置中关闭 `eyes_bar`。 |
| `detector_v2_default_checkpoint.onnx` | 可选。仅在新版主模型无法运行时用于原版 CPU 兜底。 |

仓库提供 Python 源码。模型权重、输入素材、输出文件和缓存由本地目录管理。

## 运行

从项目根目录执行其中一条命令：

```bash
python BetaSuite-0.2.4/betatv.py --all  # 先处理图片，再处理视频
python BetaSuite-0.2.4/betastare.py    # 仅处理图片
python BetaSuite-0.2.4/betatv.py       # 仅处理视频
```

图片和视频分别输出到 `censored_pics` 和 `censored_vids`，缓存分别写入 `pic_hashes` 和 `vid_hashes`。输出目录和缓存目录会在需要时创建。配置在 [betaconfig.py](BetaSuite-0.2.4/betaconfig.py)。

- `gpu_enabled = 1` 优先使用 OpenVINO **CPU**，出错时切到 ONNX Runtime CPU。设为 `0` 直接使用 ONNX Runtime CPU。两者读取同一套新版 ONNX 权重。
- `reverse_mode = 1` 保留匹配区域并遮挡其余画面；设为 `0` 则遮挡匹配区域。类别、形状、阈值、眼条和水印均可在配置中调整。
- `video_censor_fps` 是每秒检测采样次数。`time_safety` 是每个检测时刻前后窗口的总秒数。正常输出请保留 `debug_mode = 0`。源文件删除默认关闭。

## 此版改动

- 主模型用固定 `320×320` 输入推理，优化预处理和候选框筛选；PFLD 按需加载，只对需要的脸计算眼点。眼点失败时跳过黑条，继续处理。
- 支持反向遮挡、矩形和椭圆区域、自定义边框和水印。修复越界框及同一帧过期框遗漏的问题。
- 检测框和对应眼点写入同一个压缩缓存。仍可读取旧版独立眼点缓存，并在重新处理素材时迁移。
- 视频顺序抽帧并输出 H.264 MP4。优先尝试 Intel QSV，失败后使用 `libx264`。兼容的音轨直接复制，已识别的封装不兼容情况会尝试 AAC。
- 日志按行输出。中断推理时不提交未完成的缓存；中断视频编码时保留已提交的推理缓存并清理临时成品。
