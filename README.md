# DFVG: Deflicker-Awared Foggy Video Generation for UAV Defogging

### Liyao Wang, Zijie Zhang, Yongkang Cao, Yipeng Feng, Haobo Zuo, Changhong Fu

## Abstract
Intelligent unmanned aerial vehicle (UAV) is crucial in applications such as port monitoring and water rescue. However, fog significantly reduces the perceptual capability of UAV in common water scenarios. Existing methods primarily address the limitations of visual perception by training on paired foggy and fog-free data. Nevertheless, it is difficult to obtain high-quality real-world paired UAV foggy and fog-free datasets. Furthermore, the flicker caused by independently generating each frame severely leads to performance degradation of defogging methods for UAVs. Additionally, existing metrics for evaluating fog authenticity are insufficient and overlook foggy video generation. To address these issues, this work proposes a novel deflicker-awared foggy video generation framework, which reduces flicker in fog generation by fully leveraging temporal context information. Specifically, an innovative inter-frame depth-map deflicker is designed to refine distance information within sequences. A new long-term light consistency is introduced to achieve time-invariant atmospheric light values. A new benchmark comprising 84 high-quality video sequences with diverse scenes and challenging conditions captured by the UAV is established to address the scarcity of high-quality paired foggy and fog-free sequence data in water scenes. A new evaluation metric is also presented to assess the authenticity of synthetic fog data. Extensive experiments and visual comparisons with real fog demonstrate the feasibility of our method.
![Workflow of our framework](images/main.png)

This figure shows the workflow of our DFVG.

## UAV-Wafog dataset
The proposed UAV-Wafog dataset, consisting of 84 sequences with a total of 17,859 paired foggy and fog-free frames, is available for download here: [Baidu Netdisk](https://pan.baidu.com/s/1XRXwhJAIYyV0EeN4rhG8JA)(q2me).

## Demonstration running instructions
### 1. Requirements
This code has been tested on an A100.

1.Linux 4.15.0

2.Python 3.11.8

3.CUDA 12.1

4.Pytorch 2.4.1

5.torchvision 0.19.1

Please install related libraries before running this code: 
```bash
pip install -r requirements.txt
```

### 2. Foggy video generation

```bash 
python run.py                                
```
The generated foggy videos  will be saved in the `data/output_haze_dynamicA_beta2.0` directory.

### 3. Weights and Checkpoints

The weights can be downloaded here: [Google Drive](https://drive.google.com/file/d/16jW9iK9KTdNsjfpGHW_GLXh9TYZtt0Km/view?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1-tqu1nQgmUNjbPjgIHehMw)(5cvi).

The checkpoints can be downloaded here: [Google Drive](https://drive.google.com/file/d/1hbp0wepRZixrZro6yb0GMgLghPsi1WQp/view?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1Y4NGYrJYn7Co4SoQqr4qyA)(rhfd).

Notification: We use marigold-v1-0 as marigold weights, you can download it from the [official website](https://github.com/prs-eth/Marigold) or directly download the one we uploaded: [Google Drive](https://drive.google.com/file/d/15UAD5KqPKhgpL5PrIz3gVnQCthipIGtW/view?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1cdFA9hwU5JXSFE_sFiLFiA)(pub4).

### 4. Contact
If you have any questions, please contact me.

Liyao Wang

Email: [2250400@tongji.edu.cn](2250400@tongji.edu.cn)

For more evaluations, please refer to our paper.

## Acknowledgements 

We sincerely thank [Marigold](https://github.com/prs-eth/Marigold), [All-In-One-Deflicker](https://github.com/ChenyangLEI/All-In-One-Deflicker) and [fast_blind_video_consistency](https://github.com/phoenix104104/fast_blind_video_consistency) for their efforts.

