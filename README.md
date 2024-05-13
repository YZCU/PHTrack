# PHTrack (Under Review)   ♪ Hopefully something good will happen for all of us ♪ 

The official implementation for "**PHTrack**"

- Authors: 
[Yuzeng Chen](https://yzcu.github.io/), 
[Yuqi Tang](https://faculty.csu.edu.cn/yqtang/zh_CN/zdylm/66781/list/index.htm),
[Xin Su*](http://jszy.whu.edu.cn/xinsu_rs/zh_CN/index.htm),
[Jie Li*](http://jli89.users.sgg.whu.edu.cn/),
[Yi Xiao](https://xy-boy.github.io/),
[Jiang He](https://jianghe96.github.io/),
[Qiangqiang Yuan](http://qqyuan.users.sgg.whu.edu.cn/)
- Affiliations: Wuhan University ([School of Geodesy and Geomatics](http://main.sgg.whu.edu.cn/), [School of Remote Sensing and Information Engineering](https://rsgis.whu.edu.cn/), Central South University ([School of Geosciences and Info-Physics](https://gip.csu.edu.cn/index.htm))

 
##  Install
```
git clone https://github.com/YZCU/PHTrack.git
```

## Environment
 > * CUDA 11.8
 > * Python 3.9.18
 > * PyTorch 2.0.0
 > * Torchvision 0.15.0
 > * numpy 1.25.0 
 - **Note:** Please check the `requirement.txt` for environmental details.

### Quick Start
- **Step I.**  Download the RGB/Hyperspectral training/test datasets:
[GOT-10K](http://got-10k.aitestunion.com/downloads), 
[DET](http://image-net.org/challenges/LSVRC/2017/), 
[LaSOT](https://cis.temple.edu/lasot/),
[COCO](http://cocodataset.org),
[YOUTUBEBB (code: v7s6)](https://pan.baidu.com/s/1gQKmi7o7HCw954JriLXYvg),
[VID](http://image-net.org/challenges/LSVRC/2017/),
[HOTC](https://www.hsitracking.com/hot2022/),
and put them in the path of `train_dataset/dataset_name/`.
- **Step II.**  For the train session, please download the [foundation model](https:) to `pretrained_models/`.
- **Step III.**  Run the `setup.py` to set the local path. You can also run `python setup.py` directly.
- **Step IV.**  To train a model, switch directory to `tools/` and run `train.py` with the desired configs. --`cd tools/, python train.py`
- **Step V.**  For the test session, the well-trained [PHTrack model](https://) will be released. Please put it to the path of `tools/snapshot/`.
- **Step VI.**  Switch directory to `tools/` and run the `tools/test.py`. --`cd tools/, python test.py`
- **Step VII.**  Results are saved in the path of `tools/results/`.
- **Step VIII.**   For the evaluation session, please download the evaluation benchmark [Toolkit](http://cvlab.hanyang.ac.kr/tracker_benchmark/) and [vlfeat](http://www.vlfeat.org/index.html) for more precise evaluations.
- **Step IX.**  Download the file of the `tracking results` and put it into the path of `\tracker_benchmark_v1.0\results\results_OPE_PHTrack`.
- **Step X.**  Evaluation of the PHTrack tracker. Run `\tracker_benchmark_v1.0\perfPlot.m`

## Results
- Comparison with SOTA hyperspectral trackers
- *Characteristics and results of hyperspectral trackers*
 ![image](/fig/5.jpg)
- *Hyperspectral videos: (a) Precision plot. (b) Success plot*
 ![image](/fig/6.jpg)
- *Accuracy-speed comparisons. (a) Pre vs. FPS. (b) Suc vs. FPS*
 ![image](/fig/7.jpg)
 
- Comparison with SOTA RGB trackers (overall)
 ![image](/fig/0.jpg)
 
- Comparison with hand-crafted feature-based RGB trackers
- *RGB videos: (a) Precision plot. (b) Success plot*
 ![image](/fig/1.jpg)
- *False color videos: (a) Precision plot. (b) Success plot*
 ![image](/fig/2.jpg)
 
- Comparison with deep feature-based RGB trackers
- *RGB videos: (a) Precision plot. (b) Success plot*
 ![image](/fig/3.jpg)
- *False color videos: (a) Precision plot. (b) Success plot*
 ![image](/fig/4.jpg)
 
- Attribute-based Evaluations
- *Pre scores for each attribute and overall*
 ![image](/fig/8.jpg)
- *Suc scores for each attribute and overall*
 ![image](/fig/9.jpg)

- *Precision plots for each attribute and overall*
 ![image](/fig/10.jpg)
- *Success plots for each attribute and overall*
 ![image](/fig/11.jpg)

- Qualitative results
 ![image](/fig/12.jpg)
 
:heart:For more comprehensive results, please review the upcoming manuscript.

## Contact
If you have any questions or suggestions, feel free to contact me.  
Email: yuzeng_chen@whu.edu.cn
 
## Citation
If you find our work helpful in your research, kindly consider citing it. We appreciate your support.


```
@ARTICLE{,
  author={},
  journal={}, 
  title={}, 
  year={},
  volume={},
  number={},
  pages={},
  keywords={},
  doi={}
```

