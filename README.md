# PHTrack: Prompting for Hyperspectral Video Tracking
The official implementation for "**PHTrack: Prompting for Hyperspectral Video Tracking**"

# Abstract
Hyperspectral (HS) video captures continuous spectral information of objects, enhancing material identification in tracking tasks. It is expected to overcome the inherent limitations of RGB and multi-modal tracking, such as finite spectral cues and cumbersome modality alignment. However, HS tracking faces challenges like data anxiety, band gaps, and huge volumes. In this study, inspired by prompt learning in language models, we propose the Prompting for Hyperspectral Video Tracking (PHTrack) framework. PHTrack learns prompts to adapt foundation models, mitigating data anxiety and enhancing performance and efficiency. First, the modality prompter (MOP) is proposed to capture rich spectral cues and bridge band gaps for improved model adaptation and knowledge enhancement. Additionally, the distillation prompter (DIP) is developed to refine cross-modal features. PHTrack follows feature-level fusion, effectively managing huge volumes compared to traditional decision-level fusion fashions. Extensive experiments validate the proposed framework, offering valuable insights for future research. The code and data will be available at https://github.com/YZCU/PHTrack.
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
## Usage
- Download the RGB/Hyperspectral training/test datasets [GOT-10K](http://got-10k.aitestunion.com/downloads), [DET](http://image-net.org/challenges/LSVRC/2017/), [LaSOT](https://cis.temple.edu/lasot/), [COCO](http://cocodataset.org), [YOUTUBEBB](https://pan.baidu.com/s/1gQKmi7o7HCw954JriLXYvg) (code: v7s6), [VID](http://image-net.org/challenges/LSVRC/2017/), [HOTC](https://www.hsitracking.com/hot2020/).
- Download the pretrained model: [pretrained model](https://pan.baidu.com/s/19pmFUAA0Bvj0s0GP_4xccA) (code: abcd) to `pretrained_models/`.
- Please train the PHTrack based on the [foundation model](https://pan.baidu.com/s/19pmFUAA0Bvj0s0GP_4xccA) (code: abcd).
- We will release the well-trained model of [PHTrack](https://pan.baidu.com/s/1TpODrs5IbnfXKyNC1N0uOAA) (code: abcd).
- The generated model will be saved to the path of `tools/snapshot`.
- Please test the model. The results will be saved in the path of `tools/results/OTB100`.
- For evaluation, please download the evaluation benchmark [Toolkit](http://cvlab.hanyang.ac.kr/tracker_benchmark/) and [vlfeat](http://www.vlfeat.org/index.html) for more precision performance evaluation.
- Refer to [HOTC](https://www.hsitracking.com/hot2022/) for evaluation.
- Evaluation of the PHTrack tracker. Run `\tracker_benchmark_v1.0\perfPlot.m`
- Relevant tracking results are provided in `PHTrack\tracking_results\hotc20test`. More evaluation results are provided in a `PHTrack\tracking_results`.
--------------------------------------------------------------------------------------
:running:Keep updating:running:: More detailed tracking results for PHTrack have been released.
- [hotc20test](https://www.hsitracking.com/) ([results](https://github.com/YZCU/PHTrack/tree/master/tracking_results))
- [hotc23val_nir](https://www.hsitracking.com/) ([results](https://github.com/YZCU/PHTrack/tree/master/tracking_results))
- [hotc23val_rednir](https://www.hsitracking.com/) ([results](https://github.com/YZCU/PHTrack/tree/master/tracking_results))
- [hotc23val_vis](https://www.hsitracking.com/) ([results](https://github.com/YZCU/PHTrack/tree/master/tracking_results))
- [hotc24val_nir](https://www.hsitracking.com/) ([results](https://github.com/YZCU/PHTrack/tree/master/tracking_results))
- [hotc24val_rednir](https://www.hsitracking.com/) ([results](https://github.com/YZCU/PHTrack/tree/master/tracking_results))
- [hotc24val_vis](https://www.hsitracking.com/) ([results](https://github.com/YZCU/PHTrack/tree/master/tracking_results))
- [mssot](https://www.sciencedirect.com/science/article/pii/S0924271623002551) ([results](https://github.com/YZCU/PHTrack/tree/master/tracking_results))
- [msvt](https://www.sciencedirect.com/science/article/pii/S0924271621002860) ([results](https://github.com/YZCU/PHTrack/tree/master/tracking_results))
--------------------------------------------------------------------------------------
For more comprehensive results and codes, please review the upcoming manuscript.

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

