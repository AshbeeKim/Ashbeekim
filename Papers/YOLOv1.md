# YOLOv1
---
[//]: <> (large, Large, LARGE, huge, HUGE, https://jsfiddle.net/8ndx694g/)
## Abstract
  Our unified architecture is extremely fast. Our base YOLO model processes images in real-time at 45 frames per second. A smaller version of the network, Fast YOLO, processes an astounding 155 frames per second while still achieving double the mAP of other real-time detectors. 
  Compared to state-of-the-art detection systems, YOLO makes more localization errors but is less likely to predict false positives on background. 
  Finally, YOLO learns very general representations of objects. It outperforms other detection methods, including DPM and R-CNN, when gener- alizing from natural images to other domains like artwork.

<img alt="Figure 1: The YOLO Detection System" src="https://github.com/AshbeeKim/AshbeeKim/blob/master/YOLOv1.jpeg"/>

<li>resizes the input image to 448 * 448</li>
<li>runs a single convolutional network on the image</li>
<li>thresholds the resulting detections by the model's sonfidence</li>

## 3 points should check
  1. 빠른 처리 속도. 회귀 문제로, 복잡한 파이프라인을 작성할 필요가 없음. 이미지 처리 속도는 초당 45프레임(no batch processing, Titan X GPU). fast version(GPU로 봐야하는지, 최근 YOLO*모델로 해석해야하는지 보류)은 150 fps*를 넘음. 이는 스트리밍 비디오에서 처리 지연 속도는 25ms 미만이란 뜻.

</br><p align="center"><img src="https://render.githubusercontent.com/render/math?math=\huge%20YOLO%7B%5Cgeqq%7D2*mAP"/>of other real-time systems</p></br>

  2. 높은 예측 설득력. (sliding window와 region proposal-based techniques와 다르게) YOLO는 train과 test에서의 전체 이미지를 보기 때문에, 즉시 class별 특성을 통해 문맥상 정보(contextual information)를 인코딩한다. 탐지 방법 중 최고인 Fast R-CNN*(2016년 기준), 전체 이미지(the larger context)를 볼 수 없기 때문에, 이미지의 객체도 배경으로 처리하는(patches) 실수가 있다.

<div align="center">
  <p align="left"><i align="left">the number of background errors,</i></p></br>
  <img src="https://render.githubusercontent.com/render/math?math=\huge%20YOLO%5Cleqq%5Cfrac%7BFast%20R-CNN%7D%7B2%7D"/></div></br>

  3. 일반화 할 수 있는(generalizable) 대표 특성 학습. wide margin으로 DPM이나 R-CNN과 같은 상위 탐지 모델을 능가함. new domains나 unexpected inputs을 적용해도, 일반화 가능성이 높기 때문에 거의 깨지지 않음.

</br>
  YOLO(v1 기준)는 여전히 최신 기술(state-of-the-art) 중 정확도 측면에서 뒤쳐짐. 이미지 내 객체를 빨리 구별이 가능하지만, 특히 작은 객체 등의 몇몇 객체를 정확하게 초점을 맞추는 것(localize)은 어려워 함. training, testing 코드는 오픈소스이며, 여러 pretrained models도 다운 가능하다고 적혀있음.
</br></br>

[their demo project webpage](https://pjreddie.com/darknet/yolo/)

</br>
<p>
* fps : frames per second</br>
* <a href="https://arxiv.org/pdf/1506.02640v5.pdf">YOLO</a> : You Only Look Once:Unified, Real-time Object Detection</br>
* <a href="https://arxiv.org/pdf/1504.08083v2.pdf">Fast R-CNN</a> : Fast Regions with CNN Features</br>
* <a href="https://arxiv.org/pdf/1409.5403v2.pdf">DPM</a> : Deformable parts models</br>
* <a href="https://arxiv.org/pdf/1311.2524v5.pdf">R-CNN</a> : Regions with CNN Features</br>
</p>
