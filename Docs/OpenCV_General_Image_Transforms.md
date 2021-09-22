# General Image Transforms

* Stretch, Shrink, Warp, and Rotate
  * Resize
  * WarpAffine
  * WarpPerspective

* General Remappings
  * Polor Mappings
  * LogPolar
  * Arbitrary Mappings
* Image Repair
  * Inpainting
  * Denoising
* Histogram Equalization(Contrast)

## cv.resize


```python
cv.resize(image, (W, H), interpolation=cv.INTER_CUBIC)

cv.resize(image, fx=0.5, fy=1.5, interpolation=cv.INTER_CUBIC)
# %로 표현도 가능함
```


함수에서 사용한 cv.resize의 경우 Interpolation을 사용해서 이미지 크기 재조정으로 손실된 정보를 채운다.

* INTER_NEAREST ; Nearest neighbor
* INTER_LINEAR ; Bilinear(default)
* INTER_AREA ; Pixel area resampling
* INTER_CUBIC ; Bicubic interpolation
* INTER_LANCZOS4 ; Lanczos interpolation over 8*8 neighborhood

O'reilly에 기술된 내용에 따르면, INTER_LINEAR(bilinear)가 아닌 다른 방법을 권유하고 있다. 이미지 재조정을 할 때, 가장 쉬운 접근 방법으로 INTER_NEAREST가 있으며, 이미지별 pixel 규격(2*2, 4*4, 8*8)내 색상 혼잡도에 따라 적용해야할 방법이 달리진다.

앞서 작성한 함수에서 사이즈 확대 시, INTER_CUBIC, INTER_LINEAR, 사이즈 축소 시, INTER_AREA로 설정한 이유와도 이어진다.

## cv.warpAffine


```python
H, W, C = image.shape
M = np.float32([[1,0,x_warp],[0,1,y_warp]])
cv.warpaffine(image, M, (W, H))
```


C와 다르게 비교적 간결한 문법, M은 변환 행렬을 의미하는데, 여기서는 2*3을 사용한다.

변환 행렬 중 column 0:2는 이미지의 W, H를 의미하고 column2는 이동방향을 의미한다.

따라서 음수값을 입력하면, x,y축에서 음의 방향으로 이동하는 것을 볼 수 있다.

## cv.getRotationMatrix2D


```python
H, W = image.shape[:2]

M = cv.getRotationMatrix2D((W/2, H/2), 135, 0.5)
cv.warpAffine(image, M, (W, H))
```


회전의 방향, 배율을 설정해서 확대할 수도, 축소할 수도 있음

중심점을 image.shape으로 확인했던 Height, Width의 중점으로 설정할 수도 있지만, 특정값을 지정하면 해당 지점을 기준으로 변환함

## cv.getAffineTransform


```python
H, W, C = image.shape

input_points = np.float32([[x1, y1],[x2,y2],[x3,y3]])
output_points = np.float32([[x'1, y'1], [x'2, y'2], [x'3, y'3]])

cv.circle(image, tuple(input_points[0].astype(np.int)), point_radius, (255, 0, 0), -1)
cv.circle(image, tuple(input_points[1].astype(np.int)), point_radius, (0, 255, 0), -1)
cv.circle(image, tuple(input_points[2].astype(np.int)), point_radius, (0, 0, 255), -1)

M = cv.getAffineTransform(input_points, output_point)

cv.warpAffine(image, M, (W, H))
```


circle로 각 포인트를 선언하는 것은 선택사항이나, 어떻게 변환되었는가를 확인할 방법으로 생각됨

결과 사진을 보면서 막 변형을 하게 되면 학습의 효율이 오히려 떨어질지도 모른다고 판단

## cv.warpPerspective and  cv.getPerspectiveTransform


```python

```


~~

[//]: <> (어이가 매우 없어서 승질나는 부분,,, 불러온 이미지 shape은 H, W, C,,,근데 본인들 패키지는 W, H...그러니까 x, y순으로 받음...휴..별...)

Augmentaion과도 이어지는 이후의 내용 중 코드로 설명이 가능한 부분은 진행하고 수식으로 설명을 요하는 부분은 Albumentations에서 추가 기술할 예정이다.
