OpenCV
===
-----

## *Transform*

-----

[//]: <> (<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">)

![formula](https://render.githubusercontent.com/render/math?math=A{\cdot}M=B)

* A ; src_input image
* M ; scalar, martix, function
* B ; dst_output image

-----

**resize**

```python
cv.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) ->	dst
```

* 𝚍𝚜𝚒𝚣𝚎 = 𝚂𝚒ze(𝚛𝚘𝚞𝚗𝚍(𝚏𝚡*𝚜𝚛𝚌.𝚌𝚘𝚕𝚜), 𝚛𝚘𝚞𝚗𝚍(𝚏𝚢*𝚜𝚛𝚌.𝚛𝚘𝚠𝚜))
* fx = 0(default), (𝚍𝚘𝚞𝚋𝚕𝚎)𝚍𝚜𝚒𝚣𝚎.𝚠𝚒𝚍𝚝𝚑/𝚜𝚛𝚌.𝚌𝚘𝚕𝚜
* fy = 0(default), (𝚍𝚘𝚞𝚋𝚕𝚎)𝚍𝚜𝚒𝚣𝚎.𝚑𝚎𝚒𝚐𝚑𝚝/𝚜𝚛𝚌.𝚛𝚘𝚠𝚜
* Interpolation : 

![wikipedia interpolation image](https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/Comparison_of_1D_and_2D_interpolation.svg/1024px-Comparison_of_1D_and_2D_interpolation.svg.png)

-----

**warpAffine**

```python
cv.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) ->	dst
```

![formula](https://render.githubusercontent.com/render/math?math=\mathrm{dst}%28x,y%29=\mathrm{src}%28\mathrm{M}_{11}x%2B\mathrm{M}_{12}y%2B\mathrm{M}_{13},\mathrm{M}_{21}x%2B\mathrm{M}_{22}y%2B\mathrm{M}_{23}%29)

* M ; 2 by 3 transformation matrix
  
-----

**getRotationMatrix2D**

```python
cv.getRotationMatrix2D(center, angle, scale) ->	retval
```

$$
\begin{bmatrix} \alpha & \beta & (1- \alpha ) \cdot \texttt{center.x} - \beta \cdot \texttt{center.y} \\ - \beta & \alpha & \beta \cdot \texttt{center.x} + (1- \alpha ) \cdot \texttt{center.y} \end{bmatrix}$$

where

$$
\begin{array}{l} \alpha = \texttt{scale} \cdot \cos \texttt{angle} , \\ \beta = \texttt{scale} \cdot \sin \texttt{angle} \end{array}$$

-----

**getAffineTransform**

```python
cv.getAffineTransform(src, dst) ->	retval
```

$$\begin{bmatrix} x'_i \\ y'_i \end{bmatrix} = \mathrm{map\,matrix} \cdot \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix}$$

where

$$dst(i)=(x'_i,y'_i), src(i)=(x_i, y_i), i=0,1,2$$

-----

**warpPerspective**

```python
cv.warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) ->	dst
```

$$
\texttt{dst} (x,y) = \texttt{src} \left ( \frac{M_{11} x + M_{12} y + M_{13}}{M_{31} x + M_{32} y + M_{33}} , \frac{M_{21} x + M_{22} y + M_{23}}{M_{31} x + M_{32} y + M_{33}} \right )$$

* M ; 3 by 3 transformation matrix
  
-----

**getPerspectiveTransform**

```python
cv.getPerspectiveTransform(src, dst) ->	retval
```

$$\begin{bmatrix} t_i x'_i \\ t_i y'_i \\ t_i\end{bmatrix} = \mathrm{map\,matrix} \cdot \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix}$$

where 

$$dst(i)=(x'_i,y'_i), src(i)=(x_i, y_i), i=0,1,2,3$$

-----

**약어 정리**
* src ; the source image
* dst ; the destination image
* M ; transformation matrix
