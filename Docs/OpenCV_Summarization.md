<h1>OpenCV</h1>

-----

<h2><i>Transform</i></h2>

-----

</br>

$$A\;\cdot\;M=B$$

A ; src_input image

M ; scalar, martix, function

B ; dst_output image

</br>

**<big>resize</big>**
```python
cv.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) ->	dst
```

<p>𝚍𝚜𝚒𝚣𝚎 = 𝚂𝚒��e(𝚛𝚘𝚞𝚗𝚍(𝚏𝚡*𝚜𝚛𝚌.𝚌𝚘𝚕𝚜), 𝚛𝚘𝚞𝚗𝚍(𝚏𝚢*𝚜𝚛𝚌.𝚛𝚘𝚠𝚜))</p>
<p>fx = 0(default), (𝚍𝚘𝚞𝚋𝚕𝚎)𝚍𝚜𝚒𝚣𝚎.𝚠𝚒𝚍𝚝𝚑/𝚜𝚛𝚌.𝚌𝚘𝚕𝚜</p>
<p>fy = 0(default), (𝚍𝚘𝚞𝚋𝚕𝚎)𝚍𝚜𝚒𝚣𝚎.𝚑𝚎𝚒𝚐𝚑𝚝/𝚜𝚛𝚌.𝚛𝚘𝚠𝚜</p>
<p>Interpolation : </br><span align="center"><img width=300 alt="wikipedia interpolation image" src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/Comparison_of_1D_and_2D_interpolation.svg/1024px-Comparison_of_1D_and_2D_interpolation.svg.png"/></span></p>

</br>

**<big>warpAffine</big>**
```python
cv.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) ->	dst
```
$$\texttt{dst} (x,y) = \texttt{src} ( \texttt{M} _{11} x + \texttt{M} _{12} y + \texttt{M} _{13}, \texttt{M} _{21} x + \texttt{M} _{22} y + \texttt{M} _{23})$$

<p>M ; 2 by 3 transformation matrix</p>
</br>

**<big>getRotationMatrix2D</big>**
```python
cv.getRotationMatrix2D(center, angle, scale) ->	retval
```
$$\begin{bmatrix} \alpha & \beta & (1- \alpha ) \cdot \texttt{center.x} - \beta \cdot \texttt{center.y} \\ - \beta & \alpha & \beta \cdot \texttt{center.x} + (1- \alpha ) \cdot \texttt{center.y} \end{bmatrix}$$

</br>

where
$$\begin{array}{l} \alpha = \texttt{scale} \cdot \cos \texttt{angle} , \\ \beta = \texttt{scale} \cdot \sin \texttt{angle} \end{array}$$

</br>

**<big>getAffineTransform</big>**
```python
cv.getAffineTransform(src, dst) ->	retval
```
$$\begin{bmatrix} x'_i \\ y'_i \end{bmatrix} = \texttt{map_matrix} \cdot \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix}$$

</br>

where

$$dst(i)=(x'_i,y'_i), src(i)=(x_i, y_i), i=0,1,2$$

**<big>warpPerspective</big>**
```python
cv.warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) ->	dst
```
$$\texttt{dst} (x,y) = \texttt{src} \left ( \frac{M_{11} x + M_{12} y + M_{13}}{M_{31} x + M_{32} y + M_{33}} , \frac{M_{21} x + M_{22} y + M_{23}}{M_{31} x + M_{32} y + M_{33}} \right )$$

<p>M ; 3 by 3 transformation matrix</p>
</br>

**<big>getPerspectiveTransform</big>**
```python
cv.getPerspectiveTransform(src, dst) ->	retval
```
$$\begin{bmatrix} t_i x'_i \\ t_i y'_i \\ t_i \end{bmatrix} = \texttt{map_matrix} \cdot \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix}$$

</br>

where

$$dst(i)=(x'_i,y'_i), src(i)=(x_i, y_i), i=0,1,2,3$$


</br>

**<big>약어 정리</big>**
* src ; the source image
* dst ; the destination image
* M ; transformation matrix
