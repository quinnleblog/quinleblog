<vue-mathjax></vue-mathjax>

# Singular Value Decomposition P1 <span class="tex2jax_ignore">(</span>lý thuyết và ví dụ<span class="tex2jax_ignore">)</span>

*Reference: https://www.youtube.com/watch?v=P5mlg91as1c&t=468s*

## 1. Giới thiệu:

Singular Value Decomposition <span class="tex2jax_ignore">(</span>SVD<span class="tex2jax_ignore">)</span> đơn giản là một phép phân tích ma trận $A_{mxn}$, thành tích của các ma trận $U_{mxm}$.$\Sigma_{mxn}
$.$V_{nxn}^T$ hoặc $U_{mxr}$.$\Sigma_{rxr}
$.$V_{rxn}^T$ với r là rank<span class="tex2jax_ignore">(</span>A<span class="tex2jax_ignore">)</span>. Thuật toán SVD trong data science sẽ rơi vào trường hợp 2 nên chúng ta chỉ nói về trường hợp này. 

## 2. Tính chất và ý nghĩa của $U$, $\Sigma$ và $V^T$:

**size<span class="tex2jax_ignore">(</span>U<span class="tex2jax_ignore">)</span> = mxr, size<span class="tex2jax_ignore">(</span>$\Sigma$<span class="tex2jax_ignore">)</span> = rxr, size<span class="tex2jax_ignore">(</span>V<span class="tex2jax_ignore">)</span> = nxr, $V^T$ = transpose of V**

Nếu các bạn đã học qua Linear Algebra, các bạn sẽ biết tới phép chéo hóa một ma trận A, nghĩa là phân tích A thành tích các ma trận vuông cùng cấp $P^TDP$ với $D$ là một ma trận chéo. Mục đích của phép phân tích này là vì ma trận chéo dễ tính toán hơn và các cột của ma trận chéo là không phụ thuộc vào nhau <span class="tex2jax_ignore">(</span>linearly independent<span class="tex2jax_ignore">)</span>. Việc nhân ma trận $P^T$ và $P$ vào trước và sau $D$ có thể hiểu là ta đã đổi ma trận $A$ qua một hệ tọa độ khác và trong hệ tọa độ đó $A$ chính là $D$.


SVD là cách "chéo hóa" một ma trận chữ nhật. Cụ thể hơn, $\Sigma$ trong thuật toán SVD là một ma trận chéo với các phần tử trên đường chéo khác 0 và size của $\Sigma$ là rank<span class="tex2jax_ignore">(</span>A<span class="tex2jax_ignore">)</span>.  Các hàng của $V^T$ là các vector cơ sở của hệ tọa độ mới và $U\Sigma$ là tọa độ của $A$ trong hệ tọa độ mới đó. Điểu này ta sẽ thấy rõ hơn ở ví dụ bên dưới.

**Lưu ý:** các ma trận $U$ và $V$ là không duy nhất. Ví dụ, khi chúng ta đổi vị trí các phần tử trên dường chéo của $\Sigma$, $U$ và $V$ sẽ thay đổi theo. Tuy nhiên, nếu ta sắp xếp các phần tử của $\Sigma$ theo thứ tự giảm dần, $\Sigma$ tìm được là duy nhất.

## 3. Ứng dụng: 

*Hiện tại mình chỉ biết tới 2 ứng dụng của SVD, đó là data reduction và grouping <span class="tex2jax_ignore">(</span>clustering<span class="tex2jax_ignore">)</span>. Nếu bạn còn biết ứng dụng nào khác thì bổ sung giúp với nhé.*

### a. Data reduction:

Trong thực tế, ma trận $A$ có thể có size rất lớn, nhưng rất nhiều phần tử của $A$ đều bằng 0, nói cách khác, rank<span class="tex2jax_ignore">(</span>A<span class="tex2jax_ignore">)</span> << min{m,n}. Vì vậy, khi "chéo hóa" $A$, ta giảm được rất nhiều bộ nhớ sử dụng để lưu trữ $A$.

Trên thực tế, chúng ta có thể "ăn gian" một chút bằng việc chỉ giữ lại những giá trị lớn trên đường chéo của $\Sigma$, những giá trị nhỏ ta có thể xem như bằng 0 và loại bỏ. Xem xét ví dụ sau:

$$ \begin{bmatrix}  
1 & 1 & 1 & 0 & 0 \\\\
3 & 3 & 3 & 0 & 0 \\\\
4 & 4 & 4 & 0 & 0 \\\\
5 & 5 & 5 & 0 & 0 \\\\
0 & 2 & 0 & 4 & 4 \\\\
0 & 0 & 0 & 5 & 5 \\\\
0 & 1 & 0 & 2 & 2 
\end{bmatrix} = \begin{bmatrix}  
0.13 & 0.02 & -0.01 \\\\
0.41 & 0.07 & -0.03 \\\\
0.55 & 0.09 & -0.04 \\\\
0.68 & 0.11 & -0.05 \\\\
0.15 & -0.59 & 0.65 \\\\
0.07 & -0.73 & -0.67 \\\\
0.07 & -0.29 & 0.32 
\end{bmatrix} \begin{bmatrix}
12.4 & 0 & 0 \\\\
0 & 9.5 & 0 \\\\
0 & 0 & 1.3 
\end{bmatrix} \begin{bmatrix}  
0.56 & 0.59 & 0.56 & 0.09 & 0.09\\\\
0.12 & -0.02 & 0.12 & -0.69 & -0.69\\\\
0.40 & -0.80 & 0.04 & 0.09 & 0.09
\end{bmatrix} $$

Ta thấy rằng 2 giá trị đầu tiên của $\Sigma$ khá lớn so với giá trị thứ 3, vì vậy ta coi như giá trị thứ 3 bằng 0. Ta có ma trận $A'$ sau khi thay 1.3 thành 0:

$$ A' = \begin{bmatrix}
0.926 & 0.947 & 0.926 & 0.014 & 0.014 \\\\  
2.927 & 2.986 & 2.927 & -0.001 & -0.001 \\\\ 
3.922 & 4.007 & 3.922 & 0.024 & 0.024 \\\\
4.847 & 4.954 & 4.847 & 0.038 & 0.038 \\\\ 
0.369 & 1.21 & 0.369 & 4.035 & 4.035 \\\\ 
-0.346 & 0.651 & -0.346 & 4.863 & 4.863 \\\\ 
0.155 & 0.567 & 0.155 & 1.98 & 1.98 
\end{bmatrix} \approx A
$$

Vậy là hầu hết các thông tin của $A$ đều được giữ lại, nhưng ta chỉ cần phải lưu 7x2 + 2  <span class="tex2jax_ignore">(</span>chỉ cần lưu các giá trị trên đường chéo<span class="tex2jax_ignore">)</span> + 2x5 = 26 giá trị thay vì phải lưu 35 giá trị của $A$.

Những bài toán trong thực tế có r << min{m,n}, nghĩa là r rất nhỏ so với m,n và ta có thể tiết kiệm được một lượng lớn bộ nhớ bằng SVD  <span class="tex2jax_ignore">(</span>xem P2: real applications and coding guide để thấy rõ hơn<span class="tex2jax_ignore">)</span>. Hơn nữa, sau khi giảm chiều dữ liệu, việc tính toán cũng trở nên đơn giản hơn.

### b. Clustering:

Đây là ứng dụng mình thích nhất của SVD vì nhiều khi kết quả cho ra ảo đến bất ngờ. Cho ma trận $A$ như sau:
$$ \begin{bmatrix}
A & matrix & alien & gravity & lalaland & mebeforeyou\\\\  
user_1 & 1 & 1 & 1 & 0 & 0 \\\\
user_2 & 3 & 3 & 3 & 0 & 0 \\\\
user_3 & 4 & 4 & 4 & 0 & 0 \\\\
user_4 & 5 & 5 & 5 & 0 & 0 \\\\
user_5 & 0 & 2 & 0 & 4 & 4 \\\\
user_6 & 0 & 0 & 0 & 5 & 5 \\\\
user_7 & 0 & 1 & 0 & 2 & 2 
\end{bmatrix} $$

Giá trị các phần tử trong ma trận biểu hiện mức độ yêu thích của từng user đối với từng bộ phim. Ta có thể thấy các user chia làm 2 nhóm rất rõ rệt, các user 1,2,3,4 thích xem thể loại Science Fiction <span class="tex2jax_ignore">(</span>Sci-fi<span class="tex2jax_ignore">)</span>, trong khi user 5,6,7 thích thể loại Tình cảm lãng mạn <span class="tex2jax_ignore">(</span>Romance<span class="tex2jax_ignore">)</span>

Vậy câu hỏi là, làm sao máy tính có thể nhận diện được 2 nhóm này. Answer: SVD <span class="tex2jax_ignore">(</span>kinda obvious :D<span class="tex2jax_ignore">)</span>

Sau khi qua thuật toán SVD, ma trận A phân tích thành:

$$\begin{bmatrix}  
0.13 & 0.02 & -0.01 \\\\
0.41 & 0.07 & -0.03 \\\\
0.55 & 0.09 & -0.04 \\\\
0.68 & 0.11 & -0.05 \\\\
0.15 & -0.59 & 0.65 \\\\
0.07 & -0.73 & -0.67 \\\\
0.07 & -0.29 & 0.32 
\end{bmatrix} \begin{bmatrix}
12.4 & 0 & 0 \\\\
0 & 9.5 & 0 \\\\
0 & 0 & 1.3 
\end{bmatrix} \begin{bmatrix}  
0.56 & 0.59 & 0.56 & 0.09 & 0.09\\\\
0.12 & -0.02 & 0.12 & -0.69 & -0.69\\\\
0.40 & -0.80 & 0.04 & 0.09 & 0.09
\end{bmatrix} $$

Ta dễ dàng thấy rank của ma trận A bằng 3. Điều này nghĩa là số hàng độc lập của ma trận A bằng 3, i.e. ta có thể biểu diễn 7 vector chỉ với 3 vector cơ sở. Trong hệ tọa độ mới, 3 vector cơ sở này lần lượt là 3 hàng của $V^T$

Tích của ma trận $U$ và $\Sigma$ là:

$$\begin{bmatrix}  
user_1 & 1.612 & 0.19 & -0.013 \\\\
user_2 & 5.084 & 0.665 & -0.039 \\\\
user_3 & 6.82 & 0.855 & -0.052 \\\\
user_4 & 8.432 & 1.045 & -0.065 \\\\
user_5 & 1.86 & -5.605 & 0.845 \\\\
user_6 & 0.868 & -6.935 & -0.871 \\\\
user_7 & 0.868 & -2.755 & 0.416 
\end{bmatrix}
$$

Ta có $U$.$\Sigma[i]$ là tọa độ của user_i trong hệ tọa độ mới.

Các user 1,2,3,4 có tọa độ đầu lớn hơn hẳn tọa độ 2, và các user 5,6,7 có tọa độ thứ 2 lớn hơn tọa độ 1 <span class="tex2jax_ignore">(</span> lớn hơn theo giá trị tuyệt đối, tọa độ 3 ta bỏ qua vì quá nhỏ so với tọa độ 1 và 2<span class="tex2jax_ignore">)</span> Điều này chứng tỏ máy tính chia user 1,2,3,4 vào nhóm 1 và user 5,6,7 vào nhóm 2 mặc dù máy tính KHÔNG THỰC SỰ BIẾT nhóm 1 là Scifi và nhóm 2 là Romance.

Không những chia user thành từng nhóm, máy tính còn chia 5 bộ phim ra thành 2 nhóm riêng biệt. Lập luận tương tự như trên, ta có $\Sigma$.$V^T$ là tọa độ của từng bộ phim trong hệ tọa độ mới <span class="tex2jax_ignore">(</span>các vector cơ sở trong trường hợp này là các cột của $U$<span class="tex2jax_ignore">)</span>. Ma trận $\Sigma$$V^T$:

$$\begin{bmatrix}
matrix & alien & gravity & lalaland & mebeforeyou \\\\
6.944 & 7.316 & 6.944 & 1.116 & 1.116\\\\
1.14 & -0.19 & 1.14 & -6.555 & -6.555\\\\
0.52 & -1.04 & 0.52 & 0.117 & 0.117
\end{bmatrix} $$


Ta thấy đối với các phim Matrix, Alien và Gravity, tọa độ đầu tiên lớn hơn hẳn 2 tọa độ còn lại, và đối với Lalaland và Me before you, tọa độ thứ 2 lại lớn nhất. Điều này chứng tỏ máy tính phân Matrix, Alien, Gravity vào 1 nhóm và La la land, Me before you vào 1 nhóm. Nhắc lại là mặc dù chúng ta biết 2 nhóm này lần lượt là Scifi và Romance, máy tính không thực sự biết chính xác 2 nhóm này là gì, máy tính chỉ đọc dữ liệu từ ma trận $A$ và chia các bộ phim ra các nhóm tương ứng.



<br/>
<br/>
