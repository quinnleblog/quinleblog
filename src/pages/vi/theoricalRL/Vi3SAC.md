<vue-mathjax></vue-mathjax>
# Singular Value Decomposition P1 <span class="tex2jax_ignore">(</span>lý thuyết và ví dụ<span class="tex2jax_ignore">)</span>

*Reference: https://www.youtube.com/watch?v=P5mlg91as1c&t=468s*

## 1. Giới thiệu:

Singular Value Decomposition <span class="tex2jax_ignore">(</span>SVD<span class="tex2jax_ignore">)</span> đơn giản là một phép phân tích ma trận $A_{mxn}$, thành tích của các ma trận $U_{mxm}$.$\Sigma_{mxn}
$.$V_{nxn}^T$ hoặc $U_{mxr}$.$\Sigma_{rxr}
$.$V_{rxn}^T$ với r là rank(A). Thuật toán SVD trong data science sẽ rơi vào trường hợp 2 nên chúng ta chỉ nói về trường hợp này. 

## 2. Tính chất và ý nghĩa của $U$, $\Sigma$ và $V^T$:

*size(U) = mxr, size($\Sigma$) = rxr, size(V) = nxr, $V^T$ = transpose of V*

Nếu các bạn đã học qua Linear Algebra, các bạn sẽ biết tới phép chéo hóa một ma trận A, nghĩa là phân tích A thành tích các ma trận vuông cùng cấp $P^T$.$D$.$P$ với $D$ là một ma trận chéo. Mục đích của phép phân tích này là vì ma trận chéo dễ tính toán hơn và các cột của ma trận chéo là không phụ thuộc vào nhau <span class="tex2jax_ignore">(</span>linearly independent<span class="tex2jax_ignore">)</span>. Việc nhân ma trận $P^T$ và $P$ vào trước và sau D có thể hiểu là ta đã đổi ma trận A qua một hệ tọa độ khác và trong hệ tọa độ đó A chính là D.


SVD là cách "chéo hóa" một ma trận chữ nhật. Cụ thể hơn, $\Sigma$ trong thuật toán SVD là một ma trận chéo với các phần tử trên đường chéo khác 0 và size của $\Sigma$ là rank(A).  Các hàng của $V^T$ là các vector cơ sở của hệ tọa độ mới và $U$.$\Sigma$ là tọa độ của A trong hệ tọa độ mới đó. Điểu này ta sẽ thấy rõ hơn ở ví dụ bên dưới.

Lưu ý: các ma trận $U$ và $V$ là không duy nhất. Ví dụ, khi chúng ta đổi vị trí các phần tử trên dường chéo của $\Sigma$, U và V sẽ thay đổi theo. Tuy nhiên, nếu ta sắp xếp các phần tử của $\Sigma$ theo thứ tự giảm dần, $\Sigma$ tìm được là duy nhất.

## 3. Ứng dụng: 

*Hiện tại mình chỉ biết tới 2 ứng dụng của SVD, đó là data reduction và grouping <span class="tex2jax_ignore">(</span>clustering<span class="tex2jax_ignore">)</span>. Nếu ai còn biết ứng dụng nào khác thì bổ sung giúp với nhé.*

### a. Data reduction:

Trong thực tế, ma trận A có thể có size rất lớn, nhưng rất nhiều phần tử của A đều bằng 0, nói cách khác, rank(A) << min{m,n}. Vì vậy, khi "chéo hóa" A, ta giảm được rất nhiều bộ nhớ sử dụng để lưu trữ A. Xem hình vẽ:

Trên thực tế, chúng ta có thể "ăn gian một chút" bằng việc chỉ giữ lại những giá trị lớn trên đường chéo của $\Sigma$, những giá trị nhỏ ta có thể xem như bằng 0 và loại bỏ. Xem xét ví dụ sau:

\[
  \left[ {\begin{array}{ccccc}
   1 & 2 & 3 & 4 & 5\\
   3 & 4 & 5 & 6 & 7\\
  \end{array} } \right]
  = 

\]
Ta thấy rằng 2 giá trị đầu tiên khá lớn so với giá trị thứ 3, vì vậy ta coi như giá trị thứ 3 bằng 0. Ta có ma trận A* sau khi thay 1.3 thành 0:
*Hiển thị A* gần bằng A

Vậy là hầu hết các thông tin của A đều được giữ lại, nhưng ta chỉ cần phải lưu 7x2 + 2  <span class="tex2jax_ignore">(</span>chỉ cần lưu các giá trị trên đường chéo<span class="tex2jax_ignore">)</span> + 2x5 = 26 giá trị thay vì phải lưu 35 giá trị của A.

Những bài toán trong thực tế có r << min{m,n}, nghĩa là r rất nhỏ so với m,n và ta có thể tiết kiệm được một lượng lớn bộ nhớ bằng SVD  <span class="tex2jax_ignore">(</span>xem P2: real applications and coding guide để thấy rõ hơn<span class="tex2jax_ignore">)</span>. Hơn nữa, sau khi giảm chiều dữ liệu, việc tính toán cũng trở nên đơn giản hơn.

### b. Clustering:

Đây là ứng dụng mình thích nhất của SVD vì nhiều khi kết quả cho ra ảo đến bất ngờ. Cho ma trận A như sau:

Giá trị các phần tử trong ma trận biểu hiện mức độ yêu thích của từng user đối với từng bộ phim. Ta có thể thấy các user chia làm 2 nhóm rất rõ rệt, các user 1,2,3,4 thích xem thể loại Science Fiction <span class="tex2jax_ignore">(</span>Sci-fi<span class="tex2jax_ignore">)</span>, trong khi user 5,6,7 thích thể loại Tình cảm lãng mạn <span class="tex2jax_ignore">(</span>Romance<span class="tex2jax_ignore">)</span>

Vậy câu hỏi là, làm sao máy tính có thể nhận diện được 2 nhóm này. Answer: SVD <span class="tex2jax_ignore">(</span>kinda obvious :D<span class="tex2jax_ignore">)</span>

Sau khi qua thuật toán SVD, ma trận A phân tích thành:

Ta dễ dàng thấy rank của ma trận A bằng 3. Điều này nghĩa là số hàng độc lập của ma trận A bằng 3, i.e. ta có thể biểu diễn 7 vector chỉ với 3 vector cơ sở. Trong hệ tọa độ mới, 3 vector cơ sở này làn lượt là 3 hàng của $V^T$

Vậy ta suy ra $U$.$\Sigma[i]$ là tọa độ của user[i] trong hệ tọa độ mới.

Tích của ma trận $U$ và $\Sigma$ là:

Các user 1,2,3,4 có tọa độ đầu lớn hơn hẳn tọa độ 2, và các user 5,6,7 có tọa độ thứ 2 lớn hơn tọa độ 1 <span class="tex2jax_ignore">(</span>lưu ý là lớn hơn theo giá trị tuyệt đối, tọa độ 3 ta bỏ qua vì quá nhỏ so với tọa độ 1 và 2<span class="tex2jax_ignore">)</span> Điều này chứng tỏ máy tính chia user 1,2,3,4 vào nhóm 1 và user 5,6,7 vào nhóm 2 mặc dù máy tính KHÔNG THỰC SỰ BIẾT nhóm 1 là Scifi và nhóm 2 là Romance.

Không những chia user thành từng nhóm, máy tính còn chia 5 bộ phim ra thành 2 nhóm riêng biệt. Lập luận tương tự như trên, ta có $\Sigma$.$V^T$ là tọa độ của từng bộ phim trong hệ tọa độ mới <span class="tex2jax_ignore">(</span>các vector cơ sở trong trường hợp này là các cột của $U$<span class="tex2jax_ignore">)</span>. Ma trận  $\Sigma$.$V^T$ 

Ta thấy đối với các phim Matrix, Alien và Serenity, tọa độ đầu tiên lớn hơn hẳn 2 tọa độ còn lại, và đối với Casablanca và Amelie, tọa độ thứ 2 lại lớn nhất. Điều này chứng tỏ máy tính phân Matrix, Alien, Serenity vào 1 nhóm và Casablanca, Amelie vào 1 nhóm. Nhắc lại là mặc dù chúng ta biết 2 nhóm này lần lượt là Scifi và Romance, máy tính không thực sự biết chính xác 2 nhóm này là gì, máy tính chỉ đọc dữ liệu từ ma trận A và chia các bộ phim ra các nhóm tương ứng.

Ta lấy tọa độ của từng user và plot trên hệ tọa độ Oxy
(chèn hình)


<br/>
<br/>
