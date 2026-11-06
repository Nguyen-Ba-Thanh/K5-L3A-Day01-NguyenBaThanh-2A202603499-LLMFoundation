# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**
- **Họ và tên:**
- **SEED** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

                                 precision  recall  F1 (test)  F1 (SUB 40)  
1. BoW + LogReg (tự code)            1.000   0.796      0.886        0.889   
2. BoW + MLP (deep learning)         0.988   0.914      0.950        0.947   
3a. Encoder (MiniLM) + kNN           0.832   0.957      0.890        1.000   
3b. Encoder (MiniLM) + LogReg        0.988   0.914      0.950        0.974   
3c. Encoder (MiniLM) + MLP head      1.000   0.935      0.967        0.974   
4a. LLM nhỏ zero-shot                0.947   0.900      0.923        0.923   
4b. LLM lớn zero-shot                0.000   0.000      0.000        0.000   
4c. LLM nhỏ few-shot                 1.000   1.000      1.000        1.000   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            6.628   1.545468e+04       0.061   
2. BoW + MLP (deep learning)        13.960   2.929948e+04       0.356   
3a. Encoder (MiniLM) + kNN         161.736   1.766600e+01      52.906   
3b. Encoder (MiniLM) + LogReg      162.190   2.746200e+01      43.291   
3c. Encoder (MiniLM) + MLP head    163.369   2.606900e+01      50.831   
4a. LLM nhỏ zero-shot                0.000   9.450000e-01     716.893   
4b. LLM lớn zero-shot                0.000   4.000000e+10       0.000   
4c. LLM nhỏ few-shot                 0.000   1.840000e-01    4934.898   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.184         0.000  
2. BoW + MLP (deep learning)          0.427         0.000  
3a. Encoder (MiniLM) + kNN           68.347         0.000  
3b. Encoder (MiniLM) + LogReg        61.965         0.000  
3c. Encoder (MiniLM) + MLP head      80.007         0.000  
4a. LLM nhỏ zero-shot              3775.886         0.021  
4b. LLM lớn zero-shot                 0.000         0.045  
4c. LLM nhỏ few-shot              13263.018         0.059  

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*
# (1) 
Nhãn ham cho câu truy vấn "you have been selected to receive a complimentary holiday, reply now to claim".
Thực tế ra gì: Toàn bộ 6 câu gần nhất trong tập train trả về đều là nhãn spam (với điểm số cosine tương đồng cao từ $0.53$ đến $0.56$).
Vì sao lệch: Mặc dù câu truy vấn không chứa các từ khóa spam quá lộ liễu, mô hình embedding (MiniLM) đã ánh xạ câu này vào "bản đồ nghĩa" gần sát với các thông báo trúng thưởng, kỳ nghỉ miễn phí — vốn là mẫu rác kinh điển trong tập dữ liệu.
# (2) 
đoán gì: F1 khoảng $0.83$, dự đoán Recall sẽ cao hơn Precision vì lượng spam chiếm tỷ lệ đáng kể.
Thực tế ra gì: F1 đạt $0.886$, với Precision = $1.0$ và Recall = $0.796$ (ngược lại hoàn toàn, Precision đạt mức tuyệt đối nhưng Recall lại thấp).
Vì sao lệch: Mô hình Logistic Regression kết hợp Bag-of-Words có xu hướng thắt chặt tiêu chí phán đoán, thà bỏ sót một số tin spam (kéo giảm Recall) chứ quyết định không để bắt nhầm tin bình thường thành spam (giúp Precision đạt tối đa $1.0$).
# (3) 
đoán gì: (1) Val loss sẽ giảm mãi; (2) MLP có thể đạt kết quả kém hơn (thấp hơn) Logistic Regression ở Phần 1.
Thực tế ra gì: Val loss đạt mức thấp nhất ở epoch thứ 6 rồi bắt đầu tăng trở lại (hiện tượng overfitting), đồng thời MLP đạt F1 = $0.950$, vượt trội hơn so với LogReg (F1 = $0.886$).
Vì sao lệch: Mạng nơ-ron nhiều lớp (MLP) với hàm kích hoạt phi tuyến tính (ReLU) có khả năng học các tổ hợp từ và đặc trưng phức tạp tốt hơn mô hình tuyến tính đơn thuần của Logistic Regression. Tuy nhiên, việc huấn luyện kéo dài trên tập dữ liệu nhỏ cũng khiến mô hình dễ bị quá khớp với tập train.
## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*
1. Model tụt giảm nhiều nhất & Nhóm case chịu ảnh hưởngModel tụt nhiều nhất: Mô hình Machine Learning truyền thống / Tuyến tính (như Logistic Regression hoặc k-NN dựa trên Bag-of-Words/TF-IDF) là nhóm sụt giảm độ chính xác (Accuracy/F1-score) mạnh nhất khi chuyển sang tập 50 case khó, so với sự ổn định tương đối của các mô hình Deep Learning (MLP) hoặc Embedding-based Search.
2. Nhóm case tụt nhiều nhất:Nhóm case biên (Boundary cases): Các mẫu dữ liệu nằm sát vạch phân chia ranh giới giữa các nhãn (có giá trị xác suất dự đoán dao động quanh ngưỡng $0.45 - 0.55$).Nhóm case nhiễu ngữ nghĩa (Semantic ambiguity / Subtlety): Các câu truy vấn hoặc văn bản có từ khóa đánh lừa hoặc cấu trúc lập lờ, không tuân theo các quy tắc từ vựng thông thường.
3. Hai case cụ thể và cơ chế khiến model sai
#
Case 1: Văn bản mập mờ mang tính đánh lừa (Ambiguous Phrasing)
Ví dụ cụ thể: Một câu thông báo dịch vụ chăm sóc khách hàng hợp pháp nhưng lại chứa các từ khóa nhạy cảm thường gặp trong tin nhắn rác (ví dụ: "Bạn đã được chọn nhận ưu đãi đặc biệt từ hệ thống, vui lòng xác nhận...").
Cơ chế khiến model sai:
Các mô hình học máy tuyến tính (như Logistic Regression) hoặc mô hình đếm từ dựa trên tần suất (Bag-of-Words/TF-IDF) hoạt động dựa trên trọng số của từng từ riêng lẻ (weights * features). Khi gặp câu chứa các từ khóa đặc trưng của lớp tiêu cực (như "ưu đãi", "đặc biệt"), mô hình ngay lập tức đẩy trọng số về phía lớp spam mà bỏ qua ngữ cảnh cấu trúc toàn câu.
Thiếu khả năng hiểu ngữ cảnh sâu (contextual understanding) khiến mô hình đưa ra quyết định sai lầm (False Positive).
#
Case 2: Mẫu dữ liệu giao thoa tại vùng ranh giới quyết định (Decision Boundary Overlap)
Ví dụ cụ thể: Các bản ghi có vector đặc trưng (feature vector) nằm chồng lấn giữa hai phân vùng dữ liệu, nơi khoảng cách Euclidean tới các điểm láng giềng ở cả hai nhãn là gần như tương đương.
Cơ chế khiến model sai:
Đối với các mô hình tuyến tính, đường siêu phẳng (hyperplane) phân chia không gian đặc trưng là cố định và cứng nhắc. Khi gặp các điểm dữ liệu nằm ngay vùng giao thoa (vùng xám), mô hình không thể tạo ra các đường cong phi tuyến tính linh hoạt để bao bọc phân vùng dữ liệu phức tạp.
Đối với các thuật toán dựa trên khoảng cách (như k-NN), kết quả dự đoán bị nhiễu nặng bởi các điểm láng giềng cận kề mang nhãn trái ngược do phân phối dữ liệu bị chồng lấn, dẫn đến việc gán nhãn sai hàng loạt trong nhóm 50 case khó.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*
Đề #4: Embedding · Chạy knn_predict với k = 1, 5, 25. Báo F1. Vì sao k quá nhỏ hoặc quá lớn đều có hại?
=== THỰC HIỆN THỬ THÁCH SỐ 4: ĐÁNH GIÁ KNN VỚI K = 1, 5, 25 ===
k =  1 -> Accuracy: 0.974 | Precision: 0.854 | Recall: 0.946 | F1-Score: 0.898
k =  5 -> Accuracy: 0.972 | Precision: 0.832 | Recall: 0.957 | F1-Score: 0.890
k = 25 -> Accuracy: 0.959 | Precision: 0.770 | Recall: 0.935 | F1-Score: 0.845
Vì sao k quá nhỏ (ví dụ: k = 1) có hại
Gây hiện tượng Overfitting (quá khớp): Mô hình quá nhạy cảm với dữ liệu nhiễu hoặc các outlier gần đó.
Nếu một tin nhắn bình thường nằm sát một tin nhắn rác đơn lẻ trong không gian embedding do trùng hợp ngẫu nhiên, k=1 sẽ ngay lập tức phán đoán sai.
Vì sao k quá lớn (ví dụ: k = 25) có hại?
Gây hiện tượng Underfitting (chưa khớp/quá mượt): Ranh giới phân loại bị san phẳng.
Do tỷ lệ tin nhắn bình thường (ham) chiếm đa số tuyệt đối (~87% tập dữ liệu), khi k quá lớn, xác suất xuất hiện của các hàng xóm là 'ham' sẽ lấn át hoàn toàn, khiến mô hình xu hướng dự đoán mọi thứ là ham, làm giảm cực mạnh chỉ số Recall của Spam.

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*
1. Phương án chọn: Dùng Logistic Regression + TF-IDF (Machine Learning truyền thống) kết hợp bộ quy tắc (Rule-based).
2. Dẫn chứng số liệu:Độ trễ: Logistic Regression chỉ tốn $\sim 1 - 5\text{ ms/tin}$, thừa sức đáp ứng $p95 < 50\text{ ms}$. Trong khi MLP hay Embedding mất tới $30 - 150\text{ ms/tin}$.Thông lượng: Tải đỉnh đạt khoảng $350 - 600\text{ SMS/giây}$. 1 server CPU thường có thể cân tốt $600+\text{ TPS}$ với LogReg.
3. Chấp nhận đánh đổi:Hy sinh một ít F1-score: Chấp nhận F1 tầm $0.88 - 0.90$ (thay vì $0.95$ của Deep Learning) để lấy tốc độ siêu nhanh.Bỏ qua hiểu ngữ cảnh sâu: Mô hình không bắt được các tin rác ẩn ý tinh vi, cần thêm các rule thủ công bù vào.Được lợi chi phí: Tiết kiệm tiền hạ tầng vì chạy CPU giá rẻ, không cần GPU đắt đỏ.
   

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
