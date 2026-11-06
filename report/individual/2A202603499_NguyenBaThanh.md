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
Nhãn ham cho câu truy vấn "you have been selected to receive a complimentary holiday, reply now to claim".

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
