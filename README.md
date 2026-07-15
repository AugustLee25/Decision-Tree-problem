# Bài toán: Dự đoán khách hàng mua điện thoại bằng Decision Tree

## 1. Mô tả bài toán
Một cửa hàng điện máy muốn dự đoán xem một khách hàng bước vào cửa hàng có khả năng mua mẫu điện thoại cao cấp hay không. Cửa hàng đã thu thập được dữ liệu của 8 khách hàng cũ, bao gồm 2 đặc trưng:
1. **Tuổi** (năm).
2. **Thu nhập** (triệu VNĐ/tháng).

Dựa vào đó, khách hàng được gắn nhãn là `1` (Có mua) hoặc `0` (Không mua).

### Bảng dữ liệu lịch sử:

| Tuổi | Thu nhập (triệu VNĐ) | Quyết định mua (1: Có, 0: Không) |
| :--: | :------------------: | :------------------------------: |
| 25   | 15                   | 0                                |
| 35   | 20                   | 0                                |
| 45   | 50                   | 1                                |
| 20   | 10                   | 0                                |
| 55   | 60                   | 1                                |
| 30   | 18                   | 0                                |
| 40   | 40                   | 1                                |
| 50   | 45                   | 1                                |

**Yêu cầu:** 
Xây dựng một mô hình Decision Tree bằng thư viện `scikit-learn` để huấn luyện trên tập dữ liệu này. Sau đó, dự đoán xem một khách hàng mới **(38 tuổi, thu nhập 35 triệu)** liệu có mua điện thoại không và trực quan hóa cây quyết định.

---

## 2. Cài đặt môi trường
Đảm bảo bạn đã cài đặt các thư viện cần thiết trước khi chạy code:
```bash
pip install numpy scikit-learn matplotlib
```
---

## 3. Giải thích thuật toán
* **`X_train` và `y_train`:** Dữ liệu đầu vào (features) và nhãn (labels) để huấn luyện mô hình học máy.
* **`DecisionTreeClassifier`:** Thuật toán tự động tính toán và chia các nhánh phân loại dựa trên độ lợi thông tin (Gini impurity) để tìm ra quy luật chính xác nhất.
* **`clf.predict()`:** Hàm dùng để đưa ra dự đoán cho dữ liệu mới mà AI chưa từng thấy trước đây.
* **`tree.plot_tree()`:** Vẽ và trực quan hóa sơ đồ phân nhánh của cây quyết định, giúp chúng ta có thể dễ dàng kiểm tra và giải thích cách AI đưa ra kết luận.
