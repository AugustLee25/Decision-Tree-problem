import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# 1. Chuẩn bị dữ liệu huấn luyện (Training Data)
# Ma trận X chứa các đặc trưng: [Tuổi, Thu nhập]
X_train = np.array([
    [25, 15],
    [35, 20],
    [45, 50],
    [20, 10],
    [55, 60],
    [30, 18],
    [40, 40],
    [50, 45]
])

# Vector y chứa nhãn tương ứng: 0 (Không mua), 1 (Có mua)
y_train = np.array([0, 0, 1, 0, 1, 0, 1, 1])

# 2. Khởi tạo và huấn luyện mô hình Decision Tree
# Sử dụng thuật toán phân loại. Đặt random_state=42 để kết quả luôn ổn định mỗi lần chạy
clf = DecisionTreeClassifier(criterion='gini', random_state=42)

# Cho mô hình học (train) từ dữ liệu
clf.fit(X_train, y_train)

# 3. Dự đoán dữ liệu mới (Testing)
# Khách hàng mới: 38 tuổi, thu nhập 35 triệu
X_new = np.array([[38, 35]])

# Lấy kết quả dự đoán
y_pred = clf.predict(X_new)

# In kết quả
ket_qua = "Có mua" if y_pred[0] == 1 else "Không mua"
print(f"=> Dự đoán cho khách hàng mới (38 tuổi, thu nhập 35tr): {ket_qua}")

# 4. Trực quan hóa mô hình Cây Quyết Định (Visualize)
plt.figure(figsize=(8, 5))
tree.plot_tree(
    clf, 
    feature_names=['Tuổi', 'Thu nhập'], 
    class_names=['Không mua', 'Có mua'], 
    filled=True, 
    rounded=True
)
plt.title("Sơ đồ Decision Tree dự đoán hành vi mua hàng")
plt.show()