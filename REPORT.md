# Mục lục

I. Giới thiệu

II. Tiền xử lý dữ liệu
  a. Làm sạch dữ liệu
  b. Phân tích dữ liệu

III. Thuật toán dự đoán

IV. Đánh giá mô hình
  a. Kết quả đánh giá
  b. So sánh

V. Kết luận

VI. Tài liệu tham khảo

---

I. Giới thiệu

Bài toán phân loại chủ đề bài báo tiếng Việt là một trong những ứng dụng quan trọng của lĩnh vực xử lý ngôn ngữ tự nhiên (NLP). Mục tiêu của bài toán là dự đoán chủ đề (category) của một bài báo dựa trên các đặc trưng về tần suất xuất hiện của các từ khóa và độ dài bài viết.

Bộ dữ liệu sử dụng trong báo cáo này là `vietnamese_news_dataset.csv`, bao gồm:
- Các cột từ `ai` đến `y_te`: đại diện cho tần suất xuất hiện của các từ khóa trong từng bài báo.
- Cột `doc_length`: độ dài của bài báo.
- Cột `category`: biến mục tiêu (class attribute) cần dự đoán, dạng phân loại (nominal).
- Cột `id`: không sử dụng trong phân tích.

Sau khi loại bỏ cột `id` và cột mục tiêu `category`, số lượng feature dùng cho mô hình là tổng số cột từ khóa cộng với `doc_length`. Số lượng lớp (category) là số giá trị duy nhất trong cột `category`.

Mục tiêu của báo cáo là xây dựng và đánh giá các mô hình học máy (Naive Bayes, Random Forest) để dự đoán chủ đề của bài báo dựa trên các đặc trưng đã cho.

---

II. Tiền xử lý dữ liệu

a. Làm sạch dữ liệu

Các bước tiền xử lý dữ liệu được thực hiện bằng Python với pandas như sau:

- Đọc file CSV vào DataFrame.
- Loại bỏ cột `id` không cần thiết.
- Thay thế các giá trị thiếu (NaN) trong các cột feature (từ khóa và `doc_length`) bằng 0.
- Kiểm tra và loại bỏ các dòng dữ liệu trùng lặp (nếu có).
- Đảm bảo các cột feature là kiểu số (numeric), cột mục tiêu `category` là kiểu phân loại (categorical).

Ví dụ mã nguồn:

```python
import pandas as pd

# Đọc dữ liệu
df = pd.read_csv('vietnamese_news_dataset.csv')

# Loại bỏ cột id
df = df.drop(columns=['id'])

# Thay thế NaN trong các cột feature bằng 0
feature_cols = df.columns.difference(['category'])
df[feature_cols] = df[feature_cols].fillna(0)

# Loại bỏ dòng trùng lặp
df = df.drop_duplicates()

# Đảm bảo kiểu dữ liệu
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
df['category'] = df['category'].astype('category')

# Hiển thị 5 dòng đầu sau khi làm sạch
df.head()
```

b. Phân tích dữ liệu

- Phân bố số lượng bài báo theo từng category:

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,4))
sns.countplot(x='category', data=df, order=df['category'].value_counts().index)
plt.title('Phân bố số lượng bài báo theo category')
plt.xlabel('Category')
plt.ylabel('Số lượng bài báo')
plt.show()
```

- Phân bố của `doc_length`:

```python
plt.figure(figsize=(8,4))
sns.histplot(df['doc_length'], kde=True)
plt.title('Phân bố độ dài bài báo (doc_length)')
plt.xlabel('Độ dài bài báo')
plt.ylabel('Tần suất')
plt.show()
```

- Boxplot `doc_length` theo từng category (tùy chọn):

```python
plt.figure(figsize=(10,5))
sns.boxplot(x='category', y='doc_length', data=df)
plt.title('Boxplot độ dài bài báo theo category')
plt.xlabel('Category')
plt.ylabel('Độ dài bài báo')
plt.show()
```

---

III. Thuật toán dự đoán

Mục tiêu của bước này là xây dựng các mô hình học máy để phân loại bài báo vào các category dựa trên các đặc trưng đã xử lý.

Các bước chuẩn bị dữ liệu cho mô hình:
- Tách tập feature (X) và biến mục tiêu (y).
- Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm tra (20%) bằng `train_test_split` với random_state để đảm bảo tái lập.

Ví dụ mã nguồn:

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['category'])
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Các thuật toán được sử dụng:

- **Naive Bayes**: Sử dụng MultinomialNB phù hợp với dữ liệu tần suất từ khóa.
- **Random Forest**: Sử dụng RandomForestClassifier cho bài toán phân loại tổng quát.

Ví dụ mã nguồn huấn luyện mô hình:

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
```

Sau khi huấn luyện, các mô hình sẽ được đánh giá trên tập kiểm tra ở phần tiếp theo.

---

IV. Đánh giá mô hình

a. Kết quả đánh giá

Hiệu suất của các mô hình được đánh giá trên tập kiểm tra (test set) bằng các chỉ số: Accuracy, Kappa statistic, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Classification Report (precision, recall, f1-score), và ma trận nhầm lẫn (Confusion Matrix).

Ví dụ mã nguồn đánh giá:

```python
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error, mean_squared_error, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Dự đoán trên tập kiểm tra
nb_pred = nb_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# Đánh giá Naive Bayes
print('--- Naive Bayes ---')
print('Accuracy:', accuracy_score(y_test, nb_pred))
print('Kappa:', cohen_kappa_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))

# Tính MAE và RMSE cho Naive Bayes (label encoding)
le = LabelEncoder()
y_test_num = le.fit_transform(y_test)
nb_pred_num = le.transform(nb_pred)
print('MAE:', mean_absolute_error(y_test_num, nb_pred_num))
print('RMSE:', np.sqrt(mean_squared_error(y_test_num, nb_pred_num)))

# Ma trận nhầm lẫn Naive Bayes
ConfusionMatrixDisplay.from_predictions(y_test, nb_pred, cmap='Blues')
plt.title('Confusion Matrix - Naive Bayes')
plt.show()

# Đánh giá Random Forest
print('--- Random Forest ---')
print('Accuracy:', accuracy_score(y_test, rf_pred))
print('Kappa:', cohen_kappa_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# Tính MAE và RMSE cho Random Forest (label encoding)
rf_pred_num = le.transform(rf_pred)
print('MAE:', mean_absolute_error(y_test_num, rf_pred_num))
print('RMSE:', np.sqrt(mean_squared_error(y_test_num, rf_pred_num)))

# Ma trận nhầm lẫn Random Forest
ConfusionMatrixDisplay.from_predictions(y_test, rf_pred, cmap='Greens')
plt.title('Confusion Matrix - Random Forest')
plt.show()
```

b. So sánh

So sánh hiệu suất hai mô hình dựa trên các chỉ số Accuracy, Kappa, F1-score trung bình. Có thể trực quan hóa bằng biểu đồ cột:

```python
import matplotlib.pyplot as plt
import numpy as np

metrics = ['Accuracy', 'Kappa', 'F1-score']
nb_scores = [accuracy_score(y_test, nb_pred),
             cohen_kappa_score(y_test, nb_pred),
             classification_report(y_test, nb_pred, output_dict=True)['weighted avg']['f1-score']]
rf_scores = [accuracy_score(y_test, rf_pred),
             cohen_kappa_score(y_test, rf_pred),
             classification_report(y_test, rf_pred, output_dict=True)['weighted avg']['f1-score']]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(7,4))
plt.bar(x - width/2, nb_scores, width, label='Naive Bayes')
plt.bar(x + width/2, rf_scores, width, label='Random Forest')
plt.xticks(x, metrics)
plt.ylabel('Giá trị')
plt.title('So sánh hiệu suất mô hình')
plt.legend()
plt.show()
```

Nhận xét: Dựa trên các chỉ số đánh giá, mô hình có giá trị Accuracy, Kappa, F1-score cao hơn sẽ phù hợp hơn cho bài toán phân loại chủ đề bài báo này.

---

V. Kết luận

...

---

VI. Tài liệu tham khảo

...
