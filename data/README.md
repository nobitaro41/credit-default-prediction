# Credit Default Prediction – Data Dictionary

## 1. Tổng quan

Dataset được sử dụng cho bài toán **Supervised Learning – Binary Classification** nhằm dự đoán liệu một khách hàng có xảy ra tình trạng **quá hạn thanh toán nghiêm trọng trong vòng 2 năm** hay không.

* **Target:** `SeriousDlqin2yrs`
* `0`: Không xảy ra tình trạng quá hạn nghiêm trọng trong 2 năm.
* `1`: Có xảy ra tình trạng quá hạn nghiêm trọng trong 2 năm.

---

## 2. Ý nghĩa các cột

| Column                                 | Ý nghĩa                                                                                                                                                                      |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Unnamed: 0`                           | Cột index/ID được tạo ra khi dữ liệu được lưu từ DataFrame sang CSV. Thường không chứa thông tin có ý nghĩa đối với bài toán và nên được loại bỏ trước khi huấn luyện model. |
| `SeriousDlqin2yrs`                     | **Target variable**. Cho biết khách hàng có xảy ra tình trạng quá hạn thanh toán nghiêm trọng trong vòng 2 năm hay không. `0` = không, `1` = có.                             |
| `RevolvingUtilizationOfUnsecuredLines` | Tỷ lệ sử dụng các hạn mức tín dụng quay vòng không có tài sản đảm bảo. Giá trị càng cao cho thấy khách hàng đang sử dụng phần lớn hạn mức tín dụng của mình.                 |
| `age`                                  | Tuổi của khách hàng, tính bằng năm.                                                                                                                                          |
| `NumberOfTime30-59DaysPastDueNotWorse` | Số lần khách hàng từng quá hạn thanh toán từ **30 đến 59 ngày**, nhưng chưa chuyển sang tình trạng quá hạn nghiêm trọng hơn.                                                 |
| `DebtRatio`                            | Tỷ lệ nợ của khách hàng, phản ánh mức độ nghĩa vụ nợ so với thu nhập. Giá trị cao thường cho thấy khách hàng đang có gánh nặng nợ lớn hơn.                                   |
| `MonthlyIncome`                        | Thu nhập hàng tháng của khách hàng.                                                                                                                                          |
| `NumberOfOpenCreditLinesAndLoans`      | Số lượng tài khoản tín dụng và các khoản vay đang mở của khách hàng.                                                                                                         |
| `NumberOfTimes90DaysLate`              | Số lần khách hàng từng quá hạn thanh toán **90 ngày trở lên**. Đây là một feature quan trọng để đánh giá lịch sử tín dụng và nguy cơ default.                                |
| `NumberRealEstateLoansOrLines`         | Số lượng khoản vay hoặc hạn mức tín dụng liên quan đến bất động sản, chẳng hạn như các khoản vay thế chấp (mortgage).                                                        |
| `NumberOfTime60-89DaysPastDueNotWorse` | Số lần khách hàng từng quá hạn thanh toán từ **60 đến 89 ngày**, nhưng chưa chuyển sang tình trạng quá hạn nghiêm trọng hơn.                                                 |
| `NumberOfDependents`                   | Số người phụ thuộc tài chính vào khách hàng, chẳng hạn như con cái hoặc các thành viên khác trong gia đình.                                                                  |

---

## 3. Phân loại các biến

### 3.1. Target variable

| Column             | Loại                         | Giá trị      |
| ------------------ | ---------------------------- | ------------ |
| `SeriousDlqin2yrs` | Binary Classification Target | `0` hoặc `1` |

### 3.2. Numerical features

Các feature còn lại đều là biến số:

| Nhóm              | Columns                                                                                                   |
| ----------------- | --------------------------------------------------------------------------------------------------------- |
| Thông tin cá nhân | `age`, `NumberOfDependents`                                                                               |
| Thu nhập          | `MonthlyIncome`                                                                                           |
| Tín dụng          | `RevolvingUtilizationOfUnsecuredLines`, `NumberOfOpenCreditLinesAndLoans`                                 |
| Nợ                | `DebtRatio`                                                                                               |
| Lịch sử quá hạn   | `NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTimes90DaysLate`, `NumberOfTime60-89DaysPastDueNotWorse` |
| Vay bất động sản  | `NumberRealEstateLoansOrLines`                                                                            |

---

## 4. Cột cần xem xét loại bỏ

### `Unnamed: 0`

`Unnamed: 0` thường chỉ là index được lưu lại từ DataFrame và không đại diện cho đặc điểm thực tế của khách hàng.

Do đó có thể loại bỏ:

```python
df = df.drop(columns=["Unnamed: 0"])
```

Sau khi loại bỏ, dataset sẽ tập trung vào các thông tin thực sự liên quan đến khả năng default của khách hàng.

---

## 5. Định nghĩa bài toán Machine Learning

Bài toán có thể được biểu diễn như sau:

```text
Input X:
    - RevolvingUtilizationOfUnsecuredLines
    - age
    - NumberOfTime30-59DaysPastDueNotWorse
    - DebtRatio
    - MonthlyIncome
    - NumberOfOpenCreditLinesAndLoans
    - NumberOfTimes90DaysLate
    - NumberRealEstateLoansOrLines
    - NumberOfTime60-89DaysPastDueNotWorse
    - NumberOfDependents

Target y:
    - SeriousDlqin2yrs
```

Model sẽ học mối quan hệ giữa các đặc điểm của khách hàng và lịch sử tín dụng để dự đoán:

```text
SeriousDlqin2yrs = 0
→ Khách hàng không có quá hạn nghiêm trọng trong 2 năm.

SeriousDlqin2yrs = 1
→ Khách hàng có quá hạn nghiêm trọng trong 2 năm.
```

---

## 6. Các feature liên quan đến lịch sử quá hạn

Ba feature sau đặc biệt quan trọng vì chúng mô tả trực tiếp lịch sử thanh toán của khách hàng:

```text
NumberOfTime30-59DaysPastDueNotWorse
NumberOfTime60-89DaysPastDueNotWorse
NumberOfTimes90DaysLate
```

Có thể hiểu đơn giản:

```text
30–59 ngày  → Quá hạn ngắn/trung bình
60–89 ngày  → Quá hạn nghiêm trọng hơn
90+ ngày    → Quá hạn rất nghiêm trọng
```

Các feature này có thể giúp model nhận biết những khách hàng có lịch sử tín dụng xấu và có nguy cơ default cao hơn.

---

## 7. Tóm tắt

Dataset cung cấp ba nhóm thông tin chính về khách hàng:

1. **Thông tin cá nhân**

   * `age`
   * `NumberOfDependents`

2. **Thông tin tài chính và tín dụng**

   * `MonthlyIncome`
   * `DebtRatio`
   * `RevolvingUtilizationOfUnsecuredLines`
   * `NumberOfOpenCreditLinesAndLoans`
   * `NumberRealEstateLoansOrLines`

3. **Lịch sử thanh toán**

   * `NumberOfTime30-59DaysPastDueNotWorse`
   * `NumberOfTime60-89DaysPastDueNotWorse`
   * `NumberOfTimes90DaysLate`

Mục tiêu cuối cùng là sử dụng các đặc điểm trên để dự đoán `SeriousDlqin2yrs`, từ đó xác định khách hàng có nguy cơ xảy ra **serious delinquency** trong vòng 2 năm hay không.

