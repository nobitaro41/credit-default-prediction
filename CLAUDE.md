# CLAUDE.md — Credit Default Prediction

## 1. Mục tiêu project
Xây dựng project **supervised learning (phân loại nhị phân)** dự đoán khách hàng có vỡ nợ hay không, dùng làm portfolio gây ấn tượng với nhà tuyển dụng AI/ML engineer.
**Dataset**: Give Me Some Credit (cs-training.csv, 150K rows, target: SeriousDlqin2yrs)
**Chất lượng dữ liệu**: 
- Missing values: MonthlyIncome (19.8%), NumberOfDependents (2.6%)
- Giá trị đặc biệt: 96/98 trong các cột số lần trễ hạn (thiếu dữ liệu)
- Outliers: DebtRatio và MonthlyIncome có các giá trị cực đoan
Yêu cầu chất lượng: code sạch, tái lập được, có test, có API + giao diện demo, chạy được bằng 1 lệnh Docker.

## 2. Stack
- **Ngôn ngữ**: Python 3.11
- **ML**: pandas, numpy, scikit-learn, XGBoost, LightGBM, Optuna (tune), SHAP (giải thích), imbalanced-learn
- **Backend**: FastAPI + Pydantic + Uvicorn
- **Frontend**: Streamlit
- **MLOps**: MLflow (theo dõi thí nghiệm), DVC (version dữ liệu)
- **Chất lượng**: pytest, ruff, pre-commit, GitHub Actions
- **Đóng gói**: Docker, docker-compose

## 3. Cấu trúc thư mục và mục đích từng file

```
credit-default-prediction/
├── README.md                  # Mặt tiền project: bài toán, kết quả, ảnh demo, cách chạy
├── CLAUDE.md                  # File này: hướng dẫn cho Claude Code
├── pyproject.toml             # Dependency + cấu hình ruff, pytest
├── Makefile                   # Lệnh tắt: make setup / train / test / serve / app
├── Dockerfile                 # Đóng gói API thành container
├── docker-compose.yml         # Chạy backend + frontend + MLflow bằng 1 lệnh
├── .pre-commit-config.yaml    # Tự động lint/format trước khi commit
├── .gitignore                 # Loại dữ liệu lớn, model, file tạm khỏi Git
├── dvc.yaml                   # Pipeline: prepare → train → evaluate
│
├── .github/workflows/
│   └── ci.yml                 # CI: chạy lint + test mỗi lần push
│
├── configs/
│   └── config.yaml            # Siêu tham số, đường dẫn, seed, tên cột (tách khỏi code)
│
├── data/
│   ├── raw/                   # Dữ liệu gốc, KHÔNG sửa
│   ├── processed/             # Dữ liệu sạch, đã chia train/val/test
│   └── README.md              # Nguồn dữ liệu, giấy phép, data dictionary
│
├── notebooks/
│   ├── 01_eda.ipynb           # Khám phá dữ liệu kèm nhận xét
│   ├── 02_baseline.ipynb      # Baseline Logistic Regression làm mốc
│   └── 03_error_analysis.ipynb# Phân tích chỗ mô hình sai theo nhóm khách hàng
│
├── src/
│   ├── data/
│   │   ├── load.py            # Đọc dữ liệu, kiểm tra schema
│   │   ├── split.py           # Chia train/val/test (stratified), chống leakage
│   │   └── validate.py        # Kiểm tra chất lượng dữ liệu (kiểu, khoảng giá trị)
│   ├── features/
│   │   └── build_features.py  # Tạo đặc trưng + sklearn Pipeline/ColumnTransformer
│   ├── models/
│   │   ├── train.py           # Huấn luyện nhiều mô hình, log vào MLflow
│   │   ├── tune.py            # Optuna + cross-validation
│   │   ├── evaluate.py        # ROC-AUC, PR-AUC, F1, calibration; chọn ngưỡng theo chi phí
│   │   ├── explain.py         # SHAP toàn cục và theo từng khách hàng
│   │   └── predict.py         # Nạp model và dự đoán, dùng chung cho API và batch
│   ├── api/
│   │   ├── main.py            # FastAPI: /predict, /explain, /health
│   │   ├── schemas.py         # Pydantic schema cho input/output
│   │   └── deps.py            # Nạp model một lần khi khởi động
│   └── utils/
│       ├── logger.py          # Logging chuẩn, không dùng print
│       └── seed.py            # Cố định random seed
│
├── app/
│   └── streamlit_app.py       # UI: nhập thông tin khách hàng → xác suất vỡ nợ + biểu đồ SHAP
│
├── models/
│   ├── model.joblib           # Pipeline đã huấn luyện
│   └── metrics.json           # Số liệu model tốt nhất (README đọc từ đây)
│
├── reports/
│   ├── figures/               # Ảnh ROC, PR, SHAP cho README
│   └── model_card.md          # Mục đích, dữ liệu, giới hạn, rủi ro thiên lệch
│
├── tests/
│   ├── test_features.py       # Pipeline tiền xử lý đúng, không leakage
│   ├── test_model.py          # Xác suất trong [0,1], đúng shape
│   └── test_api.py            # Test endpoint bằng TestClient
│
└── scripts/
    ├── download_data.sh       # Tải dữ liệu tự động
    └── run_pipeline.py        # Chạy toàn bộ prepare → train → evaluate
```

## 4. Lệnh thường dùng
```bash
make setup      # uv sync (cài dependency + pre-commit)
make train      # uv run scripts/run_pipeline.py (chạy pipeline huấn luyện, log MLflow)
make test       # uv run pytest tests/ -v
make serve      # uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
make app        # uv run streamlit run app/streamlit_app.py --server.port 8501
docker-compose up --build   # chạy toàn bộ hệ thống
```

## 5. Quy tắc bắt buộc khi viết code
1. **Không data leakage**: mọi bước fit (imputer, scaler, encoder, SMOTE) chỉ fit trên tập train, và nằm trong sklearn Pipeline. Test set chỉ dùng đánh giá cuối.
2. **Tái lập được**: luôn đặt seed từ `configs/config.yaml`; không hard-code đường dẫn hay siêu tham số trong code.
3. **Metric đúng cho dữ liệu mất cân bằng**: báo cáo PR-AUC, recall, F1, ROC-AUC. Không chỉ dùng accuracy. Chọn ngưỡng quyết định theo chi phí kinh doanh (bỏ sót nợ xấu đắt hơn từ chối nhầm).
4. **Có baseline** (Logistic Regression) và so sánh công bằng với các mô hình khác, cùng split và cùng metric.
5. **Code sạch**: type hints, docstring ngắn cho hàm công khai, dùng `logging` thay `print`, mỗi file một trách nhiệm rõ ràng. Chạy `ruff` trước khi kết thúc task.
6. **Mọi module mới phải có test** tương ứng trong `tests/`.
7. **Không commit** dữ liệu lớn, model nặng, file `.env`, hay secret.
8. **Không sửa** `data/raw/`.

## 6. Thứ tự xây dựng (làm từng bước, xong bước nào chạy test bước đó)
1. Khởi tạo repo: `pyproject.toml`, `.gitignore`, `Makefile`, `configs/config.yaml`, cấu trúc thư mục, `src/utils/`.
2. Dữ liệu: `scripts/download_data.sh` (note: data already provided), `src/data/load.py`, `validate.py`, `split.py` + `data/README.md`.
   - cs-training.csv: 150K rows with labels (SeriousDlqin2yrs)
   - cs-test.csv: 101K rows without labels (for final predictions)
   - All train/val/test splits from cs-training.csv only
3. EDA: `notebooks/01_eda.ipynb` (ghi nhận xét bằng markdown).
4. Features: `src/features/build_features.py` + `tests/test_features.py`.
5. Baseline: `notebooks/02_baseline.ipynb`, rồi `train.py` với Logistic Regression.
6. Mô hình mạnh: thêm Random Forest, XGBoost, LightGBM vào `train.py`, log MLflow.
7. Tune: `tune.py` với Optuna.
8. Đánh giá + giải thích: `evaluate.py`, `explain.py`, xuất ảnh vào `reports/figures/`, ghi `models/metrics.json`.
9. Phân tích lỗi: `notebooks/03_error_analysis.ipynb`.
10. Backend: `src/api/` + `predict.py` + `tests/test_api.py`.
11. Frontend: `app/streamlit_app.py`.
12. Đóng gói: `Dockerfile`, `docker-compose.yml`, `ci.yml`, `.pre-commit-config.yaml`.
13. Tài liệu: `reports/model_card.md`, hoàn thiện `README.md` (bảng so sánh mô hình, ảnh demo, sơ đồ kiến trúc, link demo online).

## 7. Định nghĩa "xong" (Definition of Done)
- `make test` pass, `ruff` không lỗi.
- `docker-compose up` chạy được API + UI + MLflow.
- README có: mô tả bài toán, bảng so sánh mô hình với số liệu thật, ảnh demo, hướng dẫn chạy, kết quả kinh doanh.
- Có `model_card.md` nêu rõ giới hạn và rủi ro.

## 8. Cách làm việc với Claude
- Trước khi code, tóm tắt ngắn kế hoạch cho bước đang làm.
- Mỗi lần chỉ làm **một bước** trong mục 6, rồi dừng để mình kiểm tra.
- Khi cần chọn dataset hoặc thư viện chưa rõ, hỏi mình trước thay vì tự đoán.
- Khi sửa code có sẵn, giữ nguyên style hiện tại, không refactor ngoài phạm vi yêu cầu.