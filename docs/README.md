# Dự báo oxy hòa tan bằng học sâu kết hợp đặc trưng miền và ontology

Tài liệu tổng quan của đồ án: bài toán, cách tiếp cận, thứ tự đọc tài liệu, kết quả chính, cấu trúc thư mục và cách tái lập.

## 1. Bài toán

Oxy hòa tan (DO, mg/L) là lượng oxy trong nước mà cá và sinh vật thủy sinh dùng để thở. DO xuống thấp (dưới 5 đến 6 mg/L) gây hại cho thủy sinh, nên cần biết trước vài giờ để cảnh báo.

Đồ án xây dựng hệ thống web dự báo DO cho trạm quan trắc tự động Mossdale Bridge (MSD) trên sông San Joaquin, California, từ chuỗi đo 15 phút một lần của sáu biến: DO, nhiệt độ nước, pH, độ dẫn điện, độ đục, huỳnh quang chlorophyll. Cấu hình chính: đọc 24 giờ quá khứ, dự báo DO cho 6 giờ tới (24 mốc 15 phút).

## 2. Cách tiếp cận

Ba thí nghiệm dùng chung dữ liệu đã làm sạch, chung cửa sổ, chung ranh giới tập, chung mô hình và cách huấn luyện; chỉ khác cột đầu vào. Chênh lệch kết quả giữa chúng vì vậy đo đúng giá trị của phần đầu vào thêm vào.

| Thí nghiệm | Đầu vào mỗi mốc | Câu hỏi |
|---|---|---|
| E1, dữ liệu gốc | 6 biến đo và 6 cột cờ (cột cho biết ô là số đo thật hay giá trị điền) | Mô hình học được bao nhiêu chỉ từ số đo |
| E2, thêm đặc trưng miền | E1 cộng 24 đại lượng tính bằng công thức từ kiến thức về oxy trong sông (bão hòa, quang hợp, hô hấp, ánh sáng, ...) | Kiến thức miền dạng công thức giúp gì, và mô hình dùng đặc trưng nào |
| E3, thêm đặc trưng ngữ nghĩa | E1 cộng trạng thái và khái niệm suy ra từ ontology (mô hình tri thức OWL) bằng luật có thể truy vết | Tri thức dạng luật giúp gì so với dữ liệu gốc (E1), và so với kiến thức dạng công thức (E2) |

Mỗi thí nghiệm được đánh giá bằng sai số trên test (tổng, theo bước dự báo, theo mùa, theo điều kiện, quanh các đợt DO thấp) và bằng phân tích ảnh hưởng của từng đặc trưng lên dự báo. Baseline đơn giản "giữ nguyên DO" (dự báo mọi bước bằng DO đo mới nhất, không học gì từ dữ liệu) làm mức sàn: mô hình chỉ có giá trị ở các bước dự báo mà nó tốt hơn baseline.

## 3. Tài liệu

Đọc theo thứ tự:

| Tài liệu | Nội dung |
|---|---|
| [01_DATASET_ANALYSIS.md](01_DATASET_ANALYSIS.md) | Nguồn dữ liệu, các biến, đặc điểm biến động của DO, đợt DO thấp, chia train, val, test |
| [02_DATA_CLEANING.md](02_DATA_CLEANING.md) | Kiểm tra miền giá trị, điền khoảng trống, cột cờ, điểm DO đột biến |
| [03_DOMAIN_FEATURES_ANALYSIS.md](03_DOMAIN_FEATURES_ANALYSIS.md) | E2: 24 đặc trưng miền: cơ sở cân bằng oxy, công thức, ước lượng hệ số trao đổi khí k, ví dụ tính trên số thật |
| [04_MODEL_TRAINING_EVALUATION.md](04_MODEL_TRAINING_EVALUATION.md) | Cửa sổ, mô hình LSTM, huấn luyện và quá trình học, dò siêu tham số, quy tắc dùng val và test, kết quả trên test, ảnh hưởng của đầu vào lên dự báo, đối chứng các thí nghiệm |
| 05_SEMANTIC_FEATURES_ANALYSIS.md | E3: đặc trưng ngữ nghĩa (đang xây dựng), cùng cấu trúc với tài liệu 03; kết quả và ảnh hưởng của E3 được thêm vào tài liệu 04 |
| BAO_CAO_TIEN_DO_DO_AN_TOT_NGHIEP_20261121.docx | Báo cáo tiến độ gửi giảng viên hướng dẫn |

Câu hỏi thường gặp và nơi trả lời:

| Câu hỏi | Nơi đọc |
|---|---|
| Dataset có gì đặc biệt? | 01, đầu tài liệu và mục 3 |
| Các biến biến đổi thế nào qua các năm, theo các sự kiện thật (nở hoa, DO thấp, nước lớn)? | 01 mục 3.5 (hình diễn biến 7 năm, hình cận cảnh ba sự kiện) |
| Đặc trưng miền là gì, tính thế nào, có đáng tin không? | 03 mục 2 đến 5 |
| E1 và E2 mạnh, yếu ở đâu; E2 khắc phục được gì và còn hạn chế gì? | 04 mục 9 |
| Mô hình học thế nào qua từng epoch; E2 đánh đổi giữa bước gần và bước xa và vượt E1 ra sao? | 04 mục 6.3 (đường học trên train và val) |
| Vì sao các con số sai số lại như vậy? | 04 mục 7.2 (so với baseline giữ nguyên DO), 7.6 (sai số theo biên độ ngày đêm) |
| Đặc trưng nào ảnh hưởng đến dự báo, ở thời điểm nào, trong trạng thái nào của sông? | 04 mục 8 |

## 4. Kết quả chính

LSTM, đầu vào 24 giờ, dự báo 6 giờ; huấn luyện lại trên train cộng val, chấm test một lần, trung bình 3 seed:

| Thí nghiệm | MAE (mg/L) | RMSE (mg/L) | R2 | MAE bước 6 giờ |
|---|---|---|---|---|
| Baseline giữ nguyên DO | 0.4511 | 0.7487 | 0.7453 | 0.7275 |
| E1 | 0.2743 | 0.4988 | 0.8868 | 0.3519 |
| E2 | 0.2381 | 0.4150 | 0.9217 | 0.2912 |
| E3 | chưa có | | | |

- Điểm yếu chính của E1 là mùa nóng: nước từ 24 độ C chiếm 29% số điểm test nhưng 62% tổng sai số.
- E2 giảm MAE 13.2%, nhiều nhất ở nước từ 24 độ C (15.4%) và khi oxy quá bão hòa từ 130% (21.8%); E2 kém E1 ở hai bước dự báo đầu (15 và 30 phút).
- Đặc trưng miền chiếm 54 đến 63% ảnh hưởng lên dự báo của E2. E1 dự báo xa bằng cách nhìn lại DO cùng giờ hôm qua; E2 dựa vào 3 giờ gần nhất, nơi các đặc trưng trượt 24 giờ đã tóm tắt ngày hôm trước.

Chi tiết: [04_MODEL_TRAINING_EVALUATION.md](04_MODEL_TRAINING_EVALUATION.md) mục 7 đến 9.

## 5. Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| dataset/raw/cemp_MSD_2007_2025_source.csv | Bản sao nguyên vẹn file nguồn, 2007-12-26 đến 2025-12-31, 631735 dòng |
| dataset/base_splits/ | train.csv, val.csv, test.csv đã làm sạch, kèm cờ cho từng ô (E1) |
| dataset/base_domain_splits/ | base_splits cộng 24 đặc trưng miền, cùng dòng, cùng ranh giới (E2) |
| dataset/base_semantic_splits/ | Sẽ có: base_splits cộng đặc trưng ngữ nghĩa, cùng dòng, cùng ranh giới (E3) |
| dataset/metadata/ | Metadata gốc của gói dữ liệu EDI edi.1177.9 và thông tin 15 trạm CEMP |
| dataset/manifest_sha256.json | Mã băm SHA-256 và kích thước của dữ liệu, tài liệu, mã nguồn và kết quả phân tích dữ liệu, để kiểm tra không file nào bị thay đổi |
| docs/ | Tài liệu (mục 3) |
| scripts/ | Mã nguồn (bảng dưới) |
| artifacts/dataset/ | Bảng thống kê và hình của phân tích dữ liệu, làm sạch, đặc trưng miền |
| artifacts/runs/ | Các lần huấn luyện, chia theo mô hình và cấu hình dự báo rồi theo vai trò (bảng dưới): cấu hình, lịch sử huấn luyện, chỉ số; trọng số model.pt chỉ đưa lên git cho các mô hình đã chấm test (kèm dự báo pred_test.npz), vì test chỉ chấm một lần và phân tích sai số, ảnh hưởng cần đúng các mô hình đó |
| artifacts/analysis/ | Bảng và hình về quá trình huấn luyện (train, val) và phân tích sai số, ảnh hưởng của đặc trưng trên test |

Mọi file trong artifacts/ do script tạo ra và được tài liệu dẫn làm nguồn số liệu hoặc hình; không có file nào cần sửa tay:

| File trong artifacts/ | Script tạo | Dùng ở |
|---|---|---|
| dataset/feature_stats, gap_stats, do_dynamics, monthly_dynamics, monthly_by_year, split_by_regime (.csv) | describe_dataset.py | 01 mục 3, 4 |
| dataset/cleaning_summary_train, do_reversal_points, reversal_effect (.csv) | cleaning_study.py | 02 mục 3, 6 |
| dataset/gap_events.csv, gap_length_histogram (.csv, .png) | gap_study.py | 02 mục 4.1 |
| dataset/reaeration_k.csv | build_domain_splits.py | 03 mục 3; được đọc lại khi tính đặc trưng miền |
| dataset/reaeration_k_ci, reaeration_k_by_period, reaeration_k_sensitivity, er_temp_offset, domain_example, salinity_check (.csv), fig_k_ban_dem.png | domain_study.py | 03 mục 3, 4.1, 5 |
| dataset/fig_toan_chuoi, fig_ngay_dem_mua, fig_dien_bien, fig_su_kien, fig_o_thieu, fig_ffill, fig_dot_bien (.png) | figures.py | 01, 02 |
| runs/lstm_24h_6h/tuning/ (24 thư mục và tuning.csv) | tune.py | Lưới dò siêu tham số E1 trên val, 8 cấu hình x 3 seed, 04 mục 6.2. Tên thư mục là siêu tham số và seed, ví dụ lr0.001_h128_do0.3_s0; 3 thư mục lr0.001_h128_do0.3_s0, s1, s2 là cấu hình đã chọn, tức 3 lần huấn luyện chính của E1. tuning.csv là bảng tổng hợp theo cấu hình |
| runs/lstm_24h_6h/domain/ (3 thư mục) | train.py --features domain | 3 lần huấn luyện chính của E2 trên val, cùng siêu tham số với E1, 04 mục 6.4 |
| runs/lstm_24h_6h/final/ (6 thư mục, test.csv, test_tb.csv) | final_train.py | Huấn luyện lại trên train cộng val và chấm test: base_s0 đến s2 là E1, domain_s0 đến s2 là E2, mỗi thư mục kèm model.pt và pred_test.npz; test.csv là chỉ số từng seed, test_tb.csv là trung bình và độ lệch chuẩn; 04 mục 6.4, 7, 8 |
| analysis/training_curves.csv, fig_huan_luyen*.png | training_curves.py | 04 mục 6.3 |
| analysis/error_*.csv, fig_test_theo_buoc, fig_sai_so_*, fig_du_bao_dot_do_thap, fig_e2_so_voi_e1 (.png) | error_analysis.py | 04 mục 7, 9 |
| analysis/attr_*.csv, fig_anh_huong_*, fig_phu_thuoc, fig_vi_du_anh_huong (.png) | attribution.py | 04 mục 8 |

Mỗi thư mục lần chạy gồm config.json (thiết lập), history.csv (sai số theo epoch), metrics.json (chỉ số); model.pt của các lần chạy trên val không lưu, vì chạy lại cùng seed cho kết quả y hệt.

| Script | Việc |
|---|---|
| common.py | Hằng số, ranh giới tập, cấu hình dự báo, đọc dữ liệu, tạo cửa sổ, tính đặc trưng miền |
| build_base_splits.py | Làm sạch: tạo base_splits/ từ raw/ |
| build_domain_splits.py | Ước lượng k trên train, tạo base_domain_splits/ |
| describe_dataset.py, cleaning_study.py, gap_study.py, domain_study.py, figures.py | Thống kê, căn cứ và hình cho tài liệu 01, 02, 03 |
| verify_dataset.py | Kiểm chứng dataset (mục 6.3) |
| train.py | Huấn luyện một mô hình có dừng sớm trên val; `--features base` (E1) hoặc `domain` (E2); `--check` kiểm tra phần dữ liệu |
| tune.py | Dò siêu tham số trên val |
| final_train.py | Huấn luyện lại trên train cộng val và chấm test một lần |
| training_curves.py | Quá trình huấn luyện: sai số theo epoch trên train và val |
| error_analysis.py, attribution.py | Phân tích sai số và ảnh hưởng của đặc trưng trên test |

## 6. Chạy lại và tái lập kết quả

### 6.1. Cài đặt

Môi trường Python (3.12, phiên bản thư viện khóa trong uv.lock) khai báo trong pyproject.toml. Cài một lần bằng `uv sync`. Mọi lệnh chạy từ gốc repo dạng `uv run python scripts/<tên>.py`. Các bước huấn luyện cần GPU NVIDIA (chạy được trên CPU nhưng chậm hơn nhiều).

### 6.2. Script nào tạo ra kết quả nào

Chạy theo thứ tự trong bảng; mỗi script đọc kết quả của các script phía trên.

| Bước | Lệnh | Tạo ra | Dùng ở |
|---|---|---|---|
| 1 | build_base_splits.py | dataset/base_splits/ (làm sạch) | 02 |
| 2 | build_domain_splits.py | k (artifacts/dataset/reaeration_k.csv), dataset/base_domain_splits/ | 03 mục 3, 6 |
| 3 | describe_dataset.py | thống kê mô tả, theo tháng, theo từng tháng của từng năm (artifacts/dataset/*.csv) | 01 mục 3, 4 |
| 4 | cleaning_study.py, gap_study.py | căn cứ làm sạch: miền giá trị, khoảng trống, điểm đột biến | 02 mục 3, 4, 6 |
| 5 | domain_study.py | khoảng tin cậy, độ ổn định, độ nhạy của k; ví dụ tính một dòng; kiểm tra công thức độ mặn; hình hồi quy ban đêm | 03 mục 3, 4.1, 5 |
| 6 | figures.py | hình của 01 và 02 (artifacts/dataset/fig_*.png) | 01, 02 |
| 7 | verify_dataset.py | kiểm chứng dataset (mục 6.3); `--write` ghi lại manifest sau khi sửa dữ liệu hoặc tài liệu | |
| 8 | tune.py | lưới siêu tham số của E1 trên val, mỗi cấu hình 3 seed (artifacts/runs/lstm_24h_6h/tuning/); gồm cả 3 lần chạy E1 với cấu hình đã chọn | 04 mục 6.2 |
| 9 | train.py --features domain --seed 0 (lặp với seed 1, 2) | 3 lần huấn luyện E2 có dừng sớm trên val, cùng siêu tham số (artifacts/runs/lstm_24h_6h/domain/) | 04 mục 6.4 |
| 10 | training_curves.py | sai số theo epoch trên train và val, tổng và từng bước, của E1 và E2 (artifacts/analysis/training_curves.csv, fig_huan_luyen*.png) | 04 mục 6.3 |
| 11 | final_train.py --sets base domain | huấn luyện lại trên train cộng val, chấm test một lần (artifacts/runs/lstm_24h_6h/final/) | 04 mục 6.4, 7.1 |
| 12 | error_analysis.py | sai số trên test theo bước, mùa, tháng, điều kiện, đợt DO thấp; baseline giữ nguyên DO; hình đối chứng E1 và E2 (artifacts/analysis/error_*) | 04 mục 7, 9 |
| 13 | attribution.py | ảnh hưởng của đặc trưng trên test (artifacts/analysis/attr_*) | 04 mục 8 |

Mỗi con số trong tài liệu ghi kèm file nguồn, nên sau khi chạy lại có thể so trực tiếp.

### 6.3. Kiểm chứng và tái lập

Dataset: scripts/verify_dataset.py kiểm tra
- mã SHA-256 của từng file khớp manifest_sha256.json;
- file nguồn có kích thước và mã MD5 khớp metadata gốc của EDI, đủ 631735 dòng, các dòng cách đều 15 phút;
- mỗi tập trong base_splits/ đúng khoảng thời gian, đúng số dòng; ba tập nối tiếp, không chồng lấn;
- ô cờ 0 trùng nguyên văn với file nguồn; ô cờ 1 trùng nguyên văn số đo thật gần nhất phía trước (quy tắc được cài đặt lại độc lập để so từng ô);
- số ô theo cờ và số cửa sổ khớp tài liệu;
- k khớp hồi quy ban đêm tính lại độc lập; base_domain_splits/ giữ nguyên văn các cột của base_splits/, và 24 đặc trưng miền khớp cách tính lại độc lập chỉ bằng thư viện chuẩn Python (lệch không quá 2e-6, do làm tròn 6 chữ số).

Bản sao dataset/raw/ có SHA-256 eb5d072fa1e624bc6878f6aec6eac542b45eef6552f10bdc1765264400166d05 và MD5 9b3482d7f375014273bac0088a8d95d8, trùng MD5 ghi trong metadata EDI. Dựng lại bước 1 đến 6 từ dataset/raw/ trên một bản sao sạch cho các file trùng từng byte với repo.

Mô hình: cùng seed trên cùng máy cho kết quả trùng từng bit (kết quả trong tài liệu chạy trên GPU NVIDIA GeForce RTX 4050 Laptop).
- Test chỉ chấm một lần, nên final_train.py bỏ qua mọi thí nghiệm và seed đã có lần chấm trong artifacts/runs/lstm_24h_6h/final/, để tránh chấm lại sau khi đã thấy kết quả test.
- Để kiểm tra tái lập mà không chấm lại, chạy `uv run python scripts/final_train.py --sets base domain --verify`: script huấn luyện lại đúng các lần đã chấm vào một thư mục tạm, so từng bit trọng số và dự báo test với bản đã lưu, in REPRODUCED nếu trùng hết, và không ghi lần chấm mới.
- error_analysis.py và attribution.py chỉ đọc mô hình và dự báo đã lưu, chạy lại cho cùng kết quả.
- Trên máy hoặc phiên bản thư viện khác, kết quả huấn luyện có thể lệch nhỏ vì phép tính trên GPU khác nhau; khi đó so các chỉ số với tài liệu thay vì so từng bit.
