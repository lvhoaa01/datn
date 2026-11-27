# Dự báo oxy hòa tan bằng học sâu kết hợp đặc trưng miền và ontology

Tài liệu tổng quan của đồ án: bài toán, thuật ngữ chung, thứ tự đọc tài liệu, kết quả chính, cấu trúc thư mục và cách tái lập.

## 1. Bài toán và cách tiếp cận

Oxy hòa tan (DO, mg/L) là lượng oxy trong nước mà cá và sinh vật thủy sinh dùng để thở. DO xuống thấp (dưới 5 đến 6 mg/L) gây hại cho thủy sinh, nên cần biết trước vài giờ để cảnh báo. Đồ án xây dựng hệ thống web dự báo DO cho trạm quan trắc tự động Mossdale Bridge (MSD) trên sông San Joaquin, California, từ chuỗi đo 15 phút một lần của sáu biến: DO, nhiệt độ nước, pH, độ dẫn điện, độ đục, chlorophyll. Cấu hình chính đọc 24 giờ quá khứ và dự báo DO cho 6 giờ tới.

Ba thí nghiệm dùng chung dữ liệu đã làm sạch, cửa sổ, ranh giới tập, mô hình và cách huấn luyện; chỉ khác cột đầu vào. Chênh lệch kết quả giữa chúng vì vậy đo đúng giá trị của phần đầu vào thêm vào.

**Bảng 1.** Ba thí nghiệm.

| Thí nghiệm | Đầu vào mỗi mốc | Câu hỏi |
|---|---|---|
| E1, dữ liệu gốc | 6 biến đo và 6 cờ điền | Mô hình học được bao nhiêu chỉ từ số đo |
| E2, thêm đặc trưng miền | E1 và 24 đặc trưng miền | Kiến thức miền dạng công thức giúp gì, và mô hình dùng đặc trưng nào |
| E3, thêm đặc trưng ngữ nghĩa | E1 và các đặc trưng ngữ nghĩa | Tri thức dạng luật giúp gì so với dữ liệu gốc (E1) và so với kiến thức dạng công thức (E2) |

Mỗi thí nghiệm được đánh giá bằng sai số trên test (tổng, theo bước dự báo, theo điều kiện, theo tháng, quanh các đợt DO thấp) và bằng phân tích ảnh hưởng của đầu vào lên dự báo. Baseline giữ nguyên DO là mức sàn: mô hình chỉ có giá trị ở các bước dự báo mà nó tốt hơn baseline.

## 2. Thuật ngữ chung

**Bảng 2.** Thuật ngữ dùng thống nhất trong mọi tài liệu. Tên chuẩn của sáu biến ở [01_DATASET_ANALYSIS.md](01_DATASET_ANALYSIS.md) Bảng 2, tên chuẩn của 24 đặc trưng miền và 10 nhóm của chúng ở [03_DOMAIN_FEATURES_ANALYSIS.md](03_DOMAIN_FEATURES_ANALYSIS.md) Bảng 2.

| Thuật ngữ | Nghĩa |
|---|---|
| Mốc | Một dòng dữ liệu tại một thời điểm; các mốc cách nhau 15 phút, 96 mốc mỗi ngày |
| Cờ điền | Cột đi kèm mỗi biến: 0 là số đo thật, 1 là ô trống đã được điền bằng số đo thật gần nhất phía trước |
| Nhân quả | Giá trị tại một mốc chỉ được tính từ số đo tại mốc đó và các mốc trước |
| Thời điểm dự báo | Thời điểm hiện tại của một lần dự báo; mô hình chỉ thấy các mốc trước nó |
| Đầu vào | Đoạn dữ liệu quá khứ kết thúc ngay trước thời điểm dự báo; cấu hình chính dùng 24 giờ (96 mốc) |
| Nhãn | DO đo thật trong khoảng cần dự báo, bắt đầu từ thời điểm dự báo; cấu hình chính dự báo 6 giờ (24 mốc) |
| Cửa sổ | Một cặp đầu vào và nhãn, tức một mẫu học hoặc một mẫu đánh giá; thời điểm dự báo dời 15 phút một lần |
| Bước dự báo | Vị trí trong nhãn: bước 1 là 15 phút sau số đo mới nhất của đầu vào, bước 24 là 6 giờ sau |
| Điểm đánh giá | Một bước dự báo của một cửa sổ mà tại đó DO là số đo thật; sai số được tính trên các điểm này |
| MAE, RMSE, $R^2$ | Chỉ số sai số (công thức ở [04_MODEL_TRAINING_EVALUATION.md](04_MODEL_TRAINING_EVALUATION.md) mục 1): MAE và RMSE (mg/L) càng nhỏ, $R^2$ càng gần 1 thì dự báo càng tốt |
| Train, val, test | Ba tập chia theo thời gian, lần lượt để học, để dừng sớm và chọn siêu tham số, và để chấm một lần |
| Giai đoạn mùa | Lạnh (tháng 11 đến 3), xuân (tháng 4, 5), nóng (tháng 6 đến 9), thu (tháng 10); chia theo nhiệt độ nước trung bình tháng trên train (01 mục 3.3) |
| Phần trăm bão hòa oxy | 100 nhân DO chia DO bão hòa (lượng oxy tối đa nước giữ được khi cân bằng với không khí); trên 100% là nước thừa oxy |
| Biên độ DO trong ngày | DO cao nhất trừ DO thấp nhất của một ngày |
| Đợt DO thấp | Khoảng DO dưới 6 mg/L kéo dài ít nhất 1 giờ; hai khoảng cách nhau không quá 1 giờ được gộp. Ngưỡng theo mục tiêu chất lượng nước của sông San Joaquin (01 mục 3.4) |
| Tháng tảo nở hoa | Tháng có chlorophyll trung bình từ 20 ug/L, ngưỡng theo chỉ số trạng thái dinh dưỡng của Carlson (01 mục 3.5) |
| Đặc trưng miền | Đại lượng tính bằng công thức từ các biến đo và thời điểm đo, dựa trên kiến thức về oxy trong sông; đầu vào thêm của E2 |
| Đặc trưng ngữ nghĩa | Trạng thái và khái niệm suy ra từ ontology (mô hình tri thức OWL) bằng luật có thể truy vết; đầu vào thêm của E3 |
| Baseline giữ nguyên DO | Dự báo mọi bước bằng DO đo mới nhất, không học gì từ dữ liệu |
| Phần trăm số điểm, phần trăm tổng sai số | Khi chia các điểm đánh giá thành nhóm: phần của nhóm trong tổng số điểm, và phần của nhóm trong tổng độ lệch tuyệt đối của mọi điểm (04 mục 1); nhóm có phần trăm tổng sai số lớn hơn hẳn phần trăm số điểm là nơi mô hình yếu |
| Thay đổi MAE | 100 nhân (MAE của E2 chia MAE của E1, trừ 1); âm là E2 sai ít hơn (04 mục 1) |
| Tỉ trọng ảnh hưởng | Mức ảnh hưởng của một nhóm đầu vào (trung bình trên các cửa sổ, nhóm đó đẩy dự báo lệch bao nhiêu mg/L, không kể chiều) chia tổng mức ảnh hưởng của mọi nhóm, nhân 100; các nhóm cộng lại bằng 100%. Dùng để so các nhóm với nhau, không phải phần trăm của giá trị dự báo (04 mục 8.1) |

## 3. Tài liệu

**Bảng 3.** Các tài liệu, theo thứ tự đọc.

| Tài liệu | Nội dung |
|---|---|
| [01_DATASET_ANALYSIS.md](01_DATASET_ANALYSIS.md) | Nguồn dữ liệu, các biến, đặc điểm biến động của DO, đợt DO thấp, chia train, val, test |
| [02_DATA_CLEANING.md](02_DATA_CLEANING.md) | Kiểm tra miền giá trị, điền khoảng trống, cờ điền, điểm DO đột biến |
| [03_DOMAIN_FEATURES_ANALYSIS.md](03_DOMAIN_FEATURES_ANALYSIS.md) | E2: cơ sở lý thuyết, công thức và ví dụ tính của 24 đặc trưng miền; ước lượng hệ số trao đổi khí $k$ |
| [04_MODEL_TRAINING_EVALUATION.md](04_MODEL_TRAINING_EVALUATION.md) | Cửa sổ, mô hình LSTM, huấn luyện, chọn cấu hình, kết quả trên test, ảnh hưởng của đầu vào lên dự báo, so sánh các thí nghiệm |
| 05_SEMANTIC_FEATURES_ANALYSIS.md | E3: đặc trưng ngữ nghĩa (đang xây dựng), cùng cấu trúc với tài liệu 03; kết quả của E3 sẽ được thêm vào tài liệu 04 |
| BAO_CAO_TIEN_DO_DO_AN_TOT_NGHIEP_20261121.docx | Báo cáo tiến độ gửi giảng viên hướng dẫn |

**Bảng 4.** Câu hỏi thường gặp và nơi trả lời.

| Câu hỏi | Nơi đọc |
|---|---|
| Dataset có gì đặc biệt? | 01, phần đầu và mục 3 |
| Các biến thay đổi thế nào qua các năm và theo các sự kiện thật (tảo nở hoa, DO thấp, độ đục cao)? | 01 mục 3.5 |
| Mỗi đặc trưng miền được tính thế nào từ số đo? | 03 mục 3, 4 |
| Mô hình học thế nào qua từng epoch? | 04 mục 6.3 |
| Mô hình sai ở đâu và vì sao? | 04 mục 7 |
| Mô hình dựa vào đầu vào nào, ở thời điểm nào, trong giai đoạn mùa nào? | 04 mục 8 |
| E1 và E2 mạnh, yếu ở đâu? | 04 mục 9 |

## 4. Kết quả chính

**Bảng 5.** Sai số trên test của LSTM, đầu vào 24 giờ, dự báo 6 giờ; huấn luyện lại trên train và val, chấm test một lần, trung bình 3 seed.

| Thí nghiệm | MAE (mg/L) | RMSE (mg/L) | $R^2$ | MAE ở bước 6 giờ (mg/L) |
|---|---|---|---|---|
| Baseline giữ nguyên DO | 0.4511 | 0.7487 | 0.7453 | 0.7275 |
| E1 | 0.2743 | 0.4988 | 0.8868 | 0.3519 |
| E2 | 0.2381 | 0.4150 | 0.9217 | 0.2912 |

Nguồn: artifacts/analysis/error_overall.csv, dòng bo = giu_nguyen, E1, E2 (cột MAE, RMSE, R2); MAE ở bước 6 giờ: error_by_step.csv, dòng buoc = 24 (cột MAE_giu_nguyen, MAE_E1, MAE_E2). Chi tiết ở [04_MODEL_TRAINING_EVALUATION.md](04_MODEL_TRAINING_EVALUATION.md) Bảng 9 và Hình 2.

- Điểm yếu chính của E1 là mùa nóng (tháng 6 đến 9): chiếm 44% số điểm đánh giá của test nhưng 82% tổng sai số của E1 (04 Bảng 10).
- E2 giảm MAE 13.2% (04 Bảng 9), nhiều nhất ở mùa nóng (15.3%, 04 Bảng 10) và khi nước thừa oxy, bão hòa oxy từ 100% (15.4%, 04 Bảng 11); E2 kém E1 ở hai bước dự báo đầu, 15 và 30 phút (04 Hình 2).
- Tổng tỉ trọng ảnh hưởng của 10 nhóm đặc trưng miền trong E2 là 54% ở bước 15 phút và 63% ở bước 6 giờ (04 mục 8.2). Ở bước 6 giờ, E1 dựa gần như bằng nhau vào 6 giờ gần nhất (39%) và vào chính khoảng cần dự báo của hôm qua (40%); E2 dựa 97% vào 6 giờ gần nhất, nơi các đặc trưng trượt 24 giờ đã tóm tắt ngày hôm trước (04 Bảng 14).

Chi tiết: [04_MODEL_TRAINING_EVALUATION.md](04_MODEL_TRAINING_EVALUATION.md) mục 7 đến 9.

## 5. Cấu trúc thư mục

**Bảng 6.** Thư mục của repo.

| Đường dẫn | Nội dung |
|---|---|
| dataset/raw/cemp_MSD_2007_2025_source.csv | Bản sao nguyên vẹn file nguồn, 2007-12-26 đến 2025-12-31, 631735 dòng |
| dataset/base_splits/ | train.csv, val.csv, test.csv đã làm sạch, kèm cờ điền (E1) |
| dataset/base_domain_splits/ | base_splits và 24 đặc trưng miền, cùng dòng, cùng ranh giới (E2) |
| dataset/base_semantic_splits/ | Sẽ có: base_splits và các đặc trưng ngữ nghĩa (E3) |
| dataset/metadata/ | Metadata gốc của gói dữ liệu EDI edi.1177.9 và thông tin 15 trạm CEMP |
| dataset/manifest_sha256.json | Mã băm SHA-256 và kích thước của dữ liệu, tài liệu, mã nguồn và kết quả phân tích dữ liệu |
| docs/ | Tài liệu (mục 3) |
| scripts/ | Mã nguồn (Bảng 8) |
| artifacts/dataset/ | Bảng và hình của phân tích dữ liệu, làm sạch, đặc trưng miền |
| artifacts/runs/ | Các lần huấn luyện: cấu hình, lịch sử huấn luyện, chỉ số; trọng số model.pt và dự báo pred_test.npz chỉ lưu cho các mô hình đã chấm test |
| artifacts/analysis/ | Bảng và hình về quá trình huấn luyện và phân tích trên test |

Mọi file trong artifacts/ do script tạo ra; không có file nào cần sửa tay.

**Bảng 7.** File trong artifacts/, script tạo ra và nơi dùng.

| File trong artifacts/ | Script tạo | Dùng ở |
|---|---|---|
| dataset/split_counts.csv | build_base_splits.py | 02 mục 3.3; 04 mục 2 |
| dataset/feature_stats, gap_stats, do_dynamics, monthly_dynamics, monthly_by_year, split_by_regime, low_do_episodes, diel_profile, daily_series (.csv) | describe_dataset.py | 01 mục 1 đến 4; 02 mục 3, 5 |
| dataset/cleaning_summary_train, do_reversal_points, reversal_effect (.csv) | cleaning_study.py | 02 mục 2, 5 |
| dataset/gap_events.csv, gap_length_histogram.csv, fig_khoang_trong.png | gap_study.py | 02 mục 3.1 |
| dataset/reaeration_k.csv | build_domain_splits.py | 03 mục 4.3; được đọc lại khi tính đặc trưng miền |
| dataset/night_regression, reaeration_k_ci, reaeration_k_by_period, reaeration_k_sensitivity, er_temp_offset, night_steps, domain_example, salinity_check (.csv), fig_k_ban_dem.png, fig_dac_trung_vi_du.png | domain_study.py | 03 mục 3, 4 |
| dataset/fig_dien_bien, fig_ngay_dem_mua, fig_su_kien, fig_ffill, fig_dot_bien (.png), event_windows.csv | figures.py | 01, 02 |
| runs/lstm_24h_6h/tuning/ (24 thư mục và tuning.csv) | tune.py | 04 mục 6.2 |
| runs/lstm_24h_6h/domain/ (3 thư mục) | train.py --features domain | 04 mục 6.4 |
| runs/lstm_24h_6h/final/ (6 thư mục, test.csv, test_tb.csv) | final_train.py | 04 mục 6.4, 7, 8 |
| analysis/training_curves.csv, fig_huan_luyen.png | training_curves.py | 04 mục 6.3 |
| analysis/error_overall, error_by_step, error_by_condition, error_by_month, error_low_do_episodes, error_low_do_points (.csv), fig_test_theo_buoc, fig_sai_so_theo_thang, fig_du_bao_dot_do_thap (.png) | error_analysis.py | 04 mục 7 |
| analysis/attr_*.csv, fig_anh_huong_nhom, fig_anh_huong_theo_tre, fig_anh_huong_theo_mua (.png) | attribution.py | 04 mục 8 |

Trong artifacts/runs/lstm_24h_6h/, thư mục tuning/ chứa lưới dò siêu tham số của E1 (8 cấu hình nhân 3 seed), tên thư mục là siêu tham số và seed, ví dụ lr0.001_h128_do0.3_s0; ba thư mục lr0.001_h128_do0.3_s0, s1, s2 là cấu hình đã chọn, tức ba lần huấn luyện chính của E1. Thư mục domain/ chứa ba lần huấn luyện chính của E2. Thư mục final/ chứa các mô hình huấn luyện lại và chấm test: base_s0 đến s2 là E1, domain_s0 đến s2 là E2. Mỗi thư mục lần chạy gồm config.json (thiết lập), history.csv (sai số theo epoch), metrics.json (chỉ số); model.pt của các lần chạy trên val không được lưu, vì chạy lại cùng seed cho kết quả y hệt.

**Bảng 8.** Các script.

| Script | Việc |
|---|---|
| common.py | Hằng số, tên chuẩn của biến và nhóm đặc trưng, ranh giới tập, cấu hình dự báo, đọc dữ liệu, tạo cửa sổ, tính đặc trưng miền |
| plot_style.py | Kiểu trình bày chung của mọi hình (font, cỡ chữ, màu) |
| build_base_splits.py | Làm sạch: tạo base_splits/ từ raw/ |
| build_domain_splits.py | Ước lượng $k$ trên train, tạo base_domain_splits/ |
| describe_dataset.py, cleaning_study.py, gap_study.py, domain_study.py, figures.py | Thống kê, căn cứ và hình cho tài liệu 01, 02, 03 |
| verify_dataset.py | Kiểm chứng dataset (mục 6.3) |
| train.py | Huấn luyện một mô hình có dừng sớm trên val; --features base (E1) hoặc domain (E2); --check kiểm tra phần dữ liệu |
| tune.py | Dò siêu tham số trên val |
| final_train.py | Huấn luyện lại trên train và val, chấm test một lần |
| training_curves.py | Quá trình huấn luyện: sai số theo epoch trên train và val |
| error_analysis.py, attribution.py | Phân tích sai số và ảnh hưởng của đầu vào trên test |

## 6. Chạy lại và tái lập kết quả

### 6.1. Cài đặt

Môi trường Python (3.12, phiên bản thư viện khóa trong uv.lock) khai báo trong pyproject.toml. Cài một lần bằng `uv sync`. Mọi lệnh chạy từ gốc repo dạng `uv run python scripts/<tên>.py`. Các bước huấn luyện cần GPU NVIDIA (chạy được trên CPU nhưng chậm hơn nhiều). Hình dùng font Roboto; trên máy không có Roboto, hình được vẽ bằng DejaVu Sans, nội dung như nhau.

### 6.2. Script nào tạo ra kết quả nào

Chạy theo thứ tự trong Bảng 9; mỗi script đọc kết quả của các script phía trên.

**Bảng 9.** Thứ tự chạy.

| Bước | Lệnh | Tạo ra | Dùng ở |
|---|---|---|---|
| 1 | build_base_splits.py | dataset/base_splits/; số ô điền và số cửa sổ (artifacts/dataset/split_counts.csv) | 02; 04 mục 2 |
| 2 | build_domain_splits.py | $k$ (artifacts/dataset/reaeration_k.csv), dataset/base_domain_splits/ | 03 mục 4.3, 5 |
| 3 | describe_dataset.py | Thống kê mô tả, theo tháng, theo từng tháng của từng năm; từng đợt DO thấp; chuỗi theo ngày | 01 mục 1 đến 4; 02 mục 3, 5 |
| 4 | cleaning_study.py, gap_study.py | Căn cứ làm sạch: miền giá trị, khoảng trống, điểm đột biến | 02 mục 2, 3, 5 |
| 5 | domain_study.py | Khoảng tin cậy, độ ổn định, độ nhạy của $k$; ví dụ tính; kiểm tra công thức độ mặn; hình của 03 | 03 mục 3, 4 |
| 6 | figures.py | Hình của 01 và 02 | 01, 02 |
| 7 | verify_dataset.py | Kiểm chứng dataset (mục 6.3); --write ghi lại manifest sau khi sửa dữ liệu hoặc tài liệu | README mục 6.3 |
| 8 | tune.py | Lưới siêu tham số của E1 trên val, gồm cả 3 lần chạy E1 với cấu hình đã chọn | 04 mục 6.2 |
| 9 | train.py --features domain --seed 0 (lặp với seed 1, 2) | 3 lần huấn luyện E2 có dừng sớm trên val | 04 mục 6.4 |
| 10 | training_curves.py | Sai số theo epoch trên train và val; --plot chỉ vẽ lại hình từ bảng đã ghi | 04 mục 6.3 |
| 11 | final_train.py --sets base domain | Huấn luyện lại trên train và val, chấm test một lần | 04 mục 6.4, 7.1 |
| 12 | error_analysis.py | Sai số trên test: toàn bộ, theo bước, giai đoạn mùa, trạng thái bão hòa, tháng, đợt DO thấp; baseline giữ nguyên DO | 04 mục 7 |
| 13 | attribution.py | Ảnh hưởng của đầu vào trên test; --plot chỉ vẽ lại hình từ bảng đã ghi | 04 mục 8 |

Mỗi con số trong tài liệu ghi kèm file nguồn, nên sau khi chạy lại có thể so trực tiếp.

### 6.3. Kiểm chứng và tái lập

Dataset: scripts/verify_dataset.py kiểm tra
- mã SHA-256 của từng file khớp manifest_sha256.json;
- file nguồn có kích thước và mã MD5 khớp metadata gốc của EDI, đủ 631735 dòng, các dòng cách đều 15 phút;
- mỗi tập trong base_splits/ đúng khoảng thời gian, đúng số dòng; ba tập nối tiếp, không chồng lấn;
- ô có cờ điền 0 trùng nguyên văn với file nguồn; ô có cờ điền 1 trùng nguyên văn số đo thật gần nhất phía trước (quy tắc được cài đặt lại độc lập để so từng ô);
- số ô theo cờ điền và số cửa sổ khớp tài liệu và artifacts/dataset/split_counts.csv;
- $k$ khớp hồi quy ban đêm tính lại độc lập; base_domain_splits/ giữ nguyên văn các cột của base_splits/, và 24 đặc trưng miền khớp cách tính lại độc lập chỉ bằng thư viện chuẩn Python (lệch không quá 2e-6, do làm tròn 6 chữ số).

Bản sao dataset/raw/ có SHA-256 eb5d072fa1e624bc6878f6aec6eac542b45eef6552f10bdc1765264400166d05 và MD5 9b3482d7f375014273bac0088a8d95d8, trùng MD5 ghi trong metadata EDI. Dựng lại bước 1 đến 6 từ dataset/raw/ trên một bản sao sạch cho các file trùng từng byte với repo (cùng máy, cùng font).

Mô hình: cùng seed trên cùng máy cho kết quả trùng từng bit (kết quả trong tài liệu chạy trên GPU NVIDIA GeForce RTX 4050 Laptop).
- Test chỉ chấm một lần, nên final_train.py bỏ qua mọi thí nghiệm và seed đã có lần chấm trong artifacts/runs/lstm_24h_6h/final/.
- Để kiểm tra tái lập mà không chấm lại, chạy `uv run python scripts/final_train.py --sets base domain --verify`: script huấn luyện lại đúng các lần đã chấm vào một thư mục tạm, so từng bit trọng số và dự báo test với bản đã lưu, in REPRODUCED nếu trùng hết, và không ghi lần chấm mới.
- error_analysis.py và attribution.py chỉ đọc mô hình và dự báo đã lưu, chạy lại cho cùng kết quả.
- Trên máy hoặc phiên bản thư viện khác, kết quả huấn luyện có thể lệch nhỏ vì phép tính trên GPU khác nhau; khi đó so các chỉ số với tài liệu thay vì so từng bit.
