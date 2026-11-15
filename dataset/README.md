# Dataset dự báo oxy hòa tan: trạm CEMP MSD, sông San Joaquin

## 1. Tổng quan

Dataset là chuỗi số liệu quan trắc chất lượng nước đo tự động 15 phút một lần tại trạm Mossdale Bridge (MSD) trên sông San Joaquin, hạt San Joaquin, California, Hoa Kỳ (vĩ độ 37.786, kinh độ âm 121.306). Trạm nằm ở đoạn sông nước ngọt chịu ảnh hưởng của thủy triều, phía nam vùng châu thổ Sacramento và San Joaquin, thượng lưu thành phố Stockton. Số liệu do Chương trình Quan trắc Môi trường Liên tục (CEMP) của Sở Tài nguyên Nước California (DWR) thu thập và công bố.

Mục đích sử dụng: dự báo nồng độ oxy hòa tan (DO) cho nhiều giờ tới và cảnh báo khi DO xuống thấp.

| Đặc điểm | Giá trị |
|---|---|
| Bước thời gian | 15 phút, không thiếu dòng nào |
| Múi giờ | Giờ chuẩn Thái Bình Dương (PST, UTC trừ 8), không đổi giờ mùa hè |
| Khoảng dùng cho mô hình | 2019-01-01 00:00 đến 2025-11-30 23:45: 242496 dòng, 2526 ngày |
| Biến | DO (biến cần dự báo) và năm biến hóa lý |
| Dữ liệu gốc | Toàn bộ 2007 đến 2025, giữ trong raw/ |
| Dữ liệu đã làm sạch | base_splits/, quy trình ở LAM_SACH.md |

## 2. Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| raw/cemp_MSD_2007_2025_source.csv | Bản sao nguyên vẹn file nguồn, 2007-12-26 đến 2025-12-31, 631735 dòng |
| base_splits/train.csv, val.csv, test.csv | Ba tập con theo thời gian (mục 5) đã làm sạch, kèm cờ cho từng ô; dùng cho thí nghiệm với dữ liệu gốc |
| base_domain_splits/, base_semantic_splits/ | Sẽ có: base_splits cộng đặc trưng miền, cộng đặc trưng ngữ nghĩa; cùng dòng và cùng ranh giới với base_splits |
| metadata/ | Metadata gốc của gói dữ liệu EDI edi.1177.9 và thông tin 15 trạm CEMP |
| LAM_SACH.md | Quy trình làm sạch dữ liệu |
| manifest_sha256.json | Mã băm SHA-256 và kích thước các file của dataset, mã nguồn và kết quả phân tích, dùng để kiểm tra file không bị thay đổi |

Mã nguồn và kết quả phân tích nằm ngoài thư mục dataset, ở gốc repo:

| Đường dẫn | Nội dung |
|---|---|
| scripts/common.py | Hằng số, ranh giới tập, cấu hình dự báo, hàm đọc dữ liệu và tạo cửa sổ |
| scripts/build_base_splits.py, scripts/verify_dataset.py | Tạo base_splits/ từ raw/; kiểm chứng dataset (mục 6) |
| scripts/describe_dataset.py, scripts/baselines.py | Thống kê mô tả (mục 4); mức sai số tham chiếu (mục 5.3) |
| scripts/train.py | Huấn luyện và đánh giá mô hình LSTM, GRU: dự báo trực tiếp DO cho các bước của nhãn, đầu vào gồm sáu biến đã chuẩn hóa và sáu cột cờ; `--check` kiểm tra phần dữ liệu |
| scripts/tune.py | Dò siêu tham số trên val: chạy lưới cấu hình, mỗi cấu hình ba seed, tổng hợp trung bình và độ lệch chuẩn |
| scripts/cleaning_study.py, scripts/causal_imputation_benchmark.py, scripts/gap_study.py | Các phân tích làm căn cứ cho quy trình làm sạch (LAM_SACH.md): miền giá trị và điểm đột biến, benchmark các phương pháp điền, histogram khoảng trống |
| artifacts/dataset/ | Kết quả của các script trên: bảng thống kê và hình |
| artifacts/runs/ | Mỗi lần huấn luyện một thư mục: cấu hình, lịch sử huấn luyện, chỉ số trên val; bảng tổng hợp dò siêu tham số tuning_<mô hình>_<cấu hình>.csv; checkpoint model.pt không đưa lên git |

## 3. Định dạng dữ liệu

Mỗi file trong base_splits/ là file CSV, mỗi dòng là một lần đo, gồm cột thời gian, sáu biến và sáu cột cờ. Ba dòng của train.csv; trong nguồn, dòng giữa có cả sáu biến trống và được điền bằng số đo của dòng đầu (LAM_SACH.md, mục 4):

```
datetime_pst,do_mg_l,water_temp_c,ph,spc_us_cm,turbidity_fnu,chl_fluor_ug_l,flag_do_mg_l,flag_water_temp_c,flag_ph,flag_spc_us_cm,flag_turbidity_fnu,flag_chl_fluor_ug_l
2019-02-05 12:00:00,9.48,11.24,7.56,399,71.91,7.13,0,0,0,0,0,0
2019-02-05 12:15:00,9.48,11.24,7.56,399,71.91,7.13,1,1,1,1,1,1
2019-02-05 12:30:00,9.49,11.27,7.56,395,70.33,7.16,0,0,0,0,0,0
```

| Cột | Tên cột trong nguồn | Đơn vị | Ý nghĩa |
|---|---|---|---|
| datetime_pst | date, time | PST | Thời điểm đo, dạng YYYY-MM-DD HH:MM:SS |
| do_mg_l | dissolvedoxygen | mg/L | Oxy hòa tan, biến cần dự báo |
| water_temp_c | watertemperature | độ C | Nhiệt độ nước |
| ph | ph | đơn vị pH | Độ pH |
| spc_us_cm | spc | uS/cm | Độ dẫn điện quy về 25 độ C, phản ánh lượng muối hòa tan |
| turbidity_fnu | turbidity | FNU | Độ đục, đo bằng ánh sáng hồng ngoại theo ISO 7027 |
| chl_fluor_ug_l | fluorescence | ug/L | Huỳnh quang chlorophyll, chỉ thị gián tiếp lượng tảo |

Cột flag_<tên biến> cho biết nguồn gốc giá trị: 0 là số đo thật, chép nguyên văn từ nguồn; 1 là ô trống (hoặc số đo ngoài miền vật lý) được điền bằng số đo thật gần nhất phía trước; -1 là ô chưa có số đo nào phía trước, giá trị ghi NA (dataset này không có ô nào như vậy). Định nghĩa đầy đủ ở LAM_SACH.md, mục 5. Dấu thập phân là dấu chấm.

Trong file nguồn, ô ghi NA là giá trị bị CEMP loại khi thẩm định hoặc không đo được do thiết bị trục trặc. Ví dụ dòng 2019-02-05 12:15:00 trong nguồn có cả sáu biến là NA.

Thiết bị đo là đầu đo đa thông số YSI EXO2 (dùng tại trạm từ 2016-06-22). CEMP hiệu chuẩn thiết bị trước khi lắp, hiệu chỉnh sai lệch tích lũy khi đo ngoài hiện trường, kiểm tra sau khi thu hồi và rà soát số liệu hằng ngày, hằng tháng.

## 4. Đặc điểm dữ liệu

Thống kê ở mục 4 tính trên dữ liệu trước làm sạch.

### 4.1. Thống kê mô tả (2019-01 đến 2025-11)

| Biến | Thiếu % | Trung bình | Độ lệch chuẩn | Nhỏ nhất | P01 | Trung vị | P99 | Lớn nhất |
|---|---|---|---|---|---|---|---|---|
| do_mg_l | 0.26 | 9.52 | 1.33 | 3.48 | 7.04 | 9.42 | 14.21 | 19.00 |
| water_temp_c | 0.24 | 17.45 | 5.36 | 7.57 | 8.73 | 17.15 | 27.84 | 30.34 |
| ph | 0.35 | 7.73 | 0.39 | 6.92 | 7.03 | 7.67 | 9.03 | 9.44 |
| spc_us_cm | 0.35 | 432.6 | 230.2 | 70 | 78 | 400 | 1005 | 1233 |
| turbidity_fnu | 0.33 | 10.10 | 9.81 | 0.01 | 0.42 | 8.17 | 48.85 | 173.7 |
| chl_fluor_ug_l | 0.29 | 8.62 | 12.09 | 0.10 | 0.54 | 4.48 | 64.73 | 123.5 |

P01 và P99 là phân vị 1% và 99%: 1% số giá trị nhỏ hơn P01, 1% số giá trị lớn hơn P99. Thống kê theo từng tập nằm trong artifacts/dataset/feature_stats.csv.

- DO thường gần mức bão hòa và vượt xa mức này vào mùa hè. Độ bão hòa tính từ DO, nhiệt độ và độ mặn có trung vị 95.3% (P01 là 73.7%, P99 là 175.5%). Mức trên 150% là dấu hiệu tảo nở hoa: tảo quang hợp sinh oxy nhanh hơn lượng oxy thoát ra khí quyển.
- Tảo chi phối DO mạnh hơn nhiệt độ. Trên toàn chuỗi, DO tương quan cùng chiều với chlorophyll (hệ số 0.58) và pH (0.68), vì tảo quang hợp vừa sinh oxy vừa tiêu thụ CO2 làm pH tăng; tương quan với nhiệt độ yếu (âm 0.18), và trong val, test còn đổi sang cùng chiều (0.10 và 0.28), vì mùa nóng cũng là mùa tảo phát triển.
- Độ dẫn phản ánh lượng nước sông: thấp khi nước lớn (trung bình tháng 6 là 200 uS/cm), cao khi dòng chảy kiệt (tháng 12 và tháng 1, 640 đến 660 uS/cm). Độ đục thường thấp nhưng tăng vọt khi có lũ.

### 4.2. Dữ liệu khuyết

Mỗi biến thiếu 0.24 đến 0.35% số dòng; 0.59% số dòng thiếu ít nhất một biến. Phần lớn ô thiếu nằm lẻ tẻ: tính theo dòng (các dòng liên tiếp có ít nhất một biến NA), 91.8% số khoảng trống dài không quá 1 giờ. Phân bố theo từng biến ở LAM_SACH.md, mục 4.1.

Tính theo thời gian, phần lớn dữ liệu trống nằm trong vài lần cả đầu đo ngừng ghi, đều trong train: dài nhất là 64.25 giờ, từ 2020-06-12 13:45; bốn trong năm lần ngừng ghi rơi vào tháng 4 đến tháng 6 năm 2020. Trong val, khoảng trống dài nhất là 6.5 giờ (độ dẫn); trong test là 1.5 giờ. Histogram và danh sách khoảng trống dài ở LAM_SACH.md, mục 4.1.

### 4.3. Diễn biến của DO

- DO ghi với độ chính xác 0.01 mg/L; 16.5% số cặp dòng liền kề có giá trị bằng nhau.
- DO dao động theo ngày: thấp nhất khoảng 07:00 (sau một đêm tảo hô hấp), cao nhất khoảng 16:00 (sau nhiều giờ quang hợp). Chênh lệch giữa cao nhất và thấp nhất trong ngày có trung vị 0.96 mg/L; 10% số ngày chênh lệch trên 3.32 mg/L.
- Trung bình từng tháng giải thích 51.5% phương sai của DO; phần còn lại chủ yếu là dao động trong ngày. Trung bình từng năm nằm trong khoảng 8.91 đến 10.13 mg/L, không có dấu hiệu cảm biến bị trôi dần theo thời gian.
- Mùa hè có các điểm DO lệch mạnh trong đúng một mốc 15 phút rồi quay lại, gọi là điểm DO đột biến một mốc; có 212 điểm như vậy, phần lớn trong tháng 7 và 8 (LAM_SACH.md, mục 6).
- Đợt DO thấp được định nghĩa là khoảng DO dưới ngưỡng kéo dài ít nhất 1 giờ; hai khoảng cách nhau không quá 1 giờ được gộp thành một đợt. Từ 2019 đến 2025 có 13 đợt dưới 6 mg/L và 1 đợt dưới 5 mg/L. Các đợt rơi vào ba giai đoạn nắng nóng: tháng 8 năm 2020 (6 đợt, gồm đợt dưới 5 mg/L ngày 2020-08-24, DO thấp nhất 3.48 mg/L), tháng 9 năm 2022 (2 đợt) và tháng 8 năm 2025 (5 đợt); đều vào ban đêm và buổi sáng, khi nước ấm 25 đến 27.5 độ C. Không có giá trị nào dưới 3 mg/L.

### 4.4. Theo mùa

Độ khó trong bảng là sai số RMSE (mg/L) của cách dự báo đơn giản nhất: lấy DO hiện tại làm dự báo cho 6 giờ sau. Sai số càng lớn thì DO thay đổi càng nhiều trong 6 giờ, càng khó dự báo. Số liệu gộp 2019 đến 2025; chi tiết theo tháng trong artifacts/dataset/monthly_dynamics.csv.

| Giai đoạn | Tháng | DO trung bình (mg/L) | Chênh lệch trong ngày (mg/L) | Nhiệt độ (độ C) | Độ khó 6 giờ | Số đợt DO dưới 6 |
|---|---|---|---|---|---|---|
| Lạnh, ổn định | 11 đến 3 | 9.61 đến 10.15 | 0.49 đến 0.69 | 10.7 đến 14.2 | 0.30 đến 0.55 | 0 |
| Chuyển mùa xuân | 4, 5 | 9.05 đến 9.32 | 0.79 đến 1.01 | 16.9 đến 18.9 | 0.60 đến 0.62 | 0 |
| Nóng | 6 đến 9 | 8.46 đến 10.28 | 1.40 đến 2.98 | 21.3 đến 24.4 | 0.85 đến 1.52 | 13 |
| Chuyển mùa thu | 10 | 8.89 | 0.83 | 18.2 | 0.51 | 0 |

Mùa nóng khó dự báo gấp khoảng 3 lần mùa lạnh và chứa toàn bộ đợt DO thấp. Khó nhất là tháng 7 và 8, khi tảo nở hoa: chlorophyll trung bình tháng 7 là 26 ug/L, gấp 3 đến 8 lần các tháng mùa lạnh, và DO dao động gần 3 mg/L mỗi ngày, vừa vượt bão hòa ban ngày vừa có thể xuống dưới 6 mg/L ban đêm.

### 4.5. Hạn chế

- Không có số liệu thủy văn (lưu lượng, mực nước) và khí tượng (bức xạ mặt trời, gió).
- Đợt DO thấp hiếm: val không có đợt nào dưới 6 mg/L, test có 5 đợt và không có đợt nào dưới 5 mg/L. Nhiều mốc DO thấp nhất của test là điểm đột biến một mốc (LAM_SACH.md, mục 6).
- Val và test không có khoảng trống nào dài hơn 6.5 giờ, nên chất lượng dự báo khi đầu vào chứa đoạn điền dài (nhiều giờ đến vài ngày, như các lần ngừng ghi trong train) không được đo (LAM_SACH.md, mục 4.4).
- Độ đục ghi theo đơn vị FNU từ năm 2021, trước đó là NTU; theo nhà sản xuất, hai đơn vị tương đương. Metadata nguồn ghi nhầm đơn vị độ dẫn ở phần định nghĩa; đơn vị đúng là uS/cm.
- Dữ liệu trước năm 2019 không được đưa vào ba tập con. Năm 2017 thiếu nhiều (chỉ 61.5% số dòng đủ sáu biến); trước năm 2014 mỗi năm có 2 đến 6 đợt trống từ một ngày trở lên; năm 2018 có chất lượng gần với các năm sau (một sự kiện trống 75 giờ) và có thể thêm vào train nếu cần thêm dữ liệu (artifacts/dataset/yearly_quality.csv).

## 5. Phân chia dữ liệu

### 5.1. Ba tập con

Dữ liệu được chia theo thời gian, không xáo trộn. Nếu xáo trộn, hai đoạn dữ liệu chỉ cách nhau 15 phút (gần như giống hệt nhau) có thể rơi vào hai tập khác nhau, làm kết quả đánh giá tốt hơn thực tế. Ranh giới là 00:00 ngày đầu tháng.

| Tập | Từ | Đến | Số dòng | Tỉ lệ |
|---|---|---|---|---|
| train | 2019-01-01 | 2024-05-31 | 189888 | 78.3% |
| val | 2024-06-01 | 2025-02-28 | 26208 | 10.8% |
| test | 2025-03-01 | 2025-11-30 | 26400 | 10.9% |

Vì mùa nóng khó dự báo nhất và chứa mọi đợt DO thấp, val và test mỗi tập dài 9 tháng và chứa trọn mùa nóng (tháng 6 đến 9) cùng tháng chuyển mùa thu (tháng 10):

| Tập | Số tháng lạnh, xuân, nóng, thu | Số đợt DO dưới 6, dưới 5 | DO trung bình, độ lệch chuẩn | Dòng thiếu ít nhất một biến |
|---|---|---|---|---|
| train | 28, 12, 20, 5 | 8, 1 | 9.45, 1.23 | 0.62% |
| val | 4, 0, 4, 1 | 0, 0 | 10.07, 1.63 | 0.71% |
| test | 2, 2, 4, 1 | 5, 0 | 9.40, 1.48 | 0.30% |

- train: năm năm đầy đủ, gồm cả năm nhiều nước (2019, 2023) và năm khô; chứa mọi lần đầu đo ngừng ghi dài.
- val: mùa nóng 2024, tảo nở hoa mạnh (DO lên tới 17.29 mg/L), không có đợt DO thấp. Dùng để chọn cấu hình mô hình.
- test: năm gần nhất, đủ xuân, hè, thu, khó dự báo nhất trong ba tập, có 5 đợt DO dưới 6 mg/L vào tháng 8. Mô phỏng việc đưa mô hình vào dùng cho một năm chưa từng thấy.

Các cách chia khác đã xem xét:

| Cách chia | Lý do không dùng |
|---|---|
| Chia theo năm (57%, 14%, 29% hoặc 71%, 14%, 14%) | Train còn 57% hoặc 71% số dòng, so với 78.3% ở cách chia đang dùng; val và test mỗi tập dài khoảng một năm trở lên |
| Val từ 2024-07 đến 2025-03 | Val thiếu tháng 6 của mùa nóng |
| Val và test chỉ gồm tháng 5 đến 11 của 2024 và 2025 | Dữ liệu không liền mạch giữa các tập |

### 5.2. Cấu hình dự báo và số mẫu

Mỗi mẫu (gọi là cửa sổ) gồm một đoạn dữ liệu quá khứ đưa vào mô hình (đầu vào) và đoạn DO cần dự báo ngay sau đó (nhãn). Ví dụ với cấu hình đầu vào 24 giờ, dự báo 6 giờ: mẫu dự báo lúc 12:00 dùng sáu biến từ 12:00 hôm trước đến 11:45, và dự báo DO từ 12:00 đến 17:45. Mẫu tiếp theo dự báo lúc 12:15, và cứ thế dời 15 phút một lần.

Hai cấu hình được chạy song song:

| Cấu hình | Đầu vào | Nhãn | Vai trò |
|---|---|---|---|
| 24 giờ, 6 giờ | 96 mốc | 24 mốc | Chính: dự báo trong ngày, phục vụ cảnh báo sớm vài giờ |
| 7 ngày, 24 giờ | 672 mốc | 96 mốc | Phụ: dự báo trọn một chu kỳ ngày đêm, mô hình thấy được xu hướng nhiều ngày |

Quy tắc tạo cửa sổ (cài đặt trong scripts/common.py, hàm window_starts):
- nhãn nằm trọn trong một tập; đầu vào được lấy dữ liệu cuối của tập liền trước, vì đó là dữ liệu quá khứ;
- bỏ cửa sổ có ô cờ -1 (chưa có số đo nào phía trước) trong đầu vào hoặc nhãn;
- bỏ cửa sổ mà nhãn không còn mốc DO cờ 0 nào; sai số chỉ tính trên các mốc nhãn có DO cờ 0.

Số mẫu dùng được trên số mẫu có thể tạo:

| Cấu hình | Train | Val | Test |
|---|---|---|---|
| 24 giờ, 6 giờ | 189345 trên 189769 | 26185 trên 26185 | 26377 trên 26377 |
| 7 ngày, 24 giờ | 188896 trên 189121 | 26113 trên 26113 | 26305 trên 26305 |

Mọi ô trống đều được điền bằng số đo thật gần nhất phía trước, chỉ dùng quá khứ (LAM_SACH.md, mục 4). Các mẫu bị bỏ đều thuộc train: đó là các mẫu có nhãn nằm gọn trong một lần đầu đo ngừng ghi dài, nên không còn mốc DO đo thật nào để tính sai số.

Đầu vào của mô hình gồm sáu biến và sáu cột cờ (1 với ô được điền, 0 với số đo thật), để mô hình phân biệt giá trị điền với số đo thật.

### 5.3. Mức sai số tham chiếu

**Mục đích.** Một con số sai số đứng riêng (ví dụ MAE 0.15 mg/L) không cho biết mô hình tốt hay kém. Vì vậy sai số của mô hình được so với sai số của các cách dự báo đơn giản, không cần học từ dữ liệu, gọi là mức tham chiếu (baseline). Mô hình chỉ có giá trị khi sai số thấp hơn mức tham chiếu tốt nhất; khoảng cách giữa hai sai số cho biết phần mô hình thực sự học được.

**Hai cách dự báo tham chiếu.** Gọi thời điểm dự báo là t, mốc đo cuối cùng trước đó là t trừ 15 phút.

| Cách | Tên thường dùng | Giá trị dự báo cho mốc t cộng k | Giả định |
|---|---|---|---|
| Giữ nguyên | Persistence | DO tại t trừ 15 phút, như nhau cho mọi k | DO không đổi trong khoảng dự báo |
| Cùng giờ hôm qua | Seasonal naive, chu kỳ 1 ngày | DO tại t cộng k trừ 24 giờ | Diễn biến DO hôm nay lặp lại hôm qua |

Cả hai chỉ dùng dữ liệu nằm trong đầu vào, nên mô hình nào cũng có sẵn thông tin này.

**Ví dụ** (test, thời điểm dự báo 2025-06-03 12:00, DO lúc 11:45 là 9.35 mg/L):

| Mốc | DO thật | Giữ nguyên | Sai lệch | Cùng giờ hôm qua | Sai lệch |
|---|---|---|---|---|---|
| 12:00 | 9.37 | 9.35 | 0.02 | 9.28 | 0.09 |
| 14:00 | 9.84 | 9.35 | 0.49 | 9.77 | 0.07 |
| 16:00 | 10.14 | 9.35 | 0.79 | 10.05 | 0.09 |
| 17:45 | 10.21 | 9.35 | 0.86 | 10.22 | 0.01 |

Giữ nguyên chỉ đúng ở các bước đầu: buổi chiều tảo quang hợp làm DO tăng, nên sai lệch lớn dần theo tầm dự báo. Cùng giờ hôm qua bắt được nhịp ngày đêm này.

**Cách tính.** Tính trên đúng các cửa sổ ở mục 5.2 và chỉ trên các mốc nhãn có DO cờ 0, giống hệt cách đánh giá mô hình. Sai số lấy trung bình trên mọi bước của nhãn và mọi cửa sổ:
- MAE: trung bình độ lệch tuyệt đối, mg/L;
- RMSE: căn bậc hai của trung bình bình phương độ lệch, mg/L; nặng hơn MAE với các lần sai lớn;
- R2: 1 trừ tỉ số giữa tổng bình phương sai số và tổng bình phương độ lệch của DO thật quanh trung bình của tập; bằng 1 khi dự báo đúng hoàn toàn.

**Kết quả** (train và chi tiết ở artifacts/dataset/baselines.csv):

| Cấu hình | Cách | Val: MAE | Val: RMSE | Val: R2 | Test: MAE | Test: RMSE | Test: R2 |
|---|---|---|---|---|---|---|---|
| 24 giờ, 6 giờ | Giữ nguyên | 0.380 | 0.651 | 0.841 | 0.451 | 0.749 | 0.745 |
| 24 giờ, 6 giờ | Cùng giờ hôm qua | 0.328 | 0.553 | 0.885 | 0.400 | 0.637 | 0.816 |
| 7 ngày, 24 giờ | Giữ nguyên | 0.587 | 0.953 | 0.659 | 0.693 | 1.074 | 0.477 |
| 7 ngày, 24 giờ | Cùng giờ hôm qua | 0.329 | 0.554 | 0.885 | 0.401 | 0.637 | 0.816 |

**Cách đọc.**
- Cùng giờ hôm qua tốt hơn giữ nguyên ở cả hai cấu hình và là mức mô hình cần vượt. Chênh lệch giữa hai cách lớn hơn ở cấu hình 7 ngày, 24 giờ, vì giữ nguyên càng sai khi tầm dự báo càng dài.
- Test khó hơn val (MAE của cùng giờ hôm qua 0.400 so với 0.328), vì mùa nóng 2025 biến động mạnh hơn mùa nóng 2024.
- R2 của cùng giờ hôm qua là 0.82 đến 0.89, nghĩa là cách đơn giản này đã nắm được phần lớn biến động theo mùa và theo ngày. Do đó so sánh mô hình nên dựa chủ yếu vào MAE và RMSE.
- Mô hình và các nhóm đặc trưng được so trên cùng mức tham chiếu, nên phần cải thiện so với cùng giờ hôm qua cho biết đóng góp của từng nhóm đặc trưng.

### 5.4. Quy tắc sử dụng

1. Mọi thông số tính từ giá trị đo (ví dụ trung bình và độ lệch chuẩn để chuẩn hóa, ngưỡng của đặc trưng miền) chỉ tính trên train, để val và test không ảnh hưởng gián tiếp đến mô hình được chấm trên chính chúng. Các thí nghiệm so sánh mô hình (dữ liệu gốc, thêm đặc trưng miền, thêm đặc trưng ngữ nghĩa) dùng chung các mẫu và ranh giới tập.
2. Chọn siêu tham số mô hình trên val, riêng cho từng cấu hình. Sau đó huấn luyện lại trên train cộng val (2019-01-01 đến 2025-02-28) và đánh giá trên test đúng một lần.
3. Cảnh báo DO thấp chỉ đánh giá được trên test, ở ngưỡng 6 mg/L (5 đợt); val không có đợt nào dưới 6 mg/L nên không dùng để chỉnh ngưỡng cảnh báo, và cả val lẫn test đều không có đợt dưới 5 mg/L. Mục tiêu chất lượng nước của Basin Plan vùng Central Valley cho vùng châu thổ là DO không dưới 5.0 mg/L; mức 6.0 mg/L từ tháng 9 đến 11 áp dụng cho đoạn sông qua Stockton, ở hạ lưu trạm (chưa đối chiếu với phiên bản hiện hành). Vì vậy ngưỡng 6 mg/L ở trạm này là ngưỡng cảnh báo sớm, không phải mức pháp lý.
4. Báo cáo MAE, RMSE, R2 cho từng cấu hình trên toàn bộ test và theo từng giai đoạn mùa, kèm mức sai số tham chiếu ở mục 5.3.

## 6. Kiểm chứng và tái lập

scripts/verify_dataset.py kiểm tra:
- mã SHA-256 của từng file khớp manifest_sha256.json;
- file nguồn có kích thước và mã MD5 khớp metadata gốc của EDI, đủ 631735 dòng, các dòng cách đều 15 phút;
- mỗi tập trong base_splits/ đúng khoảng thời gian, đúng số dòng, không thiếu dòng; ba tập nối tiếp nhau, không chồng lấn;
- ô cờ 0 trùng nguyên văn với file nguồn; ô cờ 1 trùng nguyên văn số đo thật gần nhất phía trước; ô cờ -1 ghi NA (quy tắc được cài đặt lại độc lập để so từng ô);
- số ô theo cờ và số mẫu ở mục 5.2 khớp tài liệu.

Kết quả ngày 2026-11-15: đạt toàn bộ. Bản sao raw/ có SHA-256 eb5d072fa1e624bc6878f6aec6eac542b45eef6552f10bdc1765264400166d05 và MD5 9b3482d7f375014273bac0088a8d95d8, trùng MD5 ghi trong metadata EDI.

Môi trường Python (3.12, phiên bản thư viện khóa trong uv.lock) khai báo trong pyproject.toml ở gốc repo. Cài một lần bằng `uv sync`, rồi chạy từ gốc repo theo thứ tự (lệnh đầu tạo base_splits/ mà các lệnh sau dùng):

```
uv run python scripts/build_base_splits.py            # dataset/base_splits/ từ dataset/raw/
uv run python scripts/describe_dataset.py             # thống kê mục 4
uv run python scripts/cleaning_study.py               # căn cứ LAM_SACH.md mục 3, 6
uv run python scripts/causal_imputation_benchmark.py  # căn cứ LAM_SACH.md mục 4.2
uv run python scripts/gap_study.py                    # căn cứ LAM_SACH.md mục 4.1
uv run python scripts/baselines.py                    # mức sai số tham chiếu mục 5.3
uv run python scripts/verify_dataset.py               # kiểm chứng; --write để ghi lại manifest
```
