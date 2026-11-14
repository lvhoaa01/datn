# Dataset dự báo oxy hòa tan: trạm CEMP RRI, sông San Joaquin

## 1. Tổng quan

Dataset là chuỗi số liệu quan trắc chất lượng nước đo tự động 15 phút một lần tại trạm Rough and Ready Island (RRI) trên sông San Joaquin, thành phố Stockton, California, Hoa Kỳ (vĩ độ 37.963, kinh độ âm 121.365). Trạm nằm trong kênh tàu biển Stockton, đoạn sông nước ngọt chịu ảnh hưởng của thủy triều. Số liệu do Chương trình Quan trắc Môi trường Liên tục (CEMP) của Sở Tài nguyên Nước California (DWR) thu thập và công bố.

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
| raw/cemp_RRI_2007_2025_source.csv | Bản sao nguyên vẹn file nguồn, 2007-12-25 đến 2025-12-31, 631835 dòng |
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
| scripts/train.py | Huấn luyện mô hình. Hiện có phần chuẩn hóa (trừ trung bình, chia độ lệch chuẩn, khớp trên train) và tạo cửa sổ theo mục 5.2, đầu vào gồm sáu biến và sáu cột cờ; kiểm tra bằng `uv run python scripts/train.py --check` |
| scripts/cleaning_study.py, scripts/causal_imputation_benchmark.py, scripts/gap_threshold_study.py | Các phân tích làm căn cứ cho quy trình làm sạch (LAM_SACH.md) |
| artifacts/dataset/ | Kết quả của các script trên: bảng thống kê và hình |

## 3. Định dạng dữ liệu

Mỗi file trong base_splits/ là file CSV, mỗi dòng là một lần đo, gồm cột thời gian, sáu biến và sáu cột cờ. Ba dòng của train.csv; trong nguồn, hai dòng sau có cả sáu biến trống và được điền bằng số đo của dòng đầu:

```
datetime_pst,do_mg_l,water_temp_c,ph,spc_us_cm,turbidity_fnu,chl_fluor_ug_l,flag_do_mg_l,flag_water_temp_c,flag_ph,flag_spc_us_cm,flag_turbidity_fnu,flag_chl_fluor_ug_l
2019-06-04 07:45:00,7.42,19.86,7.09,97,6.66,2.92,0,0,0,0,0,0
2019-06-04 08:00:00,7.42,19.86,7.09,97,6.66,2.92,1,1,1,1,1,1
2019-06-04 08:15:00,7.42,19.86,7.09,97,6.66,2.92,1,1,1,1,1,1
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

Cột flag_<tên biến> cho biết nguồn gốc giá trị: 0 là số đo thật, chép nguyên văn từ nguồn; 1 là ô trống (hoặc số đo ngoài miền vật lý) được điền bằng số đo thật gần nhất phía trước; -1 là ô cách số đo thật gần nhất phía trước hơn 8 giờ, giá trị ghi NA. Định nghĩa đầy đủ ở LAM_SACH.md, mục 5. Dấu thập phân là dấu chấm.

Trong file nguồn, ô ghi NA là giá trị bị CEMP loại khi thẩm định hoặc không đo được do thiết bị trục trặc. Ví dụ dòng 2019-06-04 08:00:00 trong nguồn có cả sáu biến là NA.

Thiết bị đo là đầu đo đa thông số YSI EXO2 (dùng từ 2016-03-16). CEMP hiệu chuẩn thiết bị trước khi lắp, hiệu chỉnh sai lệch tích lũy khi đo ngoài hiện trường, kiểm tra sau khi thu hồi và rà soát số liệu hằng ngày, hằng tháng.

## 4. Đặc điểm dữ liệu

Thống kê ở mục 4 tính trên dữ liệu trước làm sạch.

### 4.1. Thống kê mô tả (2019-01 đến 2025-11)

| Biến | Thiếu % | Trung bình | Độ lệch chuẩn | Nhỏ nhất | P01 | Trung vị | P99 | Lớn nhất |
|---|---|---|---|---|---|---|---|---|
| do_mg_l | 0.22 | 8.14 | 1.31 | 4.12 | 5.65 | 8.10 | 11.16 | 13.76 |
| water_temp_c | 0.21 | 18.33 | 5.80 | 8.02 | 9.09 | 18.12 | 27.54 | 30.16 |
| ph | 0.27 | 7.54 | 0.22 | 6.85 | 7.06 | 7.54 | 8.10 | 9.23 |
| spc_us_cm | 0.44 | 488.1 | 229.9 | 74 | 88 | 460 | 1144 | 1265 |
| turbidity_fnu | 0.36 | 6.15 | 6.28 | 0.36 | 1.27 | 4.31 | 33.08 | 103.9 |
| chl_fluor_ug_l | 0.31 | 3.02 | 1.76 | 0.31 | 0.79 | 2.64 | 9.33 | 29.62 |

P01 và P99 là phân vị 1% và 99%: 1% số giá trị nhỏ hơn P01, 1% số giá trị lớn hơn P99. Thống kê theo từng tập nằm trong artifacts/dataset/feature_stats.csv.

- DO thường thấp hơn mức bão hòa một chút. Độ bão hòa tính từ DO, nhiệt độ và độ mặn có trung vị 85.6% (P01 là 65.4%, P99 là 109.8%), phù hợp với kênh tàu sâu có nhu cầu tiêu thụ oxy cao.
- Nhiệt độ là yếu tố chính quyết định DO: nước càng ấm, oxy hòa tan càng ít (hệ số tương quan âm 0.73). pH biến đổi cùng chiều DO (tương quan 0.56), vì tảo quang hợp vừa sinh oxy vừa tiêu thụ CO2 làm pH tăng.
- Độ dẫn phản ánh lượng nước sông: thấp trong năm nhiều nước (2019, 2023), cao khi dòng chảy kiệt. Độ đục thường thấp nhưng tăng vọt khi có lũ.

### 4.2. Dữ liệu khuyết

Mỗi biến thiếu 0.21 đến 0.44% số dòng; 0.77% số dòng thiếu ít nhất một biến. Phần lớn ô thiếu nằm lẻ tẻ: tính theo dòng (các dòng liên tiếp có ít nhất một biến NA), 96.7% số khoảng trống dài không quá 1 giờ. Phân bố theo từng biến ở LAM_SACH.md, mục 4.1.

Khoảng trống dài nhất trong train là 7.75 giờ (độ đục, tháng 8 năm 2019, cảm biến hỏng). Chỉ có ba sự kiện dài hơn 8 giờ, ngưỡng điền của quy trình làm sạch, đều nằm trong val và test:

| Tập | Bắt đầu | Độ dài | Biến | Bối cảnh |
|---|---|---|---|---|
| val | 2024-09-02 14:00 | 21.25 giờ | cả sáu biến | Bảo trì |
| val | 2025-02-14 03:30 | 99.25 giờ | độ dẫn | Trùng một đợt lũ |
| test | 2025-09-10 09:30 | 24.5 giờ | cả sáu biến | Bảo trì |

### 4.3. Diễn biến của DO

- DO ghi với độ chính xác 0.01 mg/L và thay đổi liên tục: chỉ 9.9% số cặp dòng liền kề có giá trị bằng nhau.
- DO dao động theo ngày: thấp nhất khoảng 06:00 (sau một đêm tảo hô hấp), cao nhất khoảng 15:00 (sau nhiều giờ quang hợp). Chênh lệch giữa cao nhất và thấp nhất trong ngày có trung vị 0.84 mg/L; 10% số ngày chênh lệch trên 2.20 mg/L.
- Biến động theo mùa chiếm 78% phương sai của DO. Trung bình từng năm ổn định trong khoảng 7.67 đến 8.75 mg/L, không có dấu hiệu cảm biến bị trôi dần theo thời gian.
- Đợt DO thấp được định nghĩa là khoảng DO dưới ngưỡng kéo dài ít nhất 1 giờ; hai khoảng cách nhau không quá 1 giờ được gộp thành một đợt. Từ 2019 đến 2025 có 320 đợt dưới 6 mg/L và 18 đợt dưới 5 mg/L. Cả 18 đợt dưới 5 mg/L xảy ra trong đợt nắng nóng tháng 7 năm 2024, khi nước ấm tới 28 độ C, vào ban đêm và rạng sáng, mỗi đợt kéo dài 1 đến 7 giờ. Không có giá trị nào dưới 3 mg/L.

### 4.4. Theo mùa

Độ khó trong bảng là sai số RMSE (mg/L) của cách dự báo đơn giản nhất: lấy DO hiện tại làm dự báo cho 6 giờ sau. Sai số càng lớn thì DO thay đổi càng nhiều trong 6 giờ, càng khó dự báo. Số liệu gộp 2019 đến 2025; chi tiết theo tháng trong artifacts/dataset/monthly_dynamics.csv.

| Giai đoạn | Tháng | DO trung bình (mg/L) | Chênh lệch trong ngày (mg/L) | Nhiệt độ (độ C) | Độ khó 6 giờ | Số đợt DO dưới 6 |
|---|---|---|---|---|---|---|
| Lạnh, ổn định | 11 đến 3 | 8.54 đến 9.57 | 0.46 đến 0.60 | 10.4 đến 14.6 | 0.21 đến 0.23 | 2 |
| Chuyển mùa xuân | 4, 5 | 8.15 đến 8.50 | 0.73 đến 0.79 | 17.4 đến 20.0 | 0.27 đến 0.35 | 0 |
| Nóng | 6 đến 9 | 6.80 đến 7.37 | 1.26 đến 1.97 | 23.3 đến 26.0 | 0.57 đến 0.79 | 287 |
| Chuyển mùa thu | 10 | 7.62 | 0.90 | 20.2 | 0.40 | 32 |

Mùa nóng khó dự báo gấp khoảng 3 lần mùa lạnh và chứa gần như toàn bộ đợt DO thấp. Thời điểm xảy ra đợt DO thấp khác nhau giữa các năm: 2019 vào tháng 7; 2022 vào tháng 8 và 10; 2023 vào tháng 9; 2025 từ tháng 7 đến 10.

### 4.5. Hạn chế

- Cảng Stockton vận hành một trạm sục khí tại Rough and Ready Island từ năm 2012 để tăng DO quanh khu vực. Dataset không có lịch vận hành, nên DO ở một số giai đoạn có thể cao hơn mức tự nhiên.
- Không có số liệu thủy văn (lưu lượng, mực nước) và khí tượng (bức xạ mặt trời, gió).
- Độ đục ghi theo đơn vị FNU từ năm 2021, trước đó là NTU; theo nhà sản xuất, hai đơn vị tương đương. Metadata nguồn ghi nhầm đơn vị độ dẫn ở phần định nghĩa; đơn vị đúng là uS/cm.
- Dữ liệu trước năm 2019 có nhiều đợt thiếu dài (năm 2018 thiếu pH liên tục 311 giờ) nên không được đưa vào ba tập con.

## 5. Phân chia dữ liệu

### 5.1. Ba tập con

Dữ liệu được chia theo thời gian, không xáo trộn. Nếu xáo trộn, hai đoạn dữ liệu chỉ cách nhau 15 phút (gần như giống hệt nhau) có thể rơi vào hai tập khác nhau, làm kết quả đánh giá tốt hơn thực tế. Ranh giới là 00:00 ngày đầu tháng.

| Tập | Từ | Đến | Số dòng | Tỉ lệ |
|---|---|---|---|---|
| train | 2019-01-01 | 2024-05-31 | 189888 | 78.3% |
| val | 2024-06-01 | 2025-02-28 | 26208 | 10.8% |
| test | 2025-03-01 | 2025-11-30 | 26400 | 10.9% |

Vì mùa nóng khó dự báo nhất và chứa gần như mọi đợt DO thấp, val và test mỗi tập dài 9 tháng và chứa trọn mùa nóng (tháng 6 đến 9) cùng tháng chuyển mùa thu (tháng 10):

| Tập | Số tháng lạnh, xuân, nóng, thu | Số đợt DO dưới 6, dưới 5 | DO trung bình, độ lệch chuẩn | Dòng thiếu ít nhất một biến |
|---|---|---|---|---|
| train | 28, 12, 20, 5 | 152, 0 | 8.30, 1.24 | 0.52% |
| val | 4, 0, 4, 1 | 68, 18 | 7.98, 1.65 | 2.56% |
| test | 2, 2, 4, 1 | 100, 0 | 7.16, 0.90 | 0.76% |

- train: năm năm đầy đủ, gồm cả năm nhiều nước (2019, 2023) và năm khô.
- val: mùa nóng 2024, gồm đợt nắng nóng và toàn bộ 18 đợt DO dưới 5 mg/L. Dùng để chọn cấu hình mô hình và ngưỡng cảnh báo.
- test: năm gần nhất, đủ xuân, hè, thu, có 100 đợt DO dưới 6 mg/L. Mô phỏng việc đưa mô hình vào dùng cho một năm chưa từng thấy.

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
- bỏ cửa sổ có ô cờ -1 trong đầu vào hoặc nhãn;
- bỏ cửa sổ mà nhãn không còn mốc DO cờ 0 nào; sai số chỉ tính trên các mốc nhãn có DO cờ 0.

Số mẫu dùng được trên số mẫu có thể tạo:

| Cấu hình | Train | Val | Test |
|---|---|---|---|
| 24 giờ, 6 giờ | 189769 trên 189769 | 25520 trên 26185 | 26183 trên 26377 |
| 7 ngày, 24 giờ | 189121 trên 189121 | 24161 trên 26113 | 25472 trên 26305 |

Các mẫu bị bỏ là các mẫu chạm vào ô cờ -1, tức phần sau 8 giờ đầu của ba khoảng trống dài ở mục 4.2. Ô trống khác được điền bằng số đo thật gần nhất phía trước, chỉ dùng quá khứ (giải thích ở LAM_SACH.md, mục 4).

Đầu vào của mô hình gồm sáu biến và sáu cột cờ (1 với ô được điền, 0 với số đo thật), để mô hình phân biệt giá trị điền với số đo thật.

### 5.3. Mức sai số tham chiếu

**Mục đích.** Một con số sai số đứng riêng (ví dụ MAE 0.15 mg/L) không cho biết mô hình tốt hay kém. Vì vậy sai số của mô hình được so với sai số của các cách dự báo đơn giản, không cần học từ dữ liệu, gọi là mức tham chiếu (baseline). Mô hình chỉ có giá trị khi sai số thấp hơn mức tham chiếu tốt nhất; khoảng cách giữa hai sai số cho biết phần mô hình thực sự học được.

**Hai cách dự báo tham chiếu.** Gọi thời điểm dự báo là t, mốc đo cuối cùng trước đó là t trừ 15 phút.

| Cách | Tên thường dùng | Giá trị dự báo cho mốc t cộng k | Giả định |
|---|---|---|---|
| Giữ nguyên | Persistence | DO tại t trừ 15 phút, như nhau cho mọi k | DO không đổi trong khoảng dự báo |
| Cùng giờ hôm qua | Seasonal naive, chu kỳ 1 ngày | DO tại t cộng k trừ 24 giờ | Diễn biến DO hôm nay lặp lại hôm qua |

Cả hai chỉ dùng dữ liệu nằm trong đầu vào, nên mô hình nào cũng có sẵn thông tin này.

**Ví dụ** (test, thời điểm dự báo 2025-07-15 12:00, DO lúc 11:45 là 6.46 mg/L):

| Mốc | DO thật | Giữ nguyên | Sai lệch | Cùng giờ hôm qua | Sai lệch |
|---|---|---|---|---|---|
| 12:00 | 6.46 | 6.46 | 0.00 | 6.33 | 0.13 |
| 14:00 | 6.81 | 6.46 | 0.35 | 6.82 | 0.01 |
| 16:00 | 8.09 | 6.46 | 1.63 | 7.99 | 0.10 |
| 17:45 | 7.13 | 6.46 | 0.67 | 6.97 | 0.16 |

Giữ nguyên chỉ đúng ở các bước đầu: buổi chiều tảo quang hợp làm DO tăng, nên sai lệch lớn dần theo tầm dự báo. Cùng giờ hôm qua bắt được nhịp ngày đêm này.

**Cách tính.** Tính trên đúng các cửa sổ ở mục 5.2 và chỉ trên các mốc nhãn có DO cờ 0, giống hệt cách đánh giá mô hình. Sai số lấy trung bình trên mọi bước của nhãn và mọi cửa sổ:
- MAE: trung bình độ lệch tuyệt đối, mg/L;
- RMSE: căn bậc hai của trung bình bình phương độ lệch, mg/L; nặng hơn MAE với các lần sai lớn;
- R2: 1 trừ tỉ số giữa tổng bình phương sai số và tổng bình phương độ lệch của DO thật quanh trung bình của tập; bằng 1 khi dự báo đúng hoàn toàn.

**Kết quả** (train và chi tiết ở artifacts/dataset/baselines.csv):

| Cấu hình | Cách | Val: MAE | Val: RMSE | Val: R2 | Test: MAE | Test: RMSE | Test: R2 |
|---|---|---|---|---|---|---|---|
| 24 giờ, 6 giờ | Giữ nguyên | 0.225 | 0.371 | 0.949 | 0.193 | 0.300 | 0.887 |
| 24 giờ, 6 giờ | Cùng giờ hôm qua | 0.212 | 0.323 | 0.962 | 0.176 | 0.245 | 0.925 |
| 7 ngày, 24 giờ | Giữ nguyên | 0.289 | 0.453 | 0.927 | 0.248 | 0.368 | 0.830 |
| 7 ngày, 24 giờ | Cùng giờ hôm qua | 0.206 | 0.315 | 0.965 | 0.175 | 0.242 | 0.926 |

**Cách đọc.**
- Cùng giờ hôm qua tốt hơn giữ nguyên ở cả hai cấu hình và là mức mô hình cần vượt. Chênh lệch giữa hai cách lớn hơn ở cấu hình 7 ngày, 24 giờ, vì giữ nguyên càng sai khi tầm dự báo càng dài.
- R2 của mức tham chiếu đã cao (0.83 đến 0.97), vì phần lớn biến động của DO trong một tập là biến động theo mùa, cách nào cũng nắm được. Do đó so sánh mô hình nên dựa chủ yếu vào MAE và RMSE.
- Mô hình và các nhóm đặc trưng được so trên cùng mức tham chiếu, nên phần cải thiện so với cùng giờ hôm qua cho biết đóng góp của từng nhóm đặc trưng.

### 5.4. Quy tắc sử dụng

1. Mọi thông số rút ra từ dữ liệu (ví dụ trung bình và độ lệch chuẩn để chuẩn hóa, ngưỡng xử lý) chỉ tính trên train. Các thí nghiệm so sánh mô hình (dữ liệu gốc, thêm đặc trưng miền, thêm đặc trưng ngữ nghĩa) dùng chung các mẫu và ranh giới tập.
2. Chọn siêu tham số mô hình trên val, riêng cho từng cấu hình. Sau đó huấn luyện lại trên train cộng val (2019-01-01 đến 2025-02-28) và đánh giá trên test đúng một lần.
3. Cảnh báo được đánh giá trên test ở ngưỡng 6 mg/L, vì test không có đợt dưới 5 mg/L; ngưỡng 5 mg/L chỉ báo cáo trên val. Hai ngưỡng này tương ứng với mục tiêu chất lượng nước cho hạ lưu sông San Joaquin: DO 5.0 mg/L quanh năm và 6.0 mg/L từ tháng 9 đến 11 (theo Basin Plan của vùng Central Valley; chưa đối chiếu với phiên bản hiện hành).
4. Báo cáo MAE, RMSE, R2 cho từng cấu hình trên toàn bộ test và theo từng giai đoạn mùa, kèm mức sai số tham chiếu ở mục 5.3.

## 6. Kiểm chứng và tái lập

scripts/verify_dataset.py kiểm tra:
- mã SHA-256 của từng file khớp manifest_sha256.json;
- file nguồn đủ 631835 dòng (khớp metadata gốc), các dòng cách đều 15 phút;
- mỗi tập trong base_splits/ đúng khoảng thời gian, đúng số dòng, không thiếu dòng; ba tập nối tiếp nhau, không chồng lấn;
- ô cờ 0 trùng nguyên văn với file nguồn; ô cờ 1 trùng nguyên văn số đo thật gần nhất phía trước, cách không quá 8 giờ; ô cờ -1 ghi NA (quy tắc được cài đặt lại độc lập để so từng ô);
- số ô theo cờ và số mẫu ở mục 5.2 khớp tài liệu.

Kết quả ngày 2026-11-14: đạt toàn bộ. Bản sao raw/ có SHA-256 199692356508935d8f6490da111f339d059d4985df900433f925c68d71650398, trùng file tải từ nguồn.

Môi trường Python (3.12, phiên bản thư viện khóa trong uv.lock) khai báo trong pyproject.toml ở gốc repo. Cài một lần bằng `uv sync`, rồi chạy từ gốc repo theo thứ tự (lệnh đầu tạo base_splits/ mà các lệnh sau dùng):

```
uv run python scripts/build_base_splits.py            # dataset/base_splits/ từ dataset/raw/
uv run python scripts/describe_dataset.py             # thống kê mục 4
uv run python scripts/cleaning_study.py               # căn cứ LAM_SACH.md mục 3, 6
uv run python scripts/causal_imputation_benchmark.py  # căn cứ LAM_SACH.md mục 4.2, 4.3
uv run python scripts/gap_threshold_study.py          # căn cứ LAM_SACH.md mục 4.1, 4.3, 4.5
uv run python scripts/baselines.py                    # mức sai số tham chiếu mục 5.3
uv run python scripts/verify_dataset.py               # kiểm chứng; --write để ghi lại manifest
```

Các lệnh trên ghi đè dataset/base_splits/ và artifacts/dataset/; phần ngẫu nhiên dùng seed cố định. Chạy toàn bộ rồi chạy verify_dataset.py không kèm --write: nếu đạt, mọi file sinh ra trùng từng byte với manifest. Đã thử ngày 2026-11-14 trên một bản sao chỉ gồm file nguồn, metadata, mã và tài liệu: cả 19 file (3 tập trong base_splits/ và 16 file trong artifacts/dataset/) được sinh lại trùng từng byte.

## 7. Nguồn và trích dẫn

Nelson, M. M., Tran, A. H. (2026). Sacramento-San Joaquin Bay-Delta Continuous (15 Minute) water quality monitoring data collected by the Continuous Environmental Monitoring Program, DWR, 2005-2025. Phiên bản 9 (edi.1177.9). Environmental Data Initiative. https://doi.org/10.6073/pasta/af06602f2547c66d9a8459f2dd7e4188

Giấy phép Creative Commons Attribution 4.0 (CC BY 4.0): mọi công bố sử dụng dữ liệu phải trích dẫn nguồn.
