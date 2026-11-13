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
| Quy trình làm sạch | DE_XUAT_TIEN_XU_LY.md |

## 2. Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| raw/cemp_RRI_2007_2025_source.csv | Bản sao nguyên vẹn file nguồn, 2007-12-25 đến 2025-12-31, 631835 dòng |
| splits/train.csv, val.csv, test.csv | Ba tập con theo thời gian (mục 5), trích nguyên văn từ file nguồn |
| metadata/ | Metadata gốc của gói dữ liệu EDI edi.1177.9 và thông tin 15 trạm CEMP |
| stats/ | Bảng thống kê mô tả, sinh bởi scripts/describe.py |
| scripts/ | make_splits.py (tạo ba tập con), verify.py (kiểm chứng), describe.py (thống kê) |
| DE_XUAT_TIEN_XU_LY.md, analysis/ | Quy trình làm sạch dữ liệu và các phân tích làm căn cứ |
| manifest_sha256.json | Mã băm SHA-256 và kích thước các file, dùng để kiểm tra file không bị thay đổi |

## 3. Định dạng dữ liệu

Mỗi file trong splits/ là file CSV, mỗi dòng là một lần đo. Ba dòng đầu của train.csv:

```
datetime_pst,do_mg_l,water_temp_c,ph,spc_us_cm,turbidity_fnu,chl_fluor_ug_l
2019-01-01 00:00:00,8.79,10.1,7.5,842,5.69,1.95
2019-01-01 00:15:00,8.8,10.09,7.5,843,5.82,1.97
2019-01-01 00:30:00,8.82,10.07,7.5,841,5.61,1.75
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

Giá trị được chép nguyên văn từ nguồn: không làm tròn, không điền ô trống, dấu thập phân là dấu chấm. Ô ghi NA là giá trị bị CEMP loại khi thẩm định hoặc không đo được do thiết bị trục trặc. Ví dụ dòng 2019-06-04 08:00:00 có cả sáu biến là NA.

Thiết bị đo là đầu đo đa thông số YSI EXO2 (dùng từ 2016-03-16). CEMP hiệu chuẩn thiết bị trước khi lắp, hiệu chỉnh sai lệch tích lũy khi đo ngoài hiện trường, kiểm tra sau khi thu hồi và rà soát số liệu hằng ngày, hằng tháng.

## 4. Đặc điểm dữ liệu

### 4.1. Thống kê mô tả (2019-01 đến 2025-11)

| Biến | Thiếu % | Trung bình | Độ lệch chuẩn | Nhỏ nhất | P01 | Trung vị | P99 | Lớn nhất |
|---|---|---|---|---|---|---|---|---|
| do_mg_l | 0.22 | 8.14 | 1.31 | 4.12 | 5.65 | 8.10 | 11.16 | 13.76 |
| water_temp_c | 0.21 | 18.33 | 5.80 | 8.02 | 9.09 | 18.12 | 27.54 | 30.16 |
| ph | 0.27 | 7.54 | 0.22 | 6.85 | 7.06 | 7.54 | 8.10 | 9.23 |
| spc_us_cm | 0.44 | 488.1 | 229.9 | 74 | 88 | 460 | 1144 | 1265 |
| turbidity_fnu | 0.36 | 6.15 | 6.28 | 0.36 | 1.27 | 4.31 | 33.08 | 103.9 |
| chl_fluor_ug_l | 0.31 | 3.02 | 1.76 | 0.31 | 0.79 | 2.64 | 9.33 | 29.62 |

P01 và P99 là phân vị 1% và 99%: 1% số giá trị nhỏ hơn P01, 1% số giá trị lớn hơn P99. Thống kê theo từng tập nằm trong stats/feature_stats.csv.

- DO thường thấp hơn mức bão hòa một chút. Độ bão hòa tính từ DO, nhiệt độ và độ mặn có trung vị 85.6% (P01 là 65.4%, P99 là 109.8%), phù hợp với kênh tàu sâu có nhu cầu tiêu thụ oxy cao.
- Nhiệt độ là yếu tố chính quyết định DO: nước càng ấm, oxy hòa tan càng ít (hệ số tương quan âm 0.73). pH biến đổi cùng chiều DO (tương quan 0.56), vì tảo quang hợp vừa sinh oxy vừa tiêu thụ CO2 làm pH tăng.
- Độ dẫn phản ánh lượng nước sông: thấp trong năm nhiều nước (2019, 2023), cao khi dòng chảy kiệt. Độ đục thường thấp nhưng tăng vọt khi có lũ.

### 4.2. Dữ liệu khuyết

Mỗi biến thiếu 0.21 đến 0.44% số dòng; 0.77% số dòng thiếu ít nhất một biến. Phần lớn ô thiếu nằm lẻ tẻ: tính theo dòng (các dòng liên tiếp có ít nhất một biến NA), 96.7% số khoảng trống dài không quá 1 giờ. Phân bố theo từng biến ở DE_XUAT_TIEN_XU_LY.md, mục 5.2.

Chỉ có sáu khoảng trống dài hơn 6 giờ:

| Tập | Bắt đầu | Độ dài | Biến | Bối cảnh |
|---|---|---|---|---|
| train | tháng 8 năm 2019 (ba đợt) | 7.25 đến 7.75 giờ | độ đục | Cảm biến độ đục hỏng |
| val | 2024-09-02 14:00 | 21.25 giờ | cả sáu biến | Bảo trì |
| val | 2025-02-14 03:30 | 99.25 giờ | độ dẫn | Trùng một đợt lũ |
| test | 2025-09-10 09:30 | 24.5 giờ | cả sáu biến | Bảo trì |

Đoạn dữ liệu liên tục không thiếu ô nào dài nhất là 833.6 ngày, nằm trong train.

### 4.3. Diễn biến của DO

- DO ghi với độ chính xác 0.01 mg/L và thay đổi liên tục: chỉ 9.9% số cặp dòng liền kề có giá trị bằng nhau.
- DO dao động theo ngày: thấp nhất khoảng 06:00 (sau một đêm tảo hô hấp), cao nhất khoảng 15:00 (sau nhiều giờ quang hợp). Chênh lệch giữa cao nhất và thấp nhất trong ngày có trung vị 0.84 mg/L; 10% số ngày chênh lệch trên 2.20 mg/L.
- Biến động theo mùa chiếm 78% phương sai của DO. Trung bình từng năm ổn định trong khoảng 7.67 đến 8.75 mg/L, không có dấu hiệu cảm biến bị trôi dần theo thời gian.
- Đợt DO thấp được định nghĩa là khoảng DO dưới ngưỡng kéo dài ít nhất 1 giờ; hai khoảng cách nhau không quá 1 giờ được gộp thành một đợt. Từ 2019 đến 2025 có 320 đợt dưới 6 mg/L và 18 đợt dưới 5 mg/L. Cả 18 đợt dưới 5 mg/L xảy ra trong đợt nắng nóng tháng 7 năm 2024, khi nước ấm tới 28 độ C, vào ban đêm và rạng sáng, mỗi đợt kéo dài 1 đến 7 giờ. Không có giá trị nào dưới 3 mg/L.

### 4.4. Theo mùa

Độ khó trong bảng là sai số RMSE (mg/L) của cách dự báo đơn giản nhất: lấy DO hiện tại làm dự báo cho 6 giờ sau. Sai số càng lớn thì DO thay đổi càng nhiều trong 6 giờ, càng khó dự báo. Số liệu gộp 2019 đến 2025; chi tiết theo tháng trong stats/monthly_dynamics.csv.

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

### 5.2. Số mẫu dùng được

Mỗi mẫu (gọi là cửa sổ) gồm một đoạn dữ liệu quá khứ đưa vào mô hình và đoạn DO cần dự báo ngay sau đó. Ví dụ với cấu hình đầu vào 24 giờ, dự báo 6 giờ: mẫu dự báo lúc 12:00 dùng sáu biến từ 12:00 hôm trước đến 11:45, và dự báo DO từ 12:00 đến 17:45. Mẫu tiếp theo dự báo lúc 12:15, và cứ thế dời 15 phút một lần.

Số mẫu dùng được sau khi làm sạch, trên số mẫu có thể tạo nếu dữ liệu không thiếu ô nào:

| Cấu hình | Train | Val | Test |
|---|---|---|---|
| Đầu vào 24 giờ, dự báo 6 giờ | 189769 trên 189769 | 25465 trên 26185 | 26160 trên 26377 |
| Đầu vào 7 ngày, dự báo 24 giờ | 189121 trên 189121 | 24097 trên 26113 | 25440 trên 26305 |

Các mẫu bị bỏ là các mẫu chạm vào ba khoảng trống dài hơn 8 giờ: hai lần bảo trì (val tháng 9 năm 2024, test tháng 9 năm 2025) và đợt mất độ dẫn trong lũ tháng 2 năm 2025 (val). Khoảng trống ngắn hơn được nội suy tuyến tính (giải thích ở DE_XUAT_TIEN_XU_LY.md, mục 5).

### 5.3. Mức sai số tham chiếu

Hai cách dự báo không cần mô hình, dùng làm mức so sánh cho sai số của các mô hình.

- Giữ nguyên giá trị hiện tại (persistence): DO sau 1 giờ hoặc 6 giờ bằng DO hiện tại.
- Cùng giờ hôm trước: DO dự báo bằng DO cùng giờ ngày hôm trước. Với tầm dự báo đến 24 giờ, cách này cho sai số bằng persistence 24 giờ.

RMSE tính theo mg/L. R2 là tỉ lệ biến động của DO được dự báo giải thích, bằng 1 khi dự báo đúng hoàn toàn.

| Tập | Persistence 1 giờ: RMSE, R2 | Persistence 6 giờ: RMSE, R2 | Cùng giờ hôm trước: RMSE, R2 |
|---|---|---|---|
| train | 0.216, 0.969 | 0.479, 0.850 | 0.311, 0.937 |
| val | 0.233, 0.980 | 0.490, 0.911 | 0.325, 0.961 |
| test | 0.189, 0.955 | 0.392, 0.808 | 0.245, 0.925 |

R2 của persistence 1 giờ đã khoảng 0.97, vì DO thay đổi chậm trong 1 giờ; ở tầm này R2 ít phân biệt được các mô hình. Độ khó tăng ở tầm 6 giờ và trong mùa nóng (mục 4.4).

### 5.4. Quy tắc sử dụng

1. Mọi thông số rút ra từ dữ liệu (ví dụ trung bình và độ lệch chuẩn để chuẩn hóa, ngưỡng xử lý) chỉ tính trên train. Các thí nghiệm so sánh mô hình (dữ liệu gốc, thêm đặc trưng miền, thêm đặc trưng ngữ nghĩa) dùng chung các mẫu và ranh giới tập.
2. Chọn cấu hình trên val. Sau đó huấn luyện lại trên train cộng val (2019-01-01 đến 2025-02-28) và đánh giá trên test đúng một lần.
3. Cảnh báo được đánh giá trên test ở ngưỡng 6 mg/L, vì test không có đợt dưới 5 mg/L; ngưỡng 5 mg/L chỉ báo cáo trên val. Hai ngưỡng này tương ứng với mục tiêu chất lượng nước cho hạ lưu sông San Joaquin: DO 5.0 mg/L quanh năm và 6.0 mg/L từ tháng 9 đến 11 (theo Basin Plan của vùng Central Valley; chưa đối chiếu với phiên bản hiện hành).
4. Báo cáo MAE, RMSE, R2 trên toàn bộ test và theo từng giai đoạn mùa, kèm mức sai số tham chiếu ở mục 5.3.

## 6. Kiểm chứng và tái lập

scripts/verify.py kiểm tra:
- mã SHA-256 của từng file khớp manifest_sha256.json;
- file nguồn đủ 631835 dòng (khớp metadata gốc), các dòng cách đều 15 phút;
- mỗi tập con đúng khoảng thời gian, đúng số dòng, không thiếu dòng;
- từng ô của tập con trùng nguyên văn với file nguồn;
- ba tập nối tiếp nhau, không chồng lấn; số ô NA khớp file nguồn.

Kết quả ngày 2026-11-13: đạt toàn bộ, 0 ô sai khác. Bản sao raw/ có SHA-256 199692356508935d8f6490da111f339d059d4985df900433f925c68d71650398, trùng file tải từ nguồn.

```
python scripts/make_splits.py      # tao lai splits/ tu raw/
python scripts/verify.py           # kiem chung; them --write de ghi lai manifest
python scripts/describe.py         # tao lai stats/, can pandas va numpy
```

## 7. Nguồn và trích dẫn

Nelson, M. M., Tran, A. H. (2026). Sacramento-San Joaquin Bay-Delta Continuous (15 Minute) water quality monitoring data collected by the Continuous Environmental Monitoring Program, DWR, 2005-2025. Phiên bản 9 (edi.1177.9). Environmental Data Initiative. https://doi.org/10.6073/pasta/af06602f2547c66d9a8459f2dd7e4188

Giấy phép Creative Commons Attribution 4.0 (CC BY 4.0): mọi công bố sử dụng dữ liệu phải trích dẫn nguồn.
