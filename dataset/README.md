# Dataset dự báo oxy hòa tan: trạm CEMP RRI, sông San Joaquin

## 1. Tổng quan

Dataset gồm chuỗi thời gian quan trắc liên tục chất lượng nước tại trạm Rough and Ready Island (mã RRI), sông San Joaquin, thành phố Stockton, bang California, Hoa Kỳ (vĩ độ 37.963, kinh độ -121.365). Đây là đoạn sông ngọt chịu ảnh hưởng thủy triều, nằm trong kênh tàu biển Stockton (Stockton Deep Water Ship Channel). Dữ liệu do Chương trình Quan trắc Môi trường Liên tục (Continuous Environmental Monitoring Program, CEMP) thuộc Sở Tài nguyên Nước California (DWR) thu thập và công bố.

Mục đích sử dụng: bài toán dự báo nhiều bước nồng độ oxy hòa tan (DO) và cảnh báo nguy cơ DO thấp, với biến đích là DO và năm biến hóa lý đi kèm.

- Bước thời gian: 15 phút, lưới đều, không mất mốc thời gian nào.
- Múi giờ: giờ chuẩn Thái Bình Dương cố định (PST, UTC-8), không áp dụng giờ mùa hè.
- Khoảng dùng cho mô hình: 2019-01-01 00:00 đến 2025-11-30 23:45, gồm 242496 mốc (2526 ngày).
- Dấu thập phân trong tài liệu này là dấu chấm, thống nhất với các file dữ liệu.

## 2. Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| raw/cemp_RRI_2007_2025_source.csv | Bản sao nguyên vẹn file nguồn của trạm RRI, 2007-12-25 đến 2025-12-31, 631835 dòng |
| splits/train.csv, val.csv, test.csv | Ba tập con theo thời gian, trích nguyên văn từ file nguồn |
| metadata/cemp_edi_1177_9_eml.xml | Metadata gốc theo chuẩn EML của gói dữ liệu EDI edi.1177.9 |
| metadata/cemp_station_metadata.csv | Thông tin 15 trạm CEMP (vị trí, loại sinh cảnh, ngày lắp cảm biến) |
| stats/ | Các bảng thống kê dùng trong tài liệu này, sinh bởi scripts/describe.py |
| scripts/make_splits.py | Tạo ba tập con từ file nguồn, chỉ dùng thư viện chuẩn Python |
| scripts/verify.py | Kiểm chứng toàn vẹn dữ liệu, chỉ dùng thư viện chuẩn Python |
| scripts/describe.py | Tính thống kê mô tả, cần pandas và numpy |
| manifest_sha256.json | Mã băm SHA-256 và kích thước của các file |

## 3. Định dạng dữ liệu

Các file trong splits/ có cùng cấu trúc: một dòng tiêu đề, mỗi dòng sau là một mốc 15 phút.

| Cột | Cột nguồn | Đơn vị | Mô tả |
|---|---|---|---|
| datetime_pst | date, time | PST | Thời điểm đo, định dạng YYYY-MM-DD HH:MM:SS |
| do_mg_l | dissolvedoxygen | mg/L | Nồng độ oxy hòa tan, biến đích |
| water_temp_c | watertemperature | độ C | Nhiệt độ nước |
| ph | ph | đơn vị pH | Độ pH |
| spc_us_cm | spc | microsiemens/cm (uS/cm) | Độ dẫn điện riêng quy về 25 độ C |
| turbidity_fnu | turbidity | FNU | Độ đục, đo bằng nguồn sáng hồng ngoại theo ISO 7027 |
| chl_fluor_ug_l | fluorescence | ug/L | Huỳnh quang chlorophyll quy đổi tương đương nồng độ |

Giá trị khuyết được ghi là NA. Theo metadata nguồn, NA là giá trị không vượt qua quy trình thẩm định của CEMP hoặc không thu được do thiết bị trục trặc. Giá trị số được chép nguyên văn từ nguồn: không làm tròn, không nội suy, không chuẩn hóa.

Thiết bị: đầu đo đa thông số YSI EXO2 (trạm RRI dùng EXO2 từ 2016-03-16, trước đó dùng YSI 6600). Quy trình chất lượng của CEMP gồm hiệu chuẩn trước khi lắp, tính trôi hiện trường, kiểm tra độ chính xác sau khi thu hồi, rà soát trực quan hằng ngày và hằng tháng.

## 4. Đặc điểm dữ liệu

### 4.1. Độ đầy đủ và cấu trúc khuyết

Trên toàn khoảng 2019-01 đến 2025-11, mỗi biến thiếu từ 0.21% (nhiệt độ) đến 0.44% (độ dẫn). Có 1860 mốc (0.77%) thiếu ít nhất một biến, phân bố trong 854 đợt. 826 đợt (96.7%) dài không quá 1 giờ, tức khuyết chủ yếu rời rạc. Sau khi nội suy tuyến tính các đợt khuyết không quá 1 giờ có giá trị hai đầu, 99.63% số mốc đủ cả sáu biến.

Chỉ có sáu đợt khuyết dài hơn 6 giờ:

| Tập | Bắt đầu | Độ dài | Biến bị thiếu |
|---|---|---|---|
| train | 2019-08-16 14:15 | 7.75 giờ | độ đục |
| train | 2019-08-18 02:15 | 7.75 giờ | độ đục |
| train | 2019-08-19 03:15 | 7.25 giờ | độ đục |
| val | 2024-09-02 14:00 | 21.25 giờ | cả sáu biến (bảo trì trạm) |
| val | 2025-02-14 03:30 | 99.25 giờ | độ dẫn |
| test | 2025-09-10 09:30 | 24.5 giờ | cả sáu biến (bảo trì trạm) |

Đoạn liên tục đủ sáu biến dài nhất (sau nội suy khuyết ngắn) kéo dài 833.6 ngày, nằm trong tập train.

### 4.2. Thống kê mô tả các biến (2019-01 đến 2025-11)

| Biến | Thiếu % | Trung bình | Độ lệch chuẩn | Min | P01 | P25 | Trung vị | P75 | P99 | Max |
|---|---|---|---|---|---|---|---|---|---|---|
| do_mg_l | 0.22 | 8.14 | 1.31 | 4.12 | 5.65 | 7.14 | 8.10 | 9.11 | 11.16 | 13.76 |
| water_temp_c | 0.21 | 18.33 | 5.80 | 8.02 | 9.09 | 12.81 | 18.12 | 23.97 | 27.54 | 30.16 |
| ph | 0.27 | 7.54 | 0.22 | 6.85 | 7.06 | 7.40 | 7.54 | 7.68 | 8.10 | 9.23 |
| spc_us_cm | 0.44 | 488.1 | 229.9 | 74 | 88 | 325 | 460 | 641 | 1144 | 1265 |
| turbidity_fnu | 0.36 | 6.15 | 6.28 | 0.36 | 1.27 | 2.79 | 4.31 | 7.20 | 33.08 | 103.9 |
| chl_fluor_ug_l | 0.31 | 3.02 | 1.76 | 0.31 | 0.79 | 1.95 | 2.64 | 3.60 | 9.33 | 29.62 |

P01, P25, P75, P99 là các phân vị 1, 25, 75, 99 phần trăm. Thống kê đầy đủ theo từng tập nằm trong stats/feature_stats.csv.

Diễn giải theo thực tế:

- DO dao động 4.12 đến 13.76 mg/L. Độ bão hòa oxy tính lại từ nhiệt độ và độ mặn (quy đổi từ độ dẫn) có trung vị 85.6%, phân vị 1 và 99 lần lượt 65.4% và 109.8%. Nước tại trạm thường hơi thiếu bão hòa, phù hợp với kênh tàu sâu có nhu cầu oxy cao. Không có giá trị âm hay bão hòa phi vật lý.
- Nhiệt độ 8 đến 30 độ C, là động lực chính của DO qua độ hòa tan (hệ số tương quan với DO bằng -0.73).
- pH 6.85 đến 9.23, biến đổi cùng chiều với DO (tương quan 0.56) do quang hợp tiêu thụ CO2 làm pH tăng và hô hấp làm pH giảm.
- Độ dẫn 74 đến 1265 uS/cm, phản ánh chế độ thủy văn: thấp trong năm nhiều nước (2019, 2023), cao khi dòng chảy kiệt và nước hồi quy nông nghiệp chiếm ưu thế.
- Độ đục có phân bố lệch phải mạnh (trung vị 4.31, max 103.9 FNU), tăng đột biến theo lũ và tàu bè.
- Huỳnh quang chlorophyll là chỉ thị gián tiếp sinh khối tảo, không phải nồng độ chlorophyll xét nghiệm.

### 4.3. Đặc điểm động học của DO

- Độ phân giải 0.01 mg/L. Chỉ 9.9% cặp giá trị liền kề bằng nhau, tín hiệu mịn, không bị lượng tử hóa.
- Chu kỳ ngày rõ: trung bình nhiều năm thấp nhất khoảng 06:00 (7.98 mg/L) và cao nhất khoảng 15:00 (8.43 mg/L), đúng nhịp quang hợp ban ngày và hô hấp ban đêm. Biên độ ngày có trung vị 0.84 mg/L, phân vị 90 là 2.20 mg/L.
- Biến động theo mùa chiếm 78% phương sai của DO (tính bằng phương sai của trung bình từng tháng).
- Sự kiện DO thấp được đếm theo đợt: đoạn liên tục dưới ngưỡng dài ít nhất 1 giờ, các đoạn cách nhau không quá 1 giờ được gộp. Toàn khoảng có 320 đợt DO dưới 6 mg/L và 18 đợt dưới 5 mg/L. Cả 18 đợt dưới 5 mg/L xảy ra trong tháng 7 năm 2024, giai đoạn nắng nóng, nước 28 độ C, chủ yếu ban đêm và rạng sáng, mỗi đợt dài 1 đến 7 giờ. Không có giá trị dưới 3 mg/L.

### 4.4. Diễn biến theo tháng và các chế độ mùa

Thống kê gộp các năm 2019 đến 2025 (file stats/monthly_dynamics.csv). Độ khó dự báo được đo bằng RMSE của dự báo giữ nguyên giá trị hiện tại cho 6 giờ sau. Số đợt được đếm trong từng tháng, nên một đợt kéo dài qua ranh giới tháng được tính ở cả hai tháng.

| Tháng | Chế độ | DO trung bình | Biên độ ngày | Nhiệt độ | Chlorophyll | Độ dẫn | Độ khó 6 giờ | Đợt DO dưới 6 | Đợt DO dưới 5 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | lạnh | 9.06 | 0.55 | 10.4 | 3.26 | 660 | 0.23 | 0 | 0 |
| 2 | lạnh | 9.57 | 0.57 | 11.6 | 3.76 | 584 | 0.23 | 1 | 0 |
| 3 | lạnh | 9.33 | 0.60 | 14.0 | 3.53 | 533 | 0.21 | 0 | 0 |
| 4 | chuyển mùa xuân | 8.50 | 0.73 | 17.4 | 3.09 | 493 | 0.27 | 0 | 0 |
| 5 | chuyển mùa xuân | 8.15 | 0.79 | 20.0 | 2.80 | 360 | 0.35 | 0 | 0 |
| 6 | nóng | 7.37 | 1.26 | 23.3 | 3.17 | 340 | 0.57 | 6 | 0 |
| 7 | nóng | 6.80 | 1.97 | 26.0 | 3.67 | 358 | 0.79 | 108 | 18 |
| 8 | nóng | 6.94 | 1.60 | 25.8 | 3.17 | 426 | 0.79 | 130 | 0 |
| 9 | nóng | 7.13 | 1.43 | 24.3 | 2.74 | 525 | 0.69 | 43 | 0 |
| 10 | chuyển mùa thu | 7.62 | 0.90 | 20.2 | 2.54 | 494 | 0.40 | 32 | 0 |
| 11 | lạnh | 8.54 | 0.46 | 14.6 | 2.03 | 466 | 0.22 | 1 | 0 |
| 12 | lạnh | 8.89 | 0.50 | 11.1 | 2.48 | 649 | 0.21 | 0 | 0 |

Bốn chế độ mùa:

1. Lạnh và ổn định (tháng 11 đến tháng 3): DO cao 8.5 đến 9.6 mg/L, biên độ ngày khoảng 0.5 mg/L, dễ dự báo.
2. Chuyển mùa xuân (tháng 4, 5): nước ấm dần, DO giảm, độ khó tăng.
3. Nóng và tới hạn (tháng 6 đến tháng 9): nhiệt độ 23 đến 26 độ C, DO thấp nhất năm, biên độ ngày 1.3 đến 2.0 mg/L, khó dự báo gấp khoảng 3 lần mùa lạnh, chứa gần như toàn bộ sự kiện DO thấp.
4. Chuyển mùa thu (tháng 10): DO hồi phục nhưng vẫn còn sự kiện DO thấp ở một số năm.

Thời điểm xảy ra sự kiện thay đổi giữa các năm: năm 2019 tập trung tháng 7, năm 2022 tháng 8 và tháng 10, năm 2023 tháng 9, năm 2025 trải từ tháng 7 đến tháng 10.

### 4.5. Diễn biến theo năm

| Năm | DO trung bình | Độ lệch chuẩn | Min | Max |
|---|---|---|---|---|
| 2019 | 8.75 | 0.99 | 5.28 | 13.76 |
| 2020 | 8.26 | 1.51 | 5.22 | 11.81 |
| 2021 | 8.07 | 1.04 | 5.60 | 10.83 |
| 2022 | 7.97 | 1.39 | 5.42 | 11.53 |
| 2023 | 8.20 | 1.15 | 5.40 | 12.25 |
| 2024 | 8.03 | 1.28 | 4.12 | 10.50 |
| 2025 (đến tháng 11) | 7.67 | 1.44 | 5.28 | 11.91 |

Mức DO trung bình ổn định giữa các năm, không có dấu hiệu trôi cảm biến. Chỗ giao giữa các năm liền mạch, không có khuyết, chênh lệch DO giữa hai mốc liên tiếp tại ranh giới không quá 0.10 mg/L.

### 4.6. Lưu ý khi sử dụng

- Can thiệp nhân tạo: tại Rough and Ready Island có trạm sục khí do DWR xây dựng năm 2006 đến 2007 và do Cảng Stockton vận hành từ năm 2012, được ghi nhận là làm tăng DO quanh trạm và phía hạ lưu. Dataset không có lịch vận hành trạm sục khí. Giá trị DO là giá trị đo thực, nhưng có thể bị nâng lên trong một số giai đoạn bởi yếu tố không có trong các biến quan trắc.
- Không có biến thủy văn (lưu lượng, mực nước, vận tốc) và biến khí tượng (bức xạ, gió).
- Độ đục được báo cáo theo đơn vị FNU từ năm 2021, trước đó ghi là NTU. Theo nhà sản xuất cảm biến, hai đơn vị tương đương và không cần quy đổi.
- Phần định nghĩa trong metadata nguồn ghi nhầm đơn vị độ dẫn là microgram/lít; đơn vị đúng là microsiemens/cm.
- Khoảng 2019-01 đến 2025-11 được chọn vì các năm trước đó có nhiều khuyết dài (ví dụ năm 2018 thiếu pH liên tục 311 giờ và nhiều đợt thiếu độ đục 6 đến 9 giờ). Tháng 12 năm 2025 không đưa vào các tập con. Toàn bộ dữ liệu 2007 đến 2025 vẫn được giữ trong raw/.

## 5. Phân chia dữ liệu

### 5.1. Nguyên tắc

Dữ liệu được chia theo trình tự thời gian, không xáo trộn, vì hai cửa sổ liền kề gần như trùng nhau; chia ngẫu nhiên sẽ làm rò rỉ thông tin từ tương lai vào quá trình huấn luyện. Ranh giới được đặt tại 00:00 PST ngày đầu tháng.

Kích thước tập val và test được chọn dựa trên diễn biến mùa ở mục 4.4. Vì mùa nóng khó dự báo gấp khoảng 3 lần mùa lạnh và chứa gần như toàn bộ sự kiện DO thấp, một tập đánh giá ngắn hơn 12 tháng chỉ có ý nghĩa khi chứa trọn các tháng 6 đến 10. Mỗi tập val và test vì vậy dài 9 tháng, gồm đúng một mùa nóng trọn vẹn cùng các tháng chuyển mùa liền kề. Tập train gồm năm mùa nóng.

### 5.2. Các tập con

| Tập | Từ | Đến | Số ngày | Số dòng | Tỉ lệ | Tháng 6 đến 10 | Đợt DO dưới 6 | Đợt DO dưới 5 | DO trung bình | DO độ lệch chuẩn | DO min | DO max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 2019-01-01 00:00 | 2024-05-31 23:45 | 1978 | 189888 | 78.3% | 25 | 152 | 0 | 8.30 | 1.24 | 5.22 | 13.76 |
| val | 2024-06-01 00:00 | 2025-02-28 23:45 | 273 | 26208 | 10.8% | 5 | 68 | 18 | 7.98 | 1.65 | 4.12 | 11.91 |
| test | 2025-03-01 00:00 | 2025-11-30 23:45 | 275 | 26400 | 10.9% | 5 | 100 | 0 | 7.16 | 0.90 | 5.28 | 9.72 |

Tỉ lệ tính theo số dòng. Ba tập nối tiếp nhau, không chồng lấn và không có khoảng hở.

Vai trò từng tập theo diễn biến thực tế:

- train (65 tháng): năm chu kỳ năm đầy đủ, gồm cả năm nhiều nước (2019, 2023) và năm khô, đủ để học chu kỳ ngày, chu kỳ mùa và quan hệ giữa các biến.
- val (9 tháng): mùa nóng 2024 chứa đợt nắng nóng tháng 7 và toàn bộ 18 đợt DO dưới 5 mg/L, sau đó là mùa thu và mùa đông. Dùng để chọn siêu tham số và hiệu chỉnh ngưỡng cảnh báo.
- test (9 tháng): năm gần nhất, đi trọn chu trình xuân, hè, thu (tháng 3 đến tháng 11 năm 2025) với 100 đợt DO dưới 6 mg/L, nhiều nhất trong bảy năm. Mô phỏng tình huống triển khai thực tế trên năm chưa từng thấy.

Phân bố các chế độ mùa trong từng tập (file stats/split_by_regime.csv):

| Tập | Chế độ | Số tháng | DO trung bình | DO độ lệch chuẩn | Độ khó 6 giờ |
|---|---|---|---|---|---|
| train | lạnh | 28 | 9.10 | 1.02 | 0.23 |
| train | chuyển mùa xuân | 12 | 8.43 | 0.88 | 0.31 |
| train | nóng | 20 | 7.28 | 0.88 | 0.76 |
| train | chuyển mùa thu | 5 | 7.64 | 0.94 | 0.41 |
| val | lạnh | 4 | 9.48 | 0.94 | 0.18 |
| val | nóng | 4 | 6.55 | 0.90 | 0.68 |
| val | chuyển mùa thu | 1 | 7.79 | 0.69 | 0.43 |
| test | lạnh | 2 | 7.99 | 0.76 | 0.16 |
| test | chuyển mùa xuân | 2 | 7.66 | 0.37 | 0.34 |
| test | nóng | 4 | 6.44 | 0.52 | 0.51 |
| test | chuyển mùa thu | 1 | 7.36 | 0.85 | 0.28 |

Tập val không có tháng chuyển mùa xuân; tập test có. Mùa nóng chiếm 4 trên 9 tháng ở mỗi tập val và test.

### 5.3. Độ đầy đủ theo tập

| Tập | Mốc thiếu ít nhất một biến | Số đợt khuyết | Đợt không quá 1 giờ | Đợt dài hơn 6 giờ | Đợt khuyết dài nhất | Đủ sáu biến sau nội suy khuyết không quá 1 giờ | Đoạn liên tục dài nhất |
|---|---|---|---|---|---|---|---|
| train | 989 (0.52%) | 664 | 648 | 3 | 7.75 giờ | 99.87% | 833.6 ngày |
| val | 670 (2.56%) | 107 | 98 | 2 | 99.25 giờ | 97.93% | 93.6 ngày |
| test | 201 (0.76%) | 83 | 80 | 1 | 24.5 giờ | 99.56% | 96.3 ngày |

Tỉ lệ khuyết cao hơn của val chủ yếu do đợt thiếu độ dẫn 99.25 giờ từ 2025-02-14.

### 5.4. Số cửa sổ mẫu

Cửa sổ hợp lệ khi toàn bộ đầu vào và nhãn đủ sáu biến sau nội suy khuyết không quá 1 giờ. Nhãn nằm trọn trong tập; đầu vào được phép lấy dữ liệu quá khứ trước ranh giới tập.

| Tập | Đầu vào 24 giờ, dự báo 6 giờ | Đầu vào 7 ngày, dự báo 24 giờ |
|---|---|---|
| train | 187862 | 180076 |
| val | 24571 | 20003 |
| test | 25940 | 24572 |

Các cửa sổ chồng lấn nhau, không phải mẫu độc lập.

### 5.5. Mô hình tham chiếu

Hai mô hình tham chiếu không cần huấn luyện: giữ nguyên giá trị hiện tại (persistence) và lấy giá trị cùng giờ ngày hôm trước. Với tầm dự báo đến 24 giờ, mô hình cùng giờ hôm trước có sai số trùng với persistence 24 giờ.

| Tập | RMSE 1 giờ | R2 1 giờ | RMSE 6 giờ | R2 6 giờ | RMSE 24 giờ, cùng giờ hôm trước | R2 24 giờ, cùng giờ hôm trước |
|---|---|---|---|---|---|---|
| train | 0.216 | 0.969 | 0.479 | 0.850 | 0.311 | 0.937 |
| val | 0.233 | 0.980 | 0.490 | 0.911 | 0.325 | 0.961 |
| test | 0.189 | 0.955 | 0.392 | 0.808 | 0.245 | 0.925 |

RMSE tính theo mg/L. Do chu kỳ ngày mạnh, R2 của các mô hình tham chiếu đã rất cao; kết quả của mô hình học sâu cần được so sánh với các giá trị này.

### 5.6. Quy tắc sử dụng

1. Mọi tham số học từ dữ liệu (chuẩn hóa, ngưỡng của đặc trưng miền, tham số tinh chỉnh quy tắc ontology) chỉ được ước lượng trên train. Ngưỡng lấy từ quy chuẩn pháp lý không thuộc diện này.
2. Nội suy khuyết ngắn thực hiện riêng trong từng tập, không xuyên ranh giới. Không nội suy nhãn DO trong test; cửa sổ có nhãn thiếu bị loại.
3. Chọn cấu hình trên val. Sau khi chốt cấu hình, huấn luyện lại trên train cộng val (2019-01-01 đến 2025-02-28) rồi đánh giá trên test đúng một lần.
4. Các cấu hình so sánh (dữ liệu gốc, thêm đặc trưng miền, thêm đặc trưng ngữ nghĩa) dùng chung tập cửa sổ và chung ranh giới.
5. Đánh giá cảnh báo trên test tại ngưỡng 6 mg/L, vì test không có đợt dưới 5 mg/L. Ngưỡng 5 mg/L chỉ được báo cáo trên val để tham khảo. Hai ngưỡng này tương ứng với mục tiêu DO 5.0 mg/L quanh năm và 6.0 mg/L cho tháng 9 đến tháng 11 đối với hạ lưu sông San Joaquin (cần đối chiếu văn bản Basin Plan hiện hành của Ủy ban Chất lượng Nước vùng Central Valley).
6. Báo cáo MAE, RMSE, R2 trên toàn tập test và riêng từng chế độ mùa, kèm mô hình tham chiếu.

### 5.7. Các phương án đã cân nhắc

| Phương án | Kết luận |
|---|---|
| Chia theo năm: train 2019 đến 2022, val 2023, test 2024 đến 2025 (khoảng 57/14/29%) | Loại, val và test quá lớn |
| Chia theo năm: train 2019 đến 2023, val 2024, test 2025 (khoảng 71/14/14%) | Loại, val và test lớn hơn mức cần thiết |
| Val 2024-07 đến 2025-03, test 2025-04 đến 2025-12 | Loại, val thiếu tháng 6 nên không cân bằng với test |
| Chỉ lấy mùa nóng: val tháng 5 đến 11 năm 2024, test tháng 5 đến 11 năm 2025 (khoảng 8% mỗi tập) | So sánh cùng mùa tốt hơn nhưng không liền mạch, phải xử lý riêng 5 tháng mùa lạnh ở giữa |
| Phương án được chọn: train đến 2024-05, val 2024-06 đến 2025-02, test 2025-03 đến 2025-11 | Liền mạch, cắt tròn tháng, val và test cùng dài 9 tháng và cùng chứa một mùa nóng trọn vẹn |

Để kiểm tra độ vững của kết luận, có thể đánh giá thêm theo kiểu cửa sổ trượt với cùng cách cắt 9 tháng ở các năm trước, ví dụ train đến 2022-05 với val 2022-06 đến 2023-02, hoặc train đến 2023-05 với val 2023-06 đến 2024-02.

## 6. Kiểm chứng toàn vẹn

scripts/verify.py kiểm tra:

1. Mã SHA-256 của từng file khớp manifest_sha256.json.
2. File nguồn có 631835 dòng, khớp số bản ghi khai báo trong metadata EML; lưới 15 phút đều, không trùng, không mất mốc.
3. Mỗi tập con có đúng số dòng, mốc đầu và mốc cuối như mục 5.2; lưới 15 phút liên tục.
4. Từng ô của tập con giống hệt chuỗi giá trị tương ứng trong file nguồn.
5. Ba tập nối tiếp nhau, không chồng lấn, không có khoảng hở.
6. Số NA từng cột của mỗi tập bằng số NA của file nguồn trong cùng khoảng thời gian.

Kết quả lần chạy ngày 2026-11-12: đạt toàn bộ, 0 ô sai khác trên 242496 dòng và 6 cột. Bản sao trong raw/ có cùng mã SHA-256 với file tải từ nguồn (199692356508935d8f6490da111f339d059d4985df900433f925c68d71650398). Đã thử cố ý sửa một ô trong một bản sao tạm: script phát hiện cả sai mã băm lẫn sai lệch ô và trả mã lỗi.

## 7. Tái lập

Chạy từ thư mục dataset:

```
python scripts/make_splits.py      # tạo lại splits/ từ raw/
python scripts/verify.py           # kiểm chứng; dùng --write để ghi lại manifest
python scripts/describe.py         # tạo lại stats/, cần pandas và numpy
```

## 8. Nguồn và trích dẫn

Nelson, M. M., Tran, A. H. (2026). Sacramento-San Joaquin Bay-Delta Continuous (15 Minute) water quality monitoring data collected by the Continuous Environmental Monitoring Program, DWR, 2005-2025. Phiên bản 9 (edi.1177.9). Environmental Data Initiative. https://doi.org/10.6073/pasta/af06602f2547c66d9a8459f2dd7e4188

Giấy phép: Creative Commons Attribution 4.0 (CC BY 4.0). Người dùng phải trích dẫn nguồn trong mọi công bố sử dụng dữ liệu.
