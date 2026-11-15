# Quy trình làm sạch dữ liệu dataset CEMP MSD

Phiên bản ngày 2026-11-15. Trạng thái: quy trình đã chốt và đã cài đặt trong scripts/build_base_splits.py; kết quả nằm trong base_splits/.

Tài liệu mô tả cách làm sạch ba tập train, val, test (ranh giới ở README.md, mục 5) trước khi đưa vào mô hình dự báo oxy hòa tan (DO). Quy trình gồm hai bước: kiểm tra miền giá trị (mục 3) và điền mọi khoảng trống bằng số đo thật gần nhất phía trước (mục 4). Cả hai bước chỉ dùng quá khứ. Mã phân tích làm căn cứ nằm trong thư mục scripts/ ở gốc repo, kết quả trong artifacts/dataset/.

## 1. Khái niệm

| Khái niệm | Nghĩa |
|---|---|
| Mốc | Một dòng dữ liệu; các mốc cách nhau 15 phút, mỗi mốc có sáu biến |
| Khoảng trống | Các ô không có giá trị (NA) nằm liền nhau của cùng một biến. Độ dài khoảng trống là số mốc liên tiếp bị trống |
| Thời điểm dự báo | Thời điểm hiện tại của một lần dự báo |
| Đầu vào | Đoạn dữ liệu quá khứ đưa cho mô hình, kết thúc ngay trước thời điểm dự báo. Ví dụ đầu vào 24 giờ là 96 mốc |
| Nhãn | DO đo thực tế trong khoảng cần dự báo, bắt đầu từ thời điểm dự báo; dùng để tính sai số |
| Cửa sổ | Một cặp đầu vào và nhãn, tức một mẫu học hoặc một mẫu đánh giá. Dời thời điểm dự báo mỗi lần 15 phút để tạo các cửa sổ liên tiếp |
| Biến phụ | Năm biến ngoài DO: nhiệt độ, pH, độ dẫn, độ đục, chlorophyll |
| Nhân quả | Giá trị tại một mốc chỉ được tính từ số đo tại chính mốc đó và các mốc trước nó |

Ví dụ: với thời điểm dự báo 2019-07-15 12:00, đầu vào 24 giờ gồm các mốc từ 2019-07-14 12:00 đến 2019-07-15 11:45, nhãn 6 giờ gồm DO từ 12:00 đến 17:45.

Mô hình không nhận ô trống, nên mọi ô trống trong đầu vào phải được điền. Với đầu vào 24 giờ, mỗi ô nằm trong đầu vào của 96 cửa sổ liên tiếp.

## 2. Nguyên tắc

1. Không sửa raw/. Dữ liệu sau làm sạch được ghi vào base_splits/; mọi ô được điền đều có cờ đánh dấu (mục 5).
2. Nhân quả. Lý do: nếu giá trị điền dùng số đo phía sau khoảng trống, cửa sổ có thời điểm dự báo nằm trong khoảng trống sẽ gián tiếp thấy số đo sau thời điểm dự báo; và khi hệ thống web chạy thật, số đo mới nhất có thể bị trống trong khi chưa có số đo nào phía sau. Quy trình nhân quả chạy y hệt khi huấn luyện, khi đánh giá và khi dùng thật.
3. Một quy trình cho cả ba tập, chạy liền mạch qua ranh giới các tập: đầu val được điền từ số đo cuối của train, đầu test từ số đo cuối của val, vì đó là quá khứ.
4. Mọi tham số tính từ giá trị đo (ví dụ trung bình và độ lệch chuẩn để chuẩn hóa ở giai đoạn sau, tham số của phương pháp điền trong benchmark ở mục 4.2) chỉ được rút ra từ train, để val và test không ảnh hưởng gián tiếp đến mô hình được chấm trên chính chúng.
5. Sai số luôn được tính trên DO đo thực tế. Giá trị điền chỉ dùng làm đầu vào; mốc nhãn có DO được điền không được tính sai số (mục 5.3).
6. Phạm vi chỉ là làm sạch sáu biến quan trắc. Chuẩn hóa và tạo đặc trưng thuộc các giai đoạn sau. Quy tắc tạo cửa sổ ở README.md, mục 5.2.

## 3. Kiểm tra miền giá trị

Giá trị nằm ngoài miền vật lý được coi là lỗi (sai đơn vị, cảm biến ở ngoài nước, mã lỗi của thiết bị). Giá trị lỗi bị loại và xử lý như một ô trống ở mục 4. Phép kiểm tra chỉ xét từng giá trị riêng lẻ nên là nhân quả.

| Biến | Miền chấp nhận |
|---|---|
| DO | 0 đến 20 mg/L |
| Nhiệt độ | 0 đến 35 độ C |
| pH | 6 đến 10 |
| Độ dẫn | 20 đến 3000 uS/cm |
| Độ đục | 0 đến 1000 FNU |
| Chlorophyll | 0 đến 200 ug/L |

Từ 2019 đến 2025, khoảng thời gian của ba tập, không có giá trị nào ngoài miền. File nguồn có 302 giá trị DO vượt 20 mg/L (từ 20.01 đến 28.25 mg/L), đều trước năm 2019 (các năm 2009, 2012, 2013, 2014, 2016). Phép kiểm tra vẫn được giữ để chặn số đo vô lý khi hệ thống web nhận dữ liệu mới.

## 4. Điền khoảng trống

### 4.1. Khoảng trống trong dữ liệu

![Khoảng trống theo độ dài](../artifacts/dataset/gap_length_histogram.png)

Tính trên sáu biến và cả ba tập, có 672 khoảng trống, tổng cộng 4417 ô trống:

| Độ dài | Số khoảng trống | Tỉ lệ số khoảng | Tỉ lệ thời gian trống |
|---|---|---|---|
| 15 phút | 521 | 77.5% | 11.8% |
| 30 phút đến 1 giờ | 84 | 12.5% | 5.1% |
| 1 đến 2 giờ | 24 | 3.6% | 3.5% |
| 2 đến 6 giờ | 10 | 1.5% | 3.2% |
| 6 đến 12 giờ | 14 | 2.1% | 10.2% |
| 12 đến 24 giờ | 7 | 1.0% | 9.8% |
| 24 đến 48 giờ | 6 | 0.9% | 21.5% |
| Trên 48 giờ | 6 | 0.9% | 34.9% |

Hai cách đếm cho hai bức tranh khác nhau:
- Theo số khoảng, khoảng trống gần như luôn ngắn: 90% không quá 1 giờ.
- Theo thời gian, phần lớn ô trống nằm trong vài khoảng rất dài: 56% thời gian trống thuộc 12 khoảng dài hơn 1 ngày.

Các khoảng trống dài nhất đều nằm trong train, gồm năm lần cả đầu đo ngừng ghi và hai lần chỉ thiếu pH:

| Bắt đầu | Độ dài | Biến |
|---|---|---|
| 2019-04-10 07:30 | 20 giờ | pH |
| 2019-05-13 23:30 | 7.5 giờ | cả sáu biến |
| 2020-04-25 14:30 | 39.5 giờ | cả sáu biến |
| 2020-06-03 15:15 | 14.75 giờ | cả sáu biến |
| 2020-06-09 07:15 | 8.75 giờ | cả sáu biến |
| 2020-06-12 13:45 | 64.25 giờ | cả sáu biến |
| 2023-05-24 01:45 | 8.5 giờ | pH |

Khoảng trống dài nhất của val là 6.5 giờ (độ dẫn, từ 2025-01-25 10:30), của test là 1.5 giờ. Riêng DO có 618 ô trống (train 608, val 5, test 5); khi DO trống, gần như cả đầu đo ngừng ghi: 591 trong 618 ô có cả năm biến phụ cùng trống. Danh sách mọi khoảng trống từ 2 giờ trở lên: artifacts/dataset/gap_events.csv.

Từ đó, phương pháp điền phải dùng được cho khoảng trống từ 15 phút đến gần 3 ngày (mục 4.2). Vì các khoảng dài chỉ có trong train, chất lượng điền khoảng dài ảnh hưởng đến dữ liệu học, không ảnh hưởng đến val và test.

### 4.2. Chọn phương pháp điền

**Các phương pháp.** Benchmark so sánh năm phương pháp dưới đây; tất cả chỉ dùng số đo trước khoảng trống và không biết khoảng trống sẽ dài bao nhiêu, giống khi chạy thật. Mỗi phương pháp sửa điểm yếu của phương pháp ngay trên nó:

| Phương pháp | Cách điền | Điểm yếu |
|---|---|---|
| Giữ số trước | Lặp lại số đo cuối trước khoảng trống | Khoảng trống dài thì đường điền phẳng, không theo nhịp ngày đêm |
| Cùng giờ hôm qua | Chép giá trị cùng giờ ngày hôm trước, nên có nhịp ngày đêm | Mức DO hôm nay thường khác hôm qua |
| Hôm qua cộng độ lệch | Cùng giờ hôm qua cộng chênh lệch giữa hôm nay và hôm qua tại số đo cuối, nên đúng mức của hôm nay | Ngay đầu khoảng trống đã nhảy sang đường hôm qua, trong khi số đo cuối là thông tin sát nhất |
| Hôm qua cộng độ lệch giảm dần | Như trên, độ lệch nhỏ dần theo thời gian | Vẫn nhảy sang đường hôm qua ngay đầu khoảng trống |
| Pha trộn | Đầu khoảng trống giống giữ số trước, càng về sau càng chuyển sang hôm qua cộng độ lệch | Cần chọn tốc độ chuyển riêng cho mỗi biến |

Với khoảng trống dài hơn một ngày, "hôm qua" của các ô phía sau cũng nằm trong khoảng trống; các phương pháp dựa trên hôm qua khi đó lặp lại ngày đầy đủ cuối cùng trước khoảng trống.

**Benchmark.** Trên train, xóa nhân tạo 200 đoạn cho mỗi độ dài 15 phút, 1, 2, 4, 8, 24, 48 và 64.25 giờ, ở những nơi có đủ sáu biến, rồi điền lại và so với giá trị thật (MAE). Tốc độ chuyển được chọn trên 85% đầu train (2019-01-01 đến 2023-08-09), sai số được chấm trên 15% cuối (2023-08-09 đến 2024-05-31). Sai số tổng hợp lấy trung bình theo tỉ lệ thời gian trống thật ở từng độ dài (mục 4.1), nên khoảng trống dài có trọng số lớn. Nội suy tuyến tính, nối thẳng giữa số đo trước và sau khoảng trống, được tính kèm để tham khảo.

Kết quả cho DO (MAE, mg/L):

| Phương pháp | 15 phút | 1 giờ | 2 giờ | 4 giờ | 8 giờ | 24 giờ | 48 giờ | 64 giờ | Tổng hợp |
|---|---|---|---|---|---|---|---|---|---|
| Giữ số trước | 0.018 | 0.026 | 0.042 | 0.071 | 0.128 | 0.166 | 0.209 | 0.244 | 0.170 |
| Cùng giờ hôm qua | 0.147 | 0.141 | 0.144 | 0.140 | 0.152 | 0.142 | 0.177 | 0.210 | 0.175 |
| Hôm qua cộng độ lệch | 0.020 | 0.024 | 0.033 | 0.044 | 0.066 | 0.119 | 0.155 | 0.189 | 0.127 |
| Hôm qua cộng độ lệch giảm dần | 0.021 | 0.023 | 0.032 | 0.043 | 0.067 | 0.101 | 0.151 | 0.189 | 0.124 |
| Pha trộn | 0.018 | 0.022 | 0.031 | 0.044 | 0.066 | 0.118 | 0.155 | 0.189 | 0.127 |
| Nội suy tuyến tính (tham khảo, dùng số đo sau khoảng trống) | 0.014 | 0.012 | 0.018 | 0.027 | 0.063 | 0.143 | 0.148 | 0.168 | 0.120 |

Sai số tổng hợp của sáu biến (in đậm là thấp nhất trong các phương pháp chỉ dùng quá khứ):

| Biến | Giữ số trước | Cùng giờ hôm qua | Hôm qua cộng độ lệch | Hôm qua cộng độ lệch giảm dần | Pha trộn | Nội suy tuyến tính |
|---|---|---|---|---|---|---|
| DO (mg/L) | 0.170 | 0.175 | 0.127 | **0.124** | 0.127 | 0.120 |
| Nhiệt độ (độ C) | 0.415 | 0.499 | 0.301 | **0.301** | 0.302 | 0.268 |
| pH | 0.040 | 0.041 | 0.035 | **0.033** | 0.034 | 0.031 |
| Độ dẫn (uS/cm) | 15.21 | 25.46 | 16.84 | 16.62 | **15.10** | 7.27 |
| Độ đục (FNU) | **2.33** | 3.04 | 2.86 | 2.55 | 2.35 | 1.51 |
| Chlorophyll (ug/L) | 0.658 | 0.710 | 0.626 | 0.597 | **0.591** | 0.516 |

**Kết luận: chọn giữ số trước.** Trên khoảng trống dài, các cách dựa trên hôm qua gần đúng hơn: với DO, sai số tổng hợp của giữ số trước là 0.170 mg/L, của pha trộn 0.127 mg/L. Quy trình vẫn chọn giữ số trước, vì:
- Khi DO trống, gần như cả đầu đo ngừng ghi (591 trên 618 ô, mục 4.1). Giữ số trước chép lại đúng một lần đo thật của cả sáu biến, nên các biến vẫn khớp nhau như khi đo. Các cách dựa trên hôm qua điền từng biến riêng theo giả định hôm nay lặp lại hôm qua, có thể ghép ra những tổ hợp giá trị chưa từng xảy ra, nhất là khi nước đổi đột ngột (lũ, tảo nở hoa).
- Giữ số trước không có tham số nào và áp dụng như nhau cho sáu biến; pha trộn và hôm qua cộng độ lệch giảm dần cần chọn thêm tốc độ chuyển cho từng biến.
- Ô được điền rất ít: 0.36% số ô của train, 0.13% của val, 0.07% của test. Val và test chỉ có khoảng trống không quá 6.5 giờ; ở độ dài này giữ số trước sai ít (DO 0.018 mg/L ở 15 phút, 0.026 ở 1 giờ, 0.071 ở 4 giờ). Các khoảng trống dài chỉ có trong train.
- Mô hình được báo ô nào là ô điền qua cột cờ, và không bị tính sai số trên giá trị điền (mục 5.3).
- Bước làm sạch dùng chung cho mọi thí nghiệm so sánh (dữ liệu gốc, thêm đặc trưng miền, thêm đặc trưng ngữ nghĩa), nên cách điền không làm lệch phép so sánh giữa chúng.

Các phương pháp không xét:

| Phương pháp | Lý do |
|---|---|
| Dùng số đo sau khoảng trống (nội suy tuyến tính, PCHIP, mô hình học sâu hai chiều như SAITS) | Không nhân quả. Nội suy tuyến tính chính xác hơn giữ số trước 0.050 mg/L với DO (sai số tổng hợp 0.120 so với 0.170), nhưng làm 0.40% cửa sổ train, 0.71% val, 0.30% test gián tiếp thấy số đo sau thời điểm dự báo, và không dùng được cho số đo mới nhất khi chạy thật |
| Đoán DO từ các biến khác cùng mốc (ví dụ MissForest) | Khi DO trống, 591 trên 618 ô có cả năm biến phụ cùng trống |

Kết quả đầy đủ: artifacts/dataset/causal_imputation_benchmark.csv (theo biến, phương pháp, độ dài), causal_imputation_summary.csv (sai số tổng hợp, tốc độ chuyển) và linear_leak.csv.

### 4.3. Quy tắc

Với mỗi ô trống (kể cả ô bị loại ở mục 3), xét số đo thật gần nhất phía trước của cùng biến:

| Trường hợp | Xử lý | Cờ |
|---|---|---|
| Đã có số đo thật phía trước | Chép giá trị của số đo đó, dù khoảng trống dài bao nhiêu | 1 |
| Chưa có số đo nào phía trước | Để NA; mọi cửa sổ chạm vào ô này bị bỏ | -1 |

Trường hợp thứ hai chỉ xảy ra khi khoảng trống nằm ngay đầu chuỗi; trong dataset này không có ô nào như vậy. Quy tắc áp dụng như nhau cho sáu biến và ba tập, chạy liền mạch qua ranh giới các tập. Ví dụ khoảng trống ngắn, DO ngày 2024-06-18 (val):

| Mốc | DO đo được (mg/L) | Sau làm sạch | Cờ |
|---|---|---|---|
| 11:00 | 10.89 | 10.89 | 0 (đo thật) |
| 11:15 | trống | 10.89 | 1 (điền) |
| 11:30 | trống | 10.89 | 1 (điền) |
| 11:45 | 9.65 | 9.65 | 0 (đo thật) |

Ví dụ khoảng trống dài: lần ngừng ghi trong train, cả sáu biến trống từ 2020-06-12 13:45 đến 05:45 ngày 2020-06-15 (64.25 giờ). Cả sáu biến được chép từ lần đo lúc 13:30 (DO 11.21 mg/L) suốt khoảng trống. Cửa sổ chỉ bị bỏ khi nhãn không còn mốc DO đo thật nào: với đầu vào 24 giờ và nhãn 6 giờ, đó là các cửa sổ có thời điểm dự báo từ 2020-06-12 13:45 đến 2020-06-15 00:00.

Khi chạy thật, hệ thống web điền theo đúng quy tắc này, nên vẫn dự báo được khi trạm mất số liệu. Sau nhiều giờ mất số liệu, đầu vào chủ yếu là giá trị chép lại; hệ thống nên báo cho người dùng biết dự báo đang dựa trên dữ liệu điền (mục 4.4).

### 4.4. Kết quả trên ba tập

**Số ô được điền** (tỉ lệ trên tổng số ô của tập); không có ô cờ -1:

| | train | val | test |
|---|---|---|---|
| Cờ 1: điền bằng số đo trước | 4098 (0.36%) | 212 (0.13%) | 107 (0.07%) |

**Số cửa sổ bị bỏ** trên số cửa sổ có thể tạo. Cửa sổ chỉ bị bỏ khi nhãn không còn mốc DO đo thật nào, tức nằm gọn trong một lần ngừng ghi dài; điều này chỉ xảy ra trong train:

| Cấu hình | train | val | test |
|---|---|---|---|
| Đầu vào 24 giờ, dự báo 6 giờ | 424 trên 189769 (0.22%) | 0 trên 26185 | 0 trên 26377 |
| Đầu vào 7 ngày, dự báo 24 giờ | 225 trên 189121 (0.12%) | 0 trên 26113 | 0 trên 26305 |

Không cửa sổ nào thấy số đo sau thời điểm dự báo, vì mọi giá trị điền chỉ lấy từ quá khứ.

**Giới hạn.** Val và test không có khoảng trống nào dài hơn 6.5 giờ, nên chất lượng dự báo khi đầu vào chứa đoạn chép phẳng dài (nhiều giờ đến vài ngày) không được đo trong dataset này.

## 5. Đầu ra và cờ đánh dấu

### 5.1. Đầu ra

Thư mục base_splits/ gồm ba file train.csv, val.csv, test.csv, sinh bởi scripts/build_base_splits.py. Mỗi file có cột datetime_pst, sáu biến và sáu cột cờ flag_<tên biến>, ví dụ flag_do_mg_l. Ô cờ 0 chép nguyên văn từ nguồn; ô cờ 1 chép nguyên văn số đo nguồn được dùng để điền; ô cờ -1 ghi NA.

### 5.2. Cờ

Mỗi biến đo có một cột cờ để phân biệt số đo thật với số điền. Cờ tách theo biến vì trong cùng một mốc có thể có biến đo thật và biến được điền (ví dụ từ 2025-01-25 10:30, chỉ độ dẫn trống 6.5 giờ):

| Giá trị cờ | Nghĩa |
|---|---|
| 0 | Đo thật, giữ nguyên |
| 1 | Ô trống trong nguồn hoặc số đo ngoài miền (mục 3), đã điền bằng số đo thật gần nhất phía trước |
| -1 | Chưa có số đo nào phía trước; giá trị để NA |

### 5.3. Cách dùng cờ

Trong đầu vào, mô hình đọc giá trị điền và cả sáu cột cờ như sáu đặc trưng (1 với ô được điền, 0 với số đo thật). Nhờ vậy mô hình biết đoạn nào là giá trị điền chứ không phải số đo thật. Cửa sổ chứa ô cờ -1 bị bỏ, nên trong đầu vào cờ chỉ có 0 hoặc 1.

Trong nhãn, mốc có DO được điền không được tính sai số: sai số của một cửa sổ là trung bình trên các mốc nhãn có cờ DO bằng 0. Trong mã, cờ DO được chuyển thành một dãy mặt nạ (1 với mốc tính sai số, 0 với mốc bỏ qua); sai số tại mỗi mốc được nhân với mặt nạ rồi chia cho tổng mặt nạ. Hàm mất mát khi huấn luyện được tính theo cùng cách, nên mô hình luôn học dự báo DO đo thật, không học theo giá trị điền.

Ví dụ với ô DO lúc 2024-06-18 11:15 (đầu vào 24 giờ, dự báo 6 giờ):

| Nhóm cửa sổ | Số cửa sổ | Vị trí ô 11:15 | Cách dùng |
|---|---|---|---|
| Thời điểm dự báo 11:30 đến 11:15 hôm sau | 96 | Đầu vào | Mô hình đọc giá trị điền 10.89 kèm cờ 1 |
| Thời điểm dự báo 05:30 đến 11:15 | 24 | Nhãn | Mốc 11:15 không được tính sai số |

## 6. Điểm DO đột biến một mốc

Điểm DO đột biến một mốc là mốc mà DO lệch quá 0.8 mg/L so với mốc trước rồi quay lại quá 0.8 mg/L ở mốc sau, ví dụ ngày 2025-08-14 lúc 10:15, 10:30, 10:45 DO là 7.58, 3.73, 7.16 mg/L. Bước nhảy như vậy hiếm: 99% bước thay đổi 15 phút của DO trong train không quá 0.40 mg/L.

Có 212 điểm đột biến: train 63, val 28, test 121. 211 điểm nằm trong các tháng 6 đến 9, phần lớn vào tháng 7 và 8, lúc tảo nở hoa và DO dao động mạnh nhất trong năm (README.md, mục 4.4). 136 điểm có nhiệt độ đổi từ 0.3 độ C hoặc pH đổi từ 0.05 cùng lúc, phù hợp với một khối nước khác đi qua cảm biến; 2 điểm có cảm biến khác trên cùng đầu đo trống trong vòng 1 giờ, nghi là lỗi thiết bị; 74 điểm (61 trong test) không có biến nào giải thích. Tiêu chí đổi cùng lúc không phân biệt được khối nước thật với một trục trặc làm cả đầu đo lệch theo, nên không xác định được bao nhiêu điểm là lỗi.

Các điểm này ảnh hưởng đáng kể đến mốc DO thấp của test: 8 trên 9 mốc DO dưới 5 mg/L và 17 trên 34 mốc dưới 6 mg/L là điểm đột biến. Một điểm đột biến đứng riêng không tạo thành đợt DO thấp (đợt phải dài ít nhất 1 giờ), nhưng có thể nối với các mốc thấp gần đó: trong 5 đợt DO dưới 6 mg/L của test, đợt bắt đầu 2025-08-14 10:30 có mốc thấp nhất 3.73 mg/L là một điểm đột biến.

Cả 212 điểm được giữ là số đo thật (cờ 0), vì:
- muốn nhận ra một điểm đột biến phải có số đo ở mốc sau, nên phép phát hiện không nhân quả và không áp dụng được cho số đo mới nhất khi chạy thật;
- không xác định được điểm nào là lỗi (đoạn trên);
- ảnh hưởng lên sai số nhỏ: bỏ các mốc nhãn là điểm đột biến chỉ làm MAE của cách dự báo cùng giờ hôm qua (README.md, mục 5.3) đổi từ 0.400 xuống 0.394 mg/L trên test, từ 0.328 xuống 0.327 trên val (artifacts/dataset/reversal_effect.csv).

## 7. Các xử lý không thực hiện

| Xử lý | Lý do |
|---|---|
| Làm trơn DO | Làm mất đỉnh và đáy trong ngày, là thông tin cần dự báo và cảnh báo |
| Xóa cực trị bằng ngưỡng thống kê (ví dụ bộ lọc Hampel) | Phần lớn cực trị là hiện tượng thật: đỉnh DO buổi chiều mùa hè kèm pH trên 9 do tảo quang hợp, lúc độ bão hòa oxy vượt 150%; đỉnh đơn lẻ của độ đục và chlorophyll |
| Sửa các đoạn giá trị không đổi | Train có 30 đoạn pH và 102 đoạn độ dẫn không đổi từ 6 giờ trở lên, chủ yếu vào mùa đông và mùa xuân khi nước ổn định; đó là do độ phân giải của cảm biến (pH 0.01, độ dẫn 1 uS/cm) |

## 8. Tái lập

Môi trường Python khai báo trong pyproject.toml ở gốc repo. Cài một lần bằng `uv sync`, rồi chạy từ gốc repo:

```
uv run python scripts/build_base_splits.py            # tạo dataset/base_splits/ từ dataset/raw/
uv run python scripts/cleaning_study.py               # mục 3, 6: miền giá trị, bước thay đổi 15 phút, điểm DO đột biến
uv run python scripts/causal_imputation_benchmark.py  # mục 4.2: benchmark các phương pháp điền
uv run python scripts/gap_study.py                    # mục 4.1: histogram khoảng trống, danh sách khoảng trống dài
uv run python scripts/verify_dataset.py               # kiểm chứng, gồm cả các số liệu ở mục 4.4
```

Hai script cleaning_study.py và causal_imputation_benchmark.py đọc base_splits/, nên chạy sau build_base_splits.py.
