# Quy trình làm sạch dữ liệu dataset CEMP RRI

Phiên bản ngày 2026-11-14. Trạng thái: quy trình đã chốt và đã cài đặt trong scripts/build_base_splits.py; kết quả nằm trong base_splits/.

Tài liệu mô tả cách làm sạch ba tập train, val, test (ranh giới ở README.md, mục 5) trước khi đưa vào mô hình dự báo oxy hòa tan (DO). Quy trình gồm hai bước: kiểm tra miền giá trị (mục 3) và điền khoảng trống (mục 4). Cả hai bước chỉ dùng quá khứ. Mã phân tích làm căn cứ nằm trong thư mục scripts/ ở gốc repo, kết quả trong artifacts/dataset/.

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

Mô hình không nhận ô trống: ô trống trong đầu vào phải được điền, nếu không cửa sổ bị bỏ. Với đầu vào 24 giờ, mỗi ô nằm trong đầu vào của 96 cửa sổ liên tiếp.

## 2. Nguyên tắc

1. Không sửa raw/. Dữ liệu sau làm sạch được ghi vào base_splits/; mọi ô được điền hoặc để trống đều có cờ đánh dấu (mục 5).
2. Nhân quả. Lý do: nếu giá trị điền dùng số đo phía sau khoảng trống, cửa sổ có thời điểm dự báo nằm trong khoảng trống sẽ gián tiếp thấy số đo sau thời điểm dự báo; và khi hệ thống web chạy thật, số đo mới nhất có thể bị trống trong khi chưa có số đo nào phía sau. Quy trình nhân quả chạy y hệt khi huấn luyện, khi đánh giá và khi dùng thật.
3. Một quy trình cho cả ba tập, chạy liền mạch qua ranh giới các tập: đầu val được điền từ số đo cuối của train, đầu test từ số đo cuối của val, vì đó là quá khứ.
4. Mọi ngưỡng và lựa chọn phương pháp được rút ra từ train.
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

Dataset RRI không có giá trị nào ngoài miền. Phép kiểm tra này vẫn được giữ để chặn số đo vô lý khi hệ thống web nhận dữ liệu mới.

## 4. Điền khoảng trống

Mục này trả lời lần lượt ba câu hỏi: dữ liệu có những khoảng trống nào (mục 4.1), điền bằng cách nào (mục 4.2) và điền tối đa bao lâu (mục 4.3). Quy tắc rút ra ở mục 4.4, kết quả trên ba tập ở mục 4.5.

### 4.1. Khoảng trống trong dữ liệu

![Độ dài khoảng trống](../artifacts/dataset/gap_length_histogram.png)

Tính trên sáu biến và cả ba tập (2377 khoảng trống):

| Độ dài | Số khoảng trống | Tích lũy |
|---|---|---|
| 15 phút | 2175 | 91.50% |
| 30 phút đến 1 giờ | 162 | 98.32% |
| 1 đến 2 giờ | 15 | 98.95% |
| 2 đến 4 giờ | 3 | 99.07% |
| 4 đến 8 giờ | 9 | 99.45% |
| 8 đến 16 giờ | 0 | 99.45% |
| 16 đến 24 giờ | 6 | 99.71% |
| 24 đến 48 giờ | 6 | 99.96% |
| Trên 48 giờ | 1 | 100% |

Khoảng trống chia thành hai nhóm tách biệt:
- Nhóm ngắn, dài tối đa 7.75 giờ, chiếm 99.45% số khoảng trống; riêng khoảng trống 15 phút chiếm 91.5% số khoảng trống và 74% số ô trống của nhóm này. 12 khoảng trống dài từ 2 đến 7.75 giờ đều thuộc biến phụ (pH, độ đục, chlorophyll), không có khoảng nào của DO; 10 trong số đó nằm trong train, một nửa do cảm biến độ đục hỏng tháng 8 năm 2019.
- Nhóm dài, từ 21.25 giờ trở lên: 13 khoảng trống thuộc ba sự kiện, đều nằm trong val và test.

| Tập | Thời gian trống | Độ dài | Biến | Bối cảnh |
|---|---|---|---|---|
| val | 2024-09-02 14:00 đến 2024-09-03 11:00 | 21.25 giờ | cả sáu biến | Bảo trì, giữa mùa nóng |
| val | 2025-02-14 03:30 đến 2025-02-18 06:30 | 99.25 giờ | độ dẫn | Đợt lũ sau mưa |
| test | 2025-09-10 09:30 đến 2025-09-11 09:45 | 24.5 giờ | cả sáu biến | Bảo trì, giữa mùa nóng |

Từ 7.75 đến 21.25 giờ không có khoảng trống nào. Riêng DO có 543 ô trống (train 300, val 129, test 114); khoảng trống DO dài nhất trong train là 1.25 giờ. Khi DO trống, gần như cả đầu đo ngừng ghi: 480 trong 543 ô có cả năm biến phụ cùng trống.

Hai đặc điểm này quyết định phần sau: khoảng trống chủ yếu rất ngắn (mục 4.2), và có một khoảng hở rõ giữa nhóm ngắn và nhóm dài (mục 4.3).

### 4.2. Chọn phương pháp điền

Các phương pháp dưới đây chỉ dùng số đo trước khoảng trống và không biết khoảng trống sẽ dài bao nhiêu, giống khi chạy thật. Mỗi phương pháp sửa điểm yếu của phương pháp ngay trên nó:

| Phương pháp | Cách điền | Điểm yếu |
|---|---|---|
| Giữ số trước | Lặp lại số đo cuối trước khoảng trống | Khoảng trống dài thì đường điền phẳng, không theo nhịp ngày đêm |
| Cùng giờ hôm qua | Chép giá trị cùng giờ ngày hôm trước, nên có nhịp ngày đêm | Mức DO hôm nay thường khác hôm qua |
| Hôm qua cộng độ lệch | Cùng giờ hôm qua cộng chênh lệch giữa hôm nay và hôm qua tại số đo cuối, nên đúng mức của hôm nay | Độ lệch đo tại một cặp số đo nhưng được giữ nguyên suốt khoảng trống, trong khi chênh lệch thật thay đổi dần |
| Hôm qua cộng độ lệch giảm dần | Như trên, độ lệch nhỏ dần theo thời gian | Ngay đầu khoảng trống vẫn cộng thêm biến động 15 phút của hôm qua, nên kém giữ số trước ở khoảng trống ngắn |
| Pha trộn | Đầu khoảng trống giống giữ số trước, càng về sau càng chuyển sang hôm qua cộng độ lệch | Cần chọn tốc độ chuyển riêng cho mỗi biến |

**Cách tính.** Trên 15% cuối của train (2023-08-09 đến 2024-05-31), xóa nhân tạo 400 đoạn cho mỗi độ dài 15 phút, 30 phút, 1, 2, 4, 8 giờ ở những nơi có đủ số đo, rồi điền lại và so với giá trị thật (MAE). Tốc độ giảm dần và tốc độ chuyển được chọn trên 85% đầu train. Sai số tổng hợp lấy trung bình theo tỉ lệ ô trống thật của nhóm ngắn ở từng độ dài (mục 4.1). Nội suy tuyến tính, nối thẳng giữa số đo trước và sau khoảng trống, được tính kèm để tham khảo.

**Kết quả cho DO** (MAE, mg/L; in đậm là thấp nhất trong các phương pháp chỉ dùng quá khứ):

| Phương pháp | 15 phút | 1 giờ | 2 giờ | 4 giờ | 8 giờ | Tổng hợp |
|---|---|---|---|---|---|---|
| Giữ số trước | 0.067 | **0.084** | 0.111 | 0.169 | 0.199 | 0.081 |
| Cùng giờ hôm qua | 0.196 | 0.203 | 0.190 | 0.212 | 0.202 | 0.197 |
| Hôm qua cộng độ lệch | 0.085 | 0.119 | 0.136 | 0.180 | 0.183 | 0.099 |
| Hôm qua cộng độ lệch giảm dần | 0.083 | 0.115 | 0.126 | 0.163 | 0.167 | 0.095 |
| Pha trộn | **0.066** | **0.084** | **0.106** | **0.150** | **0.161** | **0.078** |
| Nội suy tuyến tính (tham khảo, dùng số đo sau khoảng trống) | 0.046 | 0.058 | 0.073 | 0.109 | 0.138 | 0.056 |

Năm biến phụ cho cùng thứ hạng: pha trộn đứng đầu, giữ số trước thứ hai, các cách dựa trên hôm qua kém hơn rõ (cộng độ lệch cao hơn giữ số trước 15% đến 49%, chỉ chép hôm qua cao gấp 1.6 đến 6.6 lần).

**Kết luận: chọn giữ số trước.** Pha trộn tốt hơn có ý nghĩa thống kê ở năm trên sáu biến (pH ngang nhau), nhưng mức chênh rất nhỏ: với DO là 0.003 mg/L, tính trên các ô được điền, vốn chỉ chiếm khoảng 0.2% số ô; mô hình còn được báo ô nào là ô điền qua cột cờ. Đổi lại, pha trộn cần một tốc độ chuyển riêng cho mỗi biến (từ 2 giờ với nhiệt độ đến 24 giờ với độ đục). Giữ số trước không có tham số nào và áp dụng như nhau cho sáu biến.

Các phương pháp không xét:

| Phương pháp | Lý do |
|---|---|
| Dùng số đo sau khoảng trống (nội suy tuyến tính, PCHIP, mô hình học sâu hai chiều như SAITS) | Không nhân quả. Nội suy tuyến tính chính xác hơn giữ số trước 0.025 mg/L trên ô điền, nhưng làm 0.52% cửa sổ train, 0.71% val, 0.39% test gián tiếp thấy số đo sau thời điểm dự báo, và không dùng được cho số đo mới nhất khi chạy thật |
| Đoán DO từ các biến khác cùng mốc (ví dụ MissForest) | Khi DO trống, 480 trên 543 ô có cả năm biến phụ cùng trống |

Kết quả đầy đủ: artifacts/dataset/causal_imputation_benchmark.csv và causal_imputation_summary.csv.

### 4.3. Chọn ngưỡng 8 giờ

Giữ số trước chỉ hợp lý khi số đo cuối còn đủ gần. Ngưỡng T quy định điều đó: ô cách số đo thật gần nhất phía trước không quá T thì điền, xa hơn thì để NA và bỏ mọi cửa sổ chạm vào. Ngưỡng được chọn qua ba bước, dựa trên hai nhóm khoảng trống ở mục 4.1. Số cửa sổ dưới đây tính cho cấu hình đầu vào 24 giờ, dự báo 6 giờ; cấu hình 7 ngày, 24 giờ cho cùng kết luận (artifacts/dataset/threshold_sensitivity.csv).

**Bước 1: T không được nhỏ hơn 7.75 giờ.** Ngưỡng nhỏ hơn thì các khoảng trống ngắn dài nhất (pH, độ đục) chỉ được điền phần đầu, phần đuôi bị để trống và cửa sổ bị bỏ, kể cả trong train:

| T | Cửa sổ train bị bỏ | Cửa sổ val bị bỏ |
|---|---|---|
| 4 giờ | 822 | 814 |
| 6 giờ | 350 | 673 |
| 7.75 giờ | 0 | 666 |

**Bước 2: T lớn hơn 7.75 giờ không đem lại gì đáng kể.** Nhóm ngắn đã được điền hết, nên tăng T chỉ làm phần đầu của ba sự kiện dài bị chép phẳng lâu hơn. Ví dụ lần bảo trì bắt đầu 14:00 ngày 2024-09-02, số đo cuối lúc 13:45 có DO 6.83 mg/L: với T = 8 giờ, giá trị này được chép đến 21:45; với T = 21 giờ, chép suốt đêm đến gần 11:00 hôm sau, trong khi DO thật giảm dần qua đêm. Tăng T từ 8 lên 21 giờ chỉ giữ thêm 52 cửa sổ val (0.2%) và không thêm cửa sổ test nào, trong khi sai số của phần điền tăng (với DO, 0.23 mg/L ở khoảng trống 8 giờ, 0.28 mg/L ở 21 giờ; artifacts/dataset/ffill_by_length.csv), lại nằm trong val và test, phần dùng để chấm mô hình.

**Bước 3: chọn giá trị nhỏ nhất đủ dùng.** Bước 1 cho giới hạn dưới 7.75 giờ, bước 2 cho thấy vượt quá mức đó chỉ thêm dữ liệu giả. Ngưỡng là 7.75 giờ làm tròn lên: 8 giờ.

Hai cách cực đoan cũng đã xem xét: không điền ô nào (T = 0) làm mất 19.5%, 34.8%, 20.6% cửa sổ train, val, test; điền toàn bộ khoảng trống dài sẽ chép phẳng gần một ngày ở cả sáu biến, riêng độ dẫn suốt 4 ngày lũ, lúc độ dẫn biến đổi mạnh nhất.

### 4.4. Quy tắc

Từ mục 4.2 và 4.3: với mỗi ô trống (kể cả ô bị loại ở mục 3), xét số đo thật gần nhất phía trước của cùng biến.

| Trường hợp | Xử lý | Cờ |
|---|---|---|
| Cách không quá 8 giờ (32 mốc) | Chép giá trị của số đo đó | 1 |
| Cách hơn 8 giờ, hoặc chưa có số đo nào | Để NA; mọi cửa sổ có đầu vào hoặc nhãn chạm vào ô này bị bỏ | -1 |

Quy tắc áp dụng như nhau cho sáu biến và ba tập, chạy liền mạch qua ranh giới các tập. Ví dụ khoảng trống ngắn, DO ngày 2019-07-15:

| Mốc | DO đo được (mg/L) | Sau làm sạch | Cờ |
|---|---|---|---|
| 09:45 | 7.10 | 7.10 | 0 (đo thật) |
| 10:00 | trống | 7.10 | 1 (điền) |
| 10:15 | trống | 7.10 | 1 (điền) |
| 10:30 | 7.39 | 7.39 | 0 (đo thật) |

Ví dụ khoảng trống dài: lần bảo trì trong val, cả sáu biến trống từ 2024-09-02 14:00 đến 11:00 hôm sau (21.25 giờ). Từ 14:00 đến 21:45 được điền bằng số đo lúc 13:45; từ 22:00 đến 11:00 hôm sau để NA. Với đầu vào 24 giờ và nhãn 6 giờ, các cửa sổ có thời điểm dự báo từ 2024-09-02 14:00 đến 2024-09-04 11:00 bị bỏ, vì nhãn không còn DO đo thật hoặc đầu vào, nhãn chạm ô NA. Các mốc trống được giữ nguyên, không xóa rồi ghép hai đầu, vì như vậy mốc 13:45 ngày 2/9 sẽ đứng ngay trước mốc 11:15 ngày 3/9 như thể chỉ cách nhau 15 phút.

Quy tắc này trùng với những gì hệ thống web làm được khi mất số liệu: trong 8 giờ đầu chỉ biết số đo cuối cùng; sau 8 giờ thì ngừng dự báo, cho đến khi số đo trở lại và đầu vào không còn chứa phần bị trống.

### 4.5. Kết quả trên ba tập

**Số ô theo cờ** (tỉ lệ trên tổng số ô của tập):

| Cờ | train | val | test |
|---|---|---|---|
| 1: điền bằng số đo trước | 2457 (0.22%) | 529 (0.34%) | 363 (0.23%) |
| -1: để trống | 0 | 683 (0.43%) | 396 (0.25%) |

**Số cửa sổ bị bỏ** trên số cửa sổ có thể tạo. Chỉ val và test mất cửa sổ, vì ba khoảng trống dài đều nằm trong hai tập này:

| Cấu hình | train | val | test |
|---|---|---|---|
| Đầu vào 24 giờ, dự báo 6 giờ | 0 trên 189769 | 665 trên 26185 (2.54%) | 194 trên 26377 (0.74%) |
| Đầu vào 7 ngày, dự báo 24 giờ | 0 trên 189121 | 1952 trên 26113 (7.48%) | 833 trên 26305 (3.17%) |

**Ảnh hưởng đến đánh giá cảnh báo.** Đợt DO thấp là chuỗi DO dưới ngưỡng kéo dài ít nhất 1 giờ; hai chuỗi cách nhau không quá 1 giờ được gộp. Một đợt bị mất khỏi đánh giá khi không còn mốc nào của nó nằm trong nhãn của cửa sổ được giữ. Trong số các đợt DO dưới 6 mg/L (train 152, val 68, test 100):
- cấu hình 24 giờ, 6 giờ mất 2 đợt ở val, không mất đợt nào ở test;
- cấu hình 7 ngày, 24 giờ mất 2 đợt ở val và 6 đợt ở test, vì cửa sổ dài hơn nên chạm khoảng trống nhiều hơn.

Các đợt bị mất nằm trong vài ngày nóng ngay sau hai lần bảo trì, DO thấp nhất 5.6 đến 6.0 mg/L. Không đợt DO dưới 5 mg/L nào bị mất (artifacts/dataset/episode_coverage.csv).

Không cửa sổ nào thấy số đo sau thời điểm dự báo, vì mọi giá trị điền chỉ lấy từ quá khứ.

## 5. Đầu ra và cờ đánh dấu

### 5.1. Đầu ra

Thư mục base_splits/ gồm ba file train.csv, val.csv, test.csv, sinh bởi scripts/build_base_splits.py. Mỗi file có cột datetime_pst, sáu biến và sáu cột cờ flag_<tên biến>, ví dụ flag_do_mg_l. Ô cờ 0 chép nguyên văn từ nguồn; ô cờ 1 chép nguyên văn số đo nguồn được dùng để điền; ô cờ -1 ghi NA. Ở bước tạo cửa sổ chỉ cần bỏ mọi cửa sổ có ô cờ -1.

### 5.2. Cờ

Mỗi biến đo có một cột cờ để phân biệt số đo thật với số điền. Cờ tách theo biến vì trong cùng một mốc có thể có biến đo thật và biến được điền (ví dụ từ 2021-12-01 10:30, chỉ pH trống 4.5 giờ):

| Giá trị cờ | Nghĩa |
|---|---|
| 0 | Đo thật, giữ nguyên |
| 1 | Ô trống trong nguồn hoặc số đo ngoài miền (mục 3), đã điền bằng số đo thật gần nhất phía trước, cách không quá 8 giờ |
| -1 | Cách số đo thật gần nhất phía trước hơn 8 giờ, hoặc chưa có số đo nào; giá trị để NA |

### 5.3. Cách dùng cờ

Trong đầu vào, mô hình đọc giá trị điền và cả sáu cột cờ như sáu đặc trưng (1 với ô được điền, 0 với số đo thật). Nhờ vậy mô hình biết một đoạn phẳng là do điền chứ không phải DO đứng yên thật. Cửa sổ chứa ô cờ -1 đã bị bỏ, nên trong đầu vào cờ chỉ còn 0 hoặc 1.

Trong nhãn, mốc có DO được điền không được tính sai số: sai số của một cửa sổ là trung bình trên các mốc nhãn có cờ DO bằng 0. Trong mã, cờ DO được chuyển thành một dãy mặt nạ (1 với mốc tính sai số, 0 với mốc bỏ qua); sai số tại mỗi mốc được nhân với mặt nạ rồi chia cho tổng mặt nạ. Hàm mất mát khi huấn luyện được tính theo cùng cách, nên mô hình không học theo giá trị điền.

Ví dụ với ô DO lúc 2019-07-15 10:00 (đầu vào 24 giờ, dự báo 6 giờ):

| Nhóm cửa sổ | Số cửa sổ | Vị trí ô 10:00 | Cách dùng |
|---|---|---|---|
| Thời điểm dự báo 10:15 đến 10:00 hôm sau | 96 | Đầu vào | Mô hình đọc giá trị điền 7.10 kèm cờ 1 |
| Thời điểm dự báo 04:15 đến 10:00 | 24 | Nhãn | Mốc 10:00 không được tính sai số |

## 6. Điểm DO đột biến một mốc

Điểm DO đột biến một mốc là mốc mà DO lệch quá 0.8 mg/L so với mốc trước rồi quay lại quá 0.8 mg/L ở mốc sau, ví dụ ngày 2019-08-19 lúc 23:45, 00:00, 00:15 DO là 8.89, 7.72, 8.80 mg/L. Bước nhảy như vậy hiếm: 99.9% bước thay đổi 15 phút của DO trong train nhỏ hơn 0.90 mg/L.

Train có 21 điểm đột biến. 17 điểm có nhiệt độ đổi từ 0.3 độ C hoặc pH đổi từ 0.05 cùng lúc, tức một khối nước khác đi qua cảm biến, là hiện tượng thật. 4 điểm còn lại không có biến nào giải thích và bị nghi là lỗi: 2 điểm có cảm biến khác trên cùng đầu đo trống trong vòng 1 giờ (2019-08-19 00:00, 2023-07-11 08:45), 2 điểm có thể do trạm sục khí gần đó (2021-09-08 22:15, 2021-10-28 17:30). Val có 1 điểm, là hiện tượng thật; test không có.

Cả 21 điểm được giữ là số đo thật (cờ 0), vì:
- muốn nhận ra một điểm đột biến phải có số đo ở mốc sau, nên phép phát hiện không nhân quả và không áp dụng được cho số đo mới nhất khi chạy thật;
- 4 điểm nghi lỗi chỉ nằm trong train; sửa chúng làm train được xử lý khác val và test;
- 4 trên 189888 mốc không ảnh hưởng đáng kể đến việc học.

## 7. Các xử lý không thực hiện

| Xử lý | Lý do |
|---|---|
| Làm trơn DO | Làm mất đỉnh và đáy trong ngày, là thông tin cần dự báo và cảnh báo |
| Xóa cực trị bằng ngưỡng thống kê (ví dụ bộ lọc Hampel) | Phần lớn cực trị là hiện tượng thật: đỉnh DO buổi chiều mùa hè kèm pH 8 đến 9 do quang hợp, bước nhảy độ dẫn trong đợt lũ tháng 1 năm 2019, đỉnh đơn lẻ của độ đục và chlorophyll |
| Sửa các đoạn giá trị không đổi | Các đoạn pH không đổi vào mùa đông là do độ phân giải của cảm biến (0.01) |

## 8. Tái lập

Môi trường Python khai báo trong pyproject.toml ở gốc repo. Cài một lần bằng `uv sync`, rồi chạy từ gốc repo:

```
uv run python scripts/build_base_splits.py            # tạo dataset/base_splits/ từ dataset/raw/
uv run python scripts/cleaning_study.py               # mục 3, 6: miền giá trị, bước thay đổi 15 phút, điểm DO đột biến
uv run python scripts/causal_imputation_benchmark.py  # mục 4.2, 4.3: so sánh phương pháp điền, sai số theo độ dài
uv run python scripts/gap_threshold_study.py          # mục 4.1, 4.3, 4.5: tần suất và biểu đồ, độ nhạy theo ngưỡng, đợt DO thấp bị mất
uv run python scripts/verify_dataset.py               # kiểm chứng, gồm cả các số liệu ở mục 4.5
```
