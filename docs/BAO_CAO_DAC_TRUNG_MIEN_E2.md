# Báo cáo Nghiên cứu và Thiết kế Đặc trưng Miền (Domain Features - Cấu hình E2)

**Đề tài:** Xây dựng hệ thống dự báo nồng độ oxy hòa tan (DO) trong môi trường nước sử dụng học sâu kết hợp Ontology  
**Trạm quan trắc:** CEMP Mossdale Bridge (mã trạm: `MSD`), sông San Joaquin, California, Hoa Kỳ  
**Tọa độ địa lý trạm:** Vĩ độ $\phi = 37.786^\circ\,\text{N}$, Kinh độ $\lambda = -121.306^\circ\,\text{W}$, Cao độ $z \approx 4\,\text{m}$ ($P \approx 1.0\,\text{atm}$), Múi giờ chuẩn $\text{PST} = \text{UTC}-8$  
**Tập dữ liệu nền:** `dataset/base_splits/` (Train: 2019–01–01 đến 2024–05–31, 189,888 dòng 15 phút)

---

## 1. Tổng quan Kiến trúc Đầu vào Phân tầng ($E_1 \to E_2 \to E_3$)

Để đảm bảo tính nghiêm cẩn học thuật, tránh đa cộng tuyến và phân định ranh giới rõ ràng giữa các giai đoạn đóng góp của đồ án, hệ thống dữ liệu đầu vào được phân thành **3 tầng kiến trúc**:

```
[Dữ liệu quan trắc gốc] (6 biến cảm biến + 6 cờ điền khuyết)
       │
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Cấu hình E1: Baseline thuần dữ liệu (12 chiều)                          │
│   • 6 biến cảm biến thô (DO, Temp, pH, SpC, Turb, Chl)                 │
│   • 6 cờ chỉ thị điền khuyết tương ứng (flag_*)                        │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │  + Bổ sung 19 đặc trưng miền liên tục
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Cấu hình E2: Domain Feature Engineering (31 chiều = 12 + 19)           │
│   • 2 Hóa lý cân bằng khí (do_sat_mg_l, do_deficit)                    │
│   • 2 Thiên văn & Quang hợp cửa sông (solar_insolation, photo_bzi)     │
│   • 8 Sóng hài chu kỳ & Thủy triều M2 (sin/cos tod, tod2, doy, m2)     │
│   • 5 Động học sai phân nhân quả (do_diff_1h/3h/24h, ph/temp_diff_3h)  │
│   • 2 Thống kê trượt & Dị thường ngày đêm (do_roll24h_std, anom)       │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │  + Bổ sung khái niệm & luật ngữ nghĩa
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Cấu hình E3: Ontology-Guided Learning (36+ chiều)                      │
│   • 31 đặc trưng của E2                                                │
│   • 5 cờ trạng thái sinh thái & quy chuẩn pháp lý Basin Plan           │
│   • Vector nhúng tri thức / Luật tổ hợp suy diễn từ OWL/SWRL           │
└────────────────────────────────────────────────────────────────────────┘
```

> [!NOTE]
> **Xử lý Biến Trung gian có Đa cộng tuyến cao:**
> * Độ mặn thực dụng `salinity_psu` ($S$) có hệ số tương quan $r = 0.99986$ với `spc_us_cm`. Vì vậy, $S$ được tính toán chính xác theo công thức chuẩn PSS-78 / Schemel (2001) làm **biến trung gian** để suy ra $\text{DO}_{\text{sat}}$, nhưng **không nối trực tiếp vào tensor đầu vào $X$** nhằm tránh đa cộng tuyến hoàn hảo.
> * Các cờ ngưỡng trạng thái rời rạc (`flag_supersat_bloom`, `flag_low_do_warning`, `flag_thermal_stress`, v.v.) được chuyển giao toàn bộ sang cấu hình $E_3$ (Ontology-Guided) để $E_2$ thuần nhất là các đặc trưng phương trình vật lý, hóa sinh, thiên văn và động học liên tục.

---

## 2. Chi tiết 19 Đặc trưng Miền Liên tục cho Cấu hình E2

Mỗi đặc trưng dưới đây được chuẩn hóa và giải trình đầy đủ theo 5 tiêu chí:
1. **Nó là cái gì?** (Bản chất vật lý, hóa lý, sinh thái hoặc động học)
2. **Công thức toán học chuẩn $\LaTeX$**
3. **Nguồn gốc tài liệu khoa học uy tín & cơ quan quản lý**
4. **Input cần thiết**
5. **Kiểm tra độ đầy đủ của biến đầu vào thực tế**

---

### Nhóm 1: Hóa lý & Cân bằng Khí Khí quyển (2 đặc trưng)

#### Đặc trưng 1: `do_sat_mg_l` — Nồng độ Oxy hòa tan bão hòa lý thuyết
* **Nó là cái gì:** Là nồng độ oxy hòa tan tối đa mà khối nước có thể giữ được ở trạng thái cân bằng nhiệt động học với khí quyển ở áp suất $1.0\,\text{atm}$ tại nhiệt độ và độ dẫn điện đo được. Nước càng lạnh thì khả năng giữ oxy càng cao.
* **Công thức toán học:**
  Chuyển nhiệt độ nước sang thang Kelvin:
  $$T_K = T_w + 273.15$$
  Logarit tự nhiên của nồng độ bão hòa đối với nước ngọt tinh khiết:
  $$\ln(\text{DO}_{\text{fresh}}) = -139.34411 + \frac{1.575701 \times 10^5}{T_K} - \frac{6.642308 \times 10^7}{T_K^2} + \frac{1.243800 \times 10^{10}}{T_K^3} - \frac{8.621949 \times 10^{11}}{T_K^4}$$
  Hệ số hiệu chỉnh độ mặn do hiệu ứng đẩy khí của ion muối (*salting-out effect*):
  $$F_S = S \cdot \left(0.017674 - \frac{10.754}{T_K} + \frac{2140.7}{T_K^2}\right)$$
  với độ mặn thực dụng $S$ ($\text{PSU}$) tính từ tỷ số độ dẫn $R = \frac{\text{spc\_us\_cm}}{53087.0}$ qua đa thức bậc 5 PSS-78:
  $$S = 0.0120 - 0.2174\,R^{1/2} + 25.3283\,R + 13.7714\,R^{3/2} - 6.4788\,R^2 + 2.5842\,R^{5/2}$$
  Nồng độ bão hòa cuối cùng:
  $$\text{DO}_{\text{sat}} = \exp\left(\ln(\text{DO}_{\text{fresh}}) - F_S\right) \quad (\text{mg/L})$$
* **Truy vết nguồn tài liệu:**
  * **Benson, B.B., & Krause, D. (1984)**, *The concentration and isotopic fractionation of oxygen dissolved in freshwater and seawater in equilibrium with the atmosphere*, **Limnology and Oceanography**, 29(3), 620–632.
  * **USGS (2011)**, *Office of Water Quality Technical Memorandum 2011.03* (Quy chuẩn liên bang Hoa Kỳ thay thế Weiss 1970).
  * **APHA (2017)**, *Standard Methods for the Examination of Water and Wastewater*, Method 4500-O G.
  * **Schemel, L.E. (2001 - USGS)**, *Simplified Conversions Between Specific Conductance and Salinity Units for Use with Data from Monitoring Stations*, IEP Newsletter, 14(1), 17–18.
* **Input cần thiết:** `water_temp_c` ($^\circ\text{C}$), `spc_us_cm` ($\mu\text{S/cm}$).
* **Kiểm tra độ đầy đủ:** **ĐỦ 100%**. Có sẵn trực tiếp từ cảm biến trạm Mossdale; tại $20\,^\circ\text{C}, S=0$ công thức cho đúng $9.092\,\text{mg/L}$ khớp bảng APHA.

---

#### Đặc trưng 2: `do_deficit` — Độ thiếu hụt Oxy hòa tan
* **Nó là cái gì:** Là hiệu số giữa nồng độ bão hòa lý thuyết và nồng độ oxy thực tế trong nước ($D = \text{DO}_{\text{sat}} - \text{DO}$). Đây là gradient áp suất quyết định chiều và lực trao đổi khí: khi $D > 0$ nước thiếu oxy và hút oxy từ không khí vào; khi $D < 0$ nước quá bão hòa do quang hợp và oxy thoát ngược ra ngoài khí quyển.
* **Công thức toán học:**
  $$D = \text{do\_deficit} = \text{DO}_{\text{sat}} - \text{DO} \quad (\text{mg/L})$$
* **Truy vết nguồn tài liệu:**
  * **Streeter, H.W., & Phelps, E.B. (1925)**, *A Study of the Pollution and Natural Purification of the Ohio River*, **U.S. Public Health Service, Public Health Bulletin No. 146**.
  * **Wagner et al. (2006)**, *Guidelines and standard procedures for continuous water-quality monitors*, **USGS Techniques and Methods 1-D3**.
* **Input cần thiết:** `do_mg_l`, `do_sat_mg_l` (tính từ đặc trưng 1).
* **Kiểm tra độ đầy đủ:** **ĐỦ 100%**. Có cùng đơn vị $\text{mg/L}$ với nhãn dự báo, giúp mạng nơ-ron học mối quan hệ cộng tính $\text{DO} = \text{DO}_{\text{sat}} - D$.

---

### Nhóm 2: Thiên văn & Quang hợp Sinh thái Cửa sông (2 đặc trưng)

#### Đặc trưng 3: `solar_insolation` — Chỉ số Bức xạ Quang thông Thiên văn Mặt nước
* **Nó là cái gì:** Là giá trị sine của góc nâng Mặt Trời phía trên đường chân trời (cắt ngưỡng bằng 0 khi Mặt Trời lặn), đại diện cho quang thông bức xạ tiềm năng chiếu xuống mặt sông khi trời quang. Biến này tự động đo chính xác độ dài ngày chiếu sáng biến thiên theo mùa tại California (14.8 giờ vào hạ chí, 9.5 giờ vào đông chí).
* **Công thức toán học:**
  Góc ngày phân đoạn $\gamma$ (radian) từ ngày trong năm $\text{DOY} \in [1, 366]$ và giờ thập phân $t_{\text{hour}} = \text{hour} + \frac{\text{minute}}{60}$:
  $$\gamma = \frac{2\pi}{N_{\text{year}}} \left(\text{DOY} - 1 + \frac{t_{\text{hour}} - 12}{24}\right) \quad (N_{\text{year}} = 366 \text{ nếu nhuận, ngược lại } 365)$$
  Góc xích vĩ Mặt Trời $\delta$ và phương trình thời gian $E_t$ theo Spencer (1971):
  $$\begin{aligned}
  \delta = 0.006918 &- 0.399912\cos\gamma + 0.070257\sin\gamma - 0.006758\cos(2\gamma) \\
  &+ 0.000907\sin(2\gamma) - 0.002697\cos(3\gamma) + 0.00148\sin(3\gamma)
  \end{aligned}$$
  $$E_t = 229.18 \cdot \left(0.000075 + 0.001868\cos\gamma - 0.032077\sin\gamma - 0.014615\cos(2\gamma) - 0.040849\sin(2\gamma)\right)$$
  Giờ Mặt Trời thực và góc giờ $\omega$ tại kinh độ trạm $\lambda = -121.306^\circ$ so với kinh tuyến chuẩn PST $\lambda_{\text{ref}} = -120.0^\circ$:
  $$t_{\text{solar}} = 60 \cdot t_{\text{hour}} + E_t + 4 \cdot \left(\lambda - \lambda_{\text{ref}}\right) \quad (\text{phút}), \qquad \omega = \left(\frac{t_{\text{solar}}}{4} - 180^\circ\right) \times \frac{\pi}{180^\circ}$$
  Sine góc nâng Mặt Trời $\alpha_s$ tại vĩ độ trạm $\phi = 37.786^\circ\,\text{N}$:
  $$\sin\alpha_s = \sin\phi \sin\delta + \cos\phi \cos\delta \cos\omega$$
  $$I_{\text{solar}} = \max\left(0,\, \sin\alpha_s\right)$$
* **Truy vết nguồn tài liệu:**
  * **Spencer, J.W. (1971)**, *Fourier series representation of the position of the sun*, **Search**, 2(5), 172.
  * **NOAA Global Monitoring Laboratory**, *General Solar Position Calculations & Astronomical Algorithms*.
* **Input cần thiết:** `datetime_pst`, vĩ độ trạm $\phi = 37.786^\circ\,\text{N}$, kinh độ trạm $\lambda = -121.306^\circ\,\text{W}$.
* **Kiểm tra độ đầy đủ:** **ĐỦ 100%**. Tính hoàn toàn từ trường thời gian và tọa độ địa lý cố định của trạm Mossdale trong metadata.

---

#### Đặc trưng 4: `photo_bzi` — Tiềm năng Quang hợp Cửa sông theo Mô hình $B \cdot Z_p \cdot I_0$
* **Nó là cái gì:** Đại diện cho năng suất quang hợp thực tế của thực vật phù du trong cột nước sông San Joaquin. Nó kết hợp 3 yếu tố then chốt: sinh khối tảo ($B$), độ sâu tầng quang hợp ($Z_p$, bị giới hạn bởi độ đục phù sa), và bức xạ mặt trời khả dụng ($I_0$).
* **Công thức toán học:**
  Theo mô hình thực nghiệm của Cole & Cloern (1987) và phương trình suy giảm quang thông do chất lơ lửng tại vịnh San Francisco của Cloern (1987):
  $$k_d = 0.77 + 0.06 \cdot \text{Turb} \quad (\text{hệ số tắt ánh sáng PAR, m}^{-1})$$
  $$Z_p = \frac{\ln 100}{k_d} \approx \frac{4.61}{0.77 + 0.06 \cdot \text{Turb}} \quad (\text{độ sâu tầng nước có ánh sáng, m})$$
  Tham số tổng hợp quang hợp và dạng logarit nén dải động:
  $$P_{\text{BZI}} = \text{Chl} \cdot Z_p \cdot I_{\text{solar}} = \text{chl\_fluor\_ug\_l} \cdot \left(\frac{4.61}{0.77 + 0.06 \cdot \text{turbidity\_fnu}}\right) \cdot I_{\text{solar}}$$
  $$\text{photo\_bzi} = \ln\left(1 + P_{\text{BZI}}\right)$$
* **Truy vết nguồn tài liệu:**
  * **Cole, B.E., & Cloern, J.E. (1987)**, *An empirical model for estimating phytoplankton productivity in estuaries*, **Marine Ecology Progress Series**, 36, 299–305.
  * **Cloern, J.E. (1987)**, *Turbidity as a control on phytoplankton biomass and productivity in estuaries*, **Continental Shelf Research**, 7(11–12), 1367–1381.
  * **Jassby, A.D., Cloern, J.E., & Cole, B.E. (2002)**, *Annual primary production: Patterns and mechanisms of change in a nutrient-rich tidal ecosystem (Sacramento–San Joaquin Delta)*, **Limnology and Oceanography**, 47(3), 698–712.
* **Input cần thiết:** `chl_fluor_ug_l`, `turbidity_fnu`, `solar_insolation` (từ đặc trưng 3).
* **Kiểm tra độ đầy đủ:** **ĐỦ 100%**. Thay thế công thức tự chế thiếu nguồn bằng mô hình khoa học có xuất bản chính thức tại chính hệ sinh thái Delta, đạt tương quan riêng phần $r_{\text{partial}} = +0.416$.

---

### Nhóm 3: Chu kỳ Sóng hài Thời gian & Thủy triều Cửa sông (8 đặc trưng)

#### Đặc trưng 5 & 6: `sin_tod`, `cos_tod` — Sóng hài Ngày Đêm Bậc 1 (Chu kỳ 24.0 giờ)
* **Nó là cái gì:** Cặp mã hóa lượng giác thể hiện vị trí liên tục trong ngày đêm 24 giờ của Trái Đất, giúp mạng học sâu hiểu được tính đóng vòng của thời gian (23:45 và 00:00 liền kề nhau).
* **Công thức toán học:**
  $$\sin_{\text{tod}} = \sin\left(\frac{2\pi \cdot t_{\text{hour}}}{24.0}\right), \qquad \cos_{\text{tod}} = \cos\left(\frac{2\pi \cdot t_{\text{hour}}}{24.0}\right)$$
  với $t_{\text{hour}} = \text{hour} + \frac{\text{minute}}{60.0} \in [0.0, 24.0)$.
* **Truy vết nguồn tài liệu:** Chu kỳ ngày đêm tự quay của Trái Đất; kỹ thuật chuẩn trong mô hình hóa chuỗi thời gian khí tượng - thủy văn.
* **Input cần thiết:** `datetime_pst`.
* **Kiểm tra độ đầy đủ:** **ĐỦ 100%**.

---

#### Đặc trưng 7 & 8: `sin_tod2`, `cos_tod2` — Sóng hài Ngày Đêm Bậc 2 (Họa tần 12.0 giờ)
* **Nó là cái gì:** Là họa tần bậc hai của chu kỳ ngày đêm. Trong tự nhiên, đường cong DO không phải là hình sin đối xứng: DO tăng vọt rất dốc trong 4–5 tiếng buổi trưa chiều khi tảo quang hợp, nhưng lại suy giảm thoai thoải suốt 14–15 tiếng ban đêm do hô hấp. Cặp họa tần 12h kết hợp với bậc 1 (24h) cho phép chuỗi Fourier tái tạo chính xác dạng sóng phi đối xứng này.
* **Công thức toán học:**
  $$\sin_{\text{tod2}} = \sin\left(\frac{4\pi \cdot t_{\text{hour}}}{24.0}\right), \qquad \cos_{\text{tod2}} = \cos\left(\frac{4\pi \cdot t_{\text{hour}}}{24.0}\right)$$
* **Truy vết nguồn tài liệu:** Phân tích điều hòa chuỗi Fourier (Fourier harmonic expansion) cho chu kỳ ngày đêm thủy sinh vật.
* **Input cần thiết:** `datetime_pst`.
* **Kiểm tra độ đầy đủ:** **ĐỦ 100%**.

---

#### Đặc trưng 9 & 10: `sin_doy`, `cos_doy` — Sóng hài Chu kỳ Mùa trong Năm (Chu kỳ 365.25 ngày)
* **Nó là cái gì:** Cặp mã hóa lượng giác thể hiện tiến trình mùa trong năm thiên văn ($365.25$ ngày), nắm bắt chu kỳ biến thiên nền của nhiệt độ nước, chu kỳ dinh dưỡng mùa kiệt và các đợt xả lũ mùa đông-xuân tại California.
* **Công thức toán học:**
  $$\sin_{\text{doy}} = \sin\left(\frac{2\pi \cdot \text{DOY}}{365.25}\right), \qquad \cos_{\text{doy}} = \cos\left(\frac{2\pi \cdot \text{DOY}}{365.25}\right)$$
  với $\text{DOY}$ là ngày thứ mấy trong năm ($\text{Day of Year} \in [1, 366]$).
* **Truy vết nguồn tài liệu:** Chu kỳ quỹ đạo Trái Đất quanh Mặt Trời.
* **Input cần thiết:** `datetime_pst`.
* **Kiểm tra độ đầy đủ:** **ĐỦ 100%**.

---

#### Đặc trưng 11 & 12: `sin_m2`, `cos_m2` — Sóng hài Bán nhật triều Mặt Trăng $M_2$ (Chu kỳ 12.4206 giờ)
* **Nó là cái gì:** Sóng hài đại diện cho thành phần thủy triều Mặt Trăng chính ($M_2$), chi phối dao động mực nước và sự chuyển động qua lại của khối nước vùng cửa sông San Joaquin. Phân tích phổ FFT trên dữ liệu thực tế tại trạm Mossdale xác nhận đỉnh năng lượng phổ tại chu kỳ $12.42\,\text{h}$ cao gấp 18 lần nền ở biến `spc` và gấp 145 lần nền ở biến `do_mg_l`.
* **Công thức toán học:**
  $$\sin_{M_2} = \sin\left(\frac{2\pi \cdot t_{\text{hours\_total}}}{12.4206}\right), \qquad \cos_{M_2} = \cos\left(\frac{2\pi \cdot t_{\text{hours\_total}}}{12.4206}\right)$$
  với $t_{\text{hours\_total}}$ là tổng số giờ trôi qua tính từ mốc thời gian gốc cố định chung cho toàn bộ tập dữ liệu: $t_0 = \text{2019-01-01 00:00:00 PST}$.
* **Truy vết nguồn tài liệu:**
  * **NOAA Center for Operational Oceanographic Products and Services (CO-OPS)**, *Tidal Analysis and Predictions, Harmonic Constituents*.
  * **USGS Streamgage 11304200 (San Joaquin River at Mossdale Bridge)**: Xác định Mossdale là ranh giới trên cùng của vùng chịu tác động triều (*upstream limit of tidal Delta*).
* **Input cần thiết:** `datetime_pst` (tính giờ liên tục từ $t_0$).
* **Kiểm tra độ đầy đủ:** **ĐỦ 100%**. Mốc thời gian liên tục 15 phút, không ngắt quãng.

---

### Nhóm 4: Động học Sai phân Nhân quả (5 đặc trưng)

Toàn bộ các đặc trưng sai phân tuân thủ nghiêm ngặt **Nguyên tắc Nhân quả Thời gian** ($t - \tau$), chỉ nhìn về quá khứ:

#### Đặc trưng 13: `do_diff_1h` — Biến thiên DO trong 1 giờ qua
* **Nó là cái gì:** Tốc độ biến thiên tức thời của nồng độ DO qua 4 mốc quan trắc gần nhất, đo độ dốc tăng vọt của pha quang hợp hoặc sụt giảm nhanh ban đêm.
* **Công thức toán học:**
  $$\Delta \text{DO}_{1\text{h}}(t) = \text{DO}(t) - \text{DO}(t - 4)$$
* **Truy vết nguồn tài liệu:** Đạo hàm bậc 1 rời rạc (Backward finite difference) của phương trình Odum (1956).
* **Input cần thiết:** `do_mg_l`.
* **Kiểm tra độ đầy đủ:** **ĐỦ 100%**.

---

#### Đặc trưng 14: `do_diff_3h` — Biến thiên DO trong 3 giờ qua
* **Nó là cái gì:** Tốc độ biến thiên DO trên thang thời gian trung bình (12 mốc), nắm bắt quán tính thay đổi nửa buổi của quá trình sinh hóa mà không bị ảnh hưởng bởi nhiễu cảm biến tần số cao.
* **Công thức toán học:**
  $$\Delta \text{DO}_{3\text{h}}(t) = \text{DO}(t) - \text{DO}(t - 12)$$
* **Truy vết nguồn tài liệu:** Phân tích động lực học oxy hòa tan dòng sông (Chapra, 1997).
* **Input cần thiết:** `do_mg_l`.
* **Kiểm tra độ đầy đủ:** **ĐỦ 100%**.

---

#### Đặc trưng 15: `do_diff_24h` — Biến thiên DO so với cùng giờ hôm trước
* **Nó là cái gì:** Hiệu số giữa DO hiện tại và DO đúng thời điểm này 24 giờ trước (96 mốc). Biến này loại bỏ hoàn toàn chu kỳ ngày đêm tự nhiên để phản ánh xu thế tích lũy/suy thoái oxy ngày-qua-ngày (do một đợt nắng nóng kéo dài, đợt bùng phát tảo mới, hoặc nước đục sau mưa lũ).
* **Công thức toán học:**
  $$\Delta \text{DO}_{24\text{h}}(t) = \text{DO}(t) - \text{DO}(t - 96)$$
* **Truy vết nguồn tài liệu:** Sai phân chu kỳ mùa ngày đêm (Seasonal differencing lag-24h) trong phân tích chuỗi thời gian thủy văn.
* **Input cần thiết:** `do_mg_l`.
* **Kiểm tra độ đầy đủ:** **ĐỦ 100%**.

---

#### Đặc trưng 16: `ph_diff_3h` — Biến thiên pH trong 3 giờ qua
* **Nó là cái gì:** Tốc độ biến thiên độ pH trong 3 giờ qua (12 mốc). Khi tảo quang hợp mạnh rút $\text{CO}_2$, pH tăng rất nhanh; ngược lại khi hô hấp chiếm ưu thế nhả $\text{CO}_2$, pH giảm. Biến này là tín hiệu kiểm chứng hóa học đồng pha cực mạnh hỗ trợ dự báo xu thế oxy.
* **Công thức toán học:**
  $$\Delta \text{pH}_{3\text{h}}(t) = \text{pH}(t) - \text{pH}(t - 12)$$
* **Truy vết nguồn tài liệu:** Cân bằng hệ đệm cacbonat trong thủy vực tự nhiên (Stumm & Morgan, 1996; Kratzer et al., 2004).
* **Input cần thiết:** `ph`.
* **Kiểm tra độ đầy đủ:** **ĐỦ 100%**.

---

#### Đặc trưng 17: `temp_diff_3h` — Biến thiên Nhiệt độ nước trong 3 giờ qua
* **Nó là cái gì:** Tốc độ làm nóng hoặc làm mát của nước sông trong 3 giờ qua (12 mốc), phản ánh cường độ hấp thụ nhiệt từ bức xạ mặt trời hoặc pha làm lạnh nhanh ban đêm.
* **Công thức toán học:**
  $$\Delta T_{3\text{h}}(t) = T_w(t) - T_w(t - 12)$$
* **Truy vết nguồn tài liệu:** Cân bằng nhiệt mặt nước sông (Surface heat exchange modeling — Chapra, 1997).
* **Input cần thiết:** `water_temp_c`.
* **Kiểm tra độ đầy đủ:** **ĐỦ 100%**.

---

### Nhóm 5: Thống kê Trượt Nhân quả & Dị thường Ngày đêm 24 giờ (2 đặc trưng)

Tính toán trên cửa sổ trượt nhân quả đúng 96 bước 15 phút của 24 giờ quá khứ $[t - 95, \dots, t]$:

#### Đặc trưng 18: `do_roll24h_std` — Độ lệch chuẩn trượt 24 giờ của DO (Biên độ dao động ngày đêm)
* **Nó là cái gì:** Là độ phân tán của DO trong 24 giờ vừa qua, phản ánh trực tiếp biên độ dao động ngày đêm (*Diel DO Amplitude*). Trong mùa đông tảo ít hoạt động, biên độ ngày đêm rất nhỏ ($\sigma \approx 0.3 - 0.5\,\text{mg/L}$); trong mùa hè tảo nở hoa quang hợp cực thịnh, biên độ vọt lên rất lớn ($\sigma > 2.0 - 3.5\,\text{mg/L}$). Đây là chỉ thị then chốt về cường độ trao đổi chất của hệ sinh thái.
* **Công thức toán học:**
  $$\mu_{\text{DO}, 24\text{h}}(t) = \frac{1}{96} \sum_{i=0}^{95} \text{DO}(t - i)$$
  $$\sigma_{\text{DO}, 24\text{h}}(t) = \sqrt{\frac{1}{95} \sum_{i=0}^{95} \left(\text{DO}(t - i) - \mu_{\text{DO}, 24\text{h}}(t)\right)^2}$$
* **Truy vết nguồn tài liệu:** Phương pháp ước tính trao đổi chất dòng sông bằng biên độ dao động oxy ngày đêm (Odum, 1956; Chapra, 1997).
* **Input cần thiết:** `do_mg_l`.
* **Kiểm tra độ đầy đủ:** **ĐỦ 100%**.

---

#### Đặc trưng 19: `do_diel_anomaly` — Độ dị thường Ngày đêm của Oxy hòa tan
* **Nó là cái gì:** Là khoảng cách chênh lệch giữa nồng độ DO tức thời tại thời điểm $t$ so với mức trung bình 24 giờ qua của chính nó. 
  * Nếu $\text{do\_diel\_anomaly} \gg 0$: Nước đang nằm ở đỉnh quang hợp buổi chiều; theo quy luật hồi quy tự nhiên, trong 6 giờ tới DO bắt buộc phải quay đầu sụt giảm mạnh.
  * Nếu $\text{do\_diel\_anomaly} \ll 0$: Nước đang nằm ở đáy hô hấp sáng sớm; trong 6 giờ tới khi mặt trời lên, DO bắt buộc phải đảo chiều tăng vọt.
  * Đây là **đặc trưng dự báo mạnh số 1** của toàn bộ bài toán ($r_{\text{partial}} = -0.556$ sau khi đã loại trừ ảnh hưởng của giá trị DO hiện tại).
* **Công thức toán học:**
  $$\text{DO}_{\text{diel\_anomaly}}(t) = \text{DO}(t) - \mu_{\text{DO}, 24\text{h}}(t) = \text{DO}(t) - \frac{1}{96} \sum_{i=0}^{95} \text{DO}(t - i)$$
* **Truy vết nguồn tài liệu:** Kỹ thuật bóc tách thành phần dị thường ngày đêm trong phân tích chuỗi thủy sinh học cửa sông (Estuarine diel anomaly decomposition).
* **Input cần thiết:** `do_mg_l`.
* **Kiểm tra độ đầy đủ:** **ĐỦ 100%**.

---

## 3. Bảng Tổng hợp và Đối chiếu 19 Đặc trưng Miền Liên tục ($E_2$)

| STT | Tên đặc trưng | Phân nhóm | Bản chất khoa học | Nguồn tài liệu chính | Biến đầu vào | Tình trạng |
|:---:|---|---|---|---|---|:---:|
| 1 | **`do_sat_mg_l`** | Hóa lý | Nồng độ DO bão hòa lý thuyết ở 1 atm | Benson & Krause (1984); USGS (2011) | `water_temp_c`, `spc_us_cm` | **Đủ 100%** |
| 2 | **`do_deficit`** | Hóa lý | Độ thiếu hụt oxy hòa tan ($D = \text{DO}_{\text{sat}} - \text{DO}$) | Streeter & Phelps (1925); USGS (2006) | `do_mg_l`, `do_sat_mg_l` | **Đủ 100%** |
| 3 | **`solar_insolation`** | Thiên văn | Bức xạ thiên văn mặt nước ($I_{\text{solar}} = \max(0, \sin\alpha_s)$) | Spencer (1971); NOAA Solar Calc | `datetime_pst`, Vĩ độ/Kinh độ | **Đủ 100%** |
| 4 | **`photo_bzi`** | Sinh thái | Tiềm năng quang hợp cửa sông có cản sáng độ đục | Cole & Cloern (1987); Cloern (1987) | `chl`, `turb`, `solar_insolation` | **Đủ 100%** |
| 5 | **`sin_tod`** | Chu kỳ | Sóng hài ngày đêm bậc 1 (Sine, chu kỳ 24h) | Chu kỳ Trái Đất tự quay | `datetime_pst` | **Đủ 100%** |
| 6 | **`cos_tod`** | Chu kỳ | Sóng hài ngày đêm bậc 1 (Cosine, chu kỳ 24h) | Chu kỳ Trái Đất tự quay | `datetime_pst` | **Đủ 100%** |
| 7 | **`sin_tod2`** | Chu kỳ | Sóng hài ngày đêm bậc 2 (Sine, chu kỳ 12h) | Họa tần Fourier bắt sóng phi đối xứng | `datetime_pst` | **Đủ 100%** |
| 8 | **`cos_tod2`** | Chu kỳ | Sóng hài ngày đêm bậc 2 (Cosine, chu kỳ 12h) | Họa tần Fourier bắt sóng phi đối xứng | `datetime_pst` | **Đủ 100%** |
| 9 | **`sin_doy`** | Chu kỳ | Sóng hài mùa trong năm (Sine, chu kỳ 365.25 ngày) | Chu kỳ quỹ đạo Trái Đất quanh Mặt Trời | `datetime_pst` | **Đủ 100%** |
| 10 | **`cos_doy`** | Chu kỳ | Sóng hài mùa trong năm (Cosine, chu kỳ 365.25 ngày) | Chu kỳ quỹ đạo Trái Đất quanh Mặt Trời | `datetime_pst` | **Đủ 100%** |
| 11 | **`sin_m2`** | Thủy triều | Sóng triều bán nhật triều Mặt Trăng (Sine, 12.42h) | NOAA CO-OPS Tidal Constituents | `datetime_pst` (từ mốc cố định) | **Đủ 100%** |
| 12 | **`cos_m2`** | Thủy triều | Sóng triều bán nhật triều Mặt Trăng (Cosine, 12.42h) | NOAA CO-OPS Tidal Constituents | `datetime_pst` (từ mốc cố định) | **Đủ 100%** |
| 13 | **`do_diff_1h`** | Động học | Tốc độ biến thiên DO trong 1 giờ qua (4 mốc) | Sai phân lùi nhân quả; Odum (1956) | `do_mg_l` | **Đủ 100%** |
| 14 | **`do_diff_3h`** | Động học | Tốc độ biến thiên DO trong 3 giờ qua (12 mốc) | Động lực học oxy; Chapra (1997) | `do_mg_l` | **Đủ 100%** |
| 15 | **`do_diff_24h`** | Động học | Biến thiên DO so với cùng giờ hôm trước (96 mốc) | Sai phân mùa ngày đêm (Seasonal differencing) | `do_mg_l` | **Đủ 100%** |
| 16 | **`ph_diff_3h`** | Động học | Tốc độ biến thiên pH trong 3 giờ qua (12 mốc) | Động học hệ đệm cacbonat đồng pha quang hợp | `ph` | **Đủ 100%** |
| 17 | **`temp_diff_3h`** | Động học | Tốc độ tích nhiệt cột nước trong 3 giờ qua (12 mốc) | Cân bằng nhiệt mặt nước; Chapra (1997) | `water_temp_c` | **Đủ 100%** |
| 18 | **`do_roll24h_std`** | Thống kê | Biên độ dao động ngày đêm DO qua độ lệch chuẩn 24h | Đo cường độ trao đổi chất; Odum (1956) | `do_mg_l` | **Đủ 100%** |
| 19 | **`do_diel_anomaly`** | Thống kê | Độ dị thường của DO so với trung bình 24h qua | Chỉ số hồi quy trung bình ($r_{\text{partial}} = -0.556$) | `do_mg_l` | **Đủ 100%** |

---

## 4. Danh mục Tài liệu Khoa học Uy tín Trích dẫn Trực tiếp

1. **[Benson & Krause, 1984]** Benson, B. B., & Krause, D. (1984). *The concentration and isotopic fractionation of oxygen dissolved in freshwater and seawater in equilibrium with the atmosphere*. **Limnology and Oceanography**, 29(3), 620–632. DOI: [10.4319/lo.1984.29.3.0620](https://doi.org/10.4319/lo.1984.29.3.0620).
2. **[USGS, 2011]** U.S. Geological Survey (2011). *Change to Solubility Equations for Oxygen in Water*. **Office of Water Quality Technical Memorandum 2011.03**. Reston, VA: USGS.
3. **[APHA, 2017]** American Public Health Association (APHA), AWWA, & WEF (2017). *Standard Methods for the Examination of Water and Wastewater* (23rd ed.), **Method 4500-O: Oxygen (Dissolved)**. Washington, DC.
4. **[Schemel – USGS, 2001]** Schemel, L. E. (2001). *Simplified conversions between specific conductance and salinity units for use with data from monitoring stations*. **Interagency Ecological Program (IEP) Newsletter**, 14(1), 17–18.
5. **[Streeter & Phelps, 1925]** Streeter, H. W., & Phelps, E. B. (1925). *A Study of the Pollution and Natural Purification of the Ohio River*. **U.S. Public Health Service, Public Health Bulletin No. 146**.
6. **[Wagner et al. – USGS, 2006]** Wagner, R. J., Boulger, R. W. Jr., Oblinger, C. J., & Smith, B. A. (2006). *Guidelines and standard procedures for continuous water-quality monitors: Station operation, record computation, and data reporting*. **USGS Techniques and Methods 1-D3**, 51 pp.
7. **[Spencer, 1971 & NOAA]** Spencer, J. W. (1971). *Fourier series representation of the position of the sun*. **Search**, 2(5), 172; và *NOAA Global Monitoring Laboratory Solar Calculation Details*.
8. **[Cole & Cloern, 1987]** Cole, B. E., & Cloern, J. E. (1987). *An empirical model for estimating phytoplankton productivity in estuaries*. **Marine Ecology Progress Series**, 36, 299–305.
9. **[Cloern, 1987]** Cloern, J. E. (1987). *Turbidity as a control on phytoplankton biomass and productivity in estuaries*. **Continental Shelf Research**, 7(11–12), 1367–1381.
10. **[Jassby et al., 2002]** Jassby, A. D., Cloern, J. E., & Cole, B. E. (2002). *Annual primary production: Patterns and mechanisms of change in a nutrient-rich tidal ecosystem*. **Limnology and Oceanography**, 47(3), 698–712.
11. **[Odum, 1956]** Odum, H. T. (1956). *Primary production in flowing waters*. **Limnology and Oceanography**, 1(2), 102–117.
12. **[Chapra, 1997]** Chapra, S. C. (1997). *Surface Water-Quality Modeling*. McGraw-Hill.
13. **[Kratzer et al. – USGS, 2004]** Kratzer, C. R., et al. (2004). *Sources and Transport of Nutrients, Organic Carbon, and Chlorophyll-a in the San Joaquin River Upstream of Vernalis, California, during Summer and Fall, 2000 and 2001*. **USGS Water-Resources Investigations Report 03-4127**.
14. **[Stumm & Morgan, 1996]** Stumm, W., & Morgan, J. J. (1996). *Aquatic Chemistry: Chemical Equilibria and Rates in Natural Waters* (3rd ed.). John Wiley & Sons.
