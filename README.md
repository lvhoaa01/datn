# Đồ án dự báo DO

Dữ liệu, báo cáo và mã audit được lưu tập trung tại [candidate_dataset](candidate_dataset/README.md).

- [Thứ tự dataset ứng viên và hướng dẫn](candidate_dataset/README.md)
- [Đề cương gốc](candidate_dataset/_docs/Hoa-Thinh-ver2.docx)
- [ZIP CSU 2025 gốc](candidate_dataset/_sources/2025.zip)
- [Thống kê dữ liệu](candidate_dataset/_audit/audit_summary.csv)

Chạy lại audit từ thư mục dự án:

```bash
python candidate_dataset/_audit/audit_datasets.py
```

Sau khi chủ động cập nhật dữ liệu hoặc chạy lại audit, cần cập nhật manifest trước khi dùng nó xác minh phiên bản mới. Kiểm tra snapshot đang lưu:

```bash
python candidate_dataset/verify_snapshot.py
```
