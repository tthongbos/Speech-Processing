# Mô hình 1: OmniVoice

Notebook `omnivoice_demo.ipynb` chạy OmniVoice (Zhu et al., 2026, arXiv:2604.00688) trên Kaggle: clone giọng tiếng Việt từ 3 đến 10 giây audio mẫu, so sánh số bước giải mã (8, 16, 32), thử chuẩn hoá văn bản, đo WER bằng Whisper và mở giao diện Gradio.

Trên Kaggle: Settings, Accelerator **GPU T4 x2**, **Internet: On**. Sửa `SRC_AUDIO` trong cell 2 thành đường dẫn file mẫu của bạn trong `/kaggle/input/...`, rồi Run All.

Bản tích hợp đầy đủ (cache, tiến độ theo từng bước, chuẩn hoá tiếng Việt, khoảng nghỉ `[nghỉ 1.5s]`) nằm ở `web/backend/engines/omnivoice_engine.py`; xem [web/README.md](../web/README.md).

Giấy phép: code Apache 2.0, trọng số CC-BY-NC. Chỉ clone giọng khi người đó đồng ý.
