# Speech Processing

Các demo xử lý tiếng nói bằng Python.

- [MMS TTS](mms_tts/README.md): chuyển văn bản thành WAV, mặc định tiếng Việt với `facebook/mms-tts-vie`, hỗ trợ CPU/CUDA và chọn ngôn ngữ/model từ CLI.
- [OmniVoice](omnivoice/README.md): mô hình 1, zero-shot TTS clone giọng tiếng Việt; notebook chạy trên Kaggle.
- [Vọng](web/README.md): website gộp cả hai mô hình (FastAPI + HTML/JS): thu hoặc tải giọng mẫu, chỉnh cấu hình, theo dõi tiến độ, cache 4 lớp, chuẩn hoá tiếng Việt.

```bash
cd web
python -m backend.serve --port 8000
```
