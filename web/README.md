# Vọng: studio nhân bản giọng nói tiếng Việt

Website demo cho đồ án Xử lý tiếng nói (hướng ứng dụng, chủ đề Tổng hợp tiếng nói). Người dùng thu âm hoặc tải lên 3 đến 10 giây giọng mẫu, nhập văn bản, chỉnh cấu hình, theo dõi tiến độ từng bước giải mã và nghe kết quả đọc bằng chính giọng đó.

- **Mô hình 1:** OmniVoice (Zhu et al., 2026, arXiv:2604.00688), zero-shot TTS bằng diffusion language model.
- **Mô hình 2:** MMS-TTS (Pratap et al., 2023), checkpoint `facebook/mms-tts-vie`, kiến trúc VITS, lấy từ `mms_tts/` của nhóm. Một giọng tiếng Việt có sẵn, không clone giọng, 16 kHz. Xem [Mô hình 2: MMS-TTS](#mô-hình-2-mms-tts).

## Cấu trúc

Mọi lệnh bên dưới chạy trong thư mục `web/` của repo.

```
backend/
  app.py              API FastAPI + phục vụ giao diện tĩnh
  serve.py            python -m backend.serve [--port 8000] [--mock]
  jobs.py             hàng đợi GPU một worker, tiến độ theo từng bước, huỷ giữa chừng
  store.py            4 lớp cache trên đĩa + RAM
  segment.py          tách câu để sinh theo lô và cache theo câu
  textnorm_vi.py      chuẩn hoá tiếng Việt: số, %, °C, ngày giờ, viết tắt
  audio.py            ffmpeg giải mã mọi định dạng, ghép câu
  engines/
    base.py           giao diện chung TTSEngine + tham số
    omnivoice_engine.py
    mms_engine.py     mô hình 2: MMS-TTS (VITS), không cần giọng mẫu
    mock_engine.py    engine mô phỏng để làm giao diện không cần GPU
frontend/             HTML, CSS, JS thuần (không cần build)
kaggle/vong_kaggle.ipynb   chạy toàn bộ website trên GPU Kaggle, có link công khai
```

## Chạy

### Trên Kaggle (GPU, khuyên dùng)

1. Tải thư mục dự án (ít nhất `backend/`, `frontend/`, `requirements.txt`) lên làm một Kaggle Dataset.
2. Mở `kaggle/vong_kaggle.ipynb` trên Kaggle, Add Data dataset vừa tạo, bật **GPU T4 x2** và **Internet: On**.
3. Run All. Cell cuối in ra link `https://….trycloudflare.com`, mở link đó trên trình duyệt.

Lần đầu mất vài phút để tải trọng số OmniVoice (và Vọng nếu để trống lời giọng mẫu).

### Trên Mac Apple Silicon (GPU Apple qua MPS)

Đã chạy thử trên M3 Pro 18 GB: nạp model khoảng 3,5 phút lần đầu (phần lớn là tải trọng số), 16 bước sinh 12,5 giây audio trong khoảng 13 giây (RTF ≈ 1,0, float32). Lần đầu để trống lời mẫu sẽ tải thêm Vọng.

```bash
uv venv --python python3.10 .venv
```

```bash
uv pip install --python .venv/bin/python -r requirements.txt omnivoice
```

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m backend.serve --port 8000
```

Server tự chọn `cuda:0`, rồi `mps`, rồi CPU. `VONG_DTYPE=float16` có thể nhanh hơn trên MPS nhưng chưa được kiểm tra.

### Trên máy có GPU NVIDIA (Python 3.10 trở lên)

```bash
python -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements.txt omnivoice
```

```bash
python -m backend.serve --port 8000
```

Mở http://localhost:8000. Cần `ffmpeg` trong PATH để đọc m4a, mp3, webm.

### Chế độ mô phỏng (máy không có GPU, để làm giao diện)

```bash
pip install -r requirements.txt
```

```bash
python -m backend.serve --mock --port 8000
```

Engine mô phỏng có cùng tham số và cùng số bước tiến độ như OmniVoice nhưng trả về âm thanh tổng hợp giả. Mọi phần còn lại (upload, thu âm, cache, lịch sử) chạy thật.

Biến môi trường: `VONG_MOCK` (`auto`, `1`, `0`), `VONG_DEVICE` (mặc định `auto`), `VONG_DTYPE`, `VONG_CACHE_DIR`, `VONG_MAX_TEXT_CHARS`, `VONG_MAX_REF_SECONDS`.

## Tính năng

| Yêu cầu | Cách làm |
| --- | --- |
| Giọng mẫu: thu âm hoặc tải file | Tab *Tải file lên* (kéo thả, m4a, mp3, wav, ogg, flac, aac, webm, opus) và *Thu âm* (MediaRecorder, đo âm lượng trực tiếp, tự dừng ở 30 giây). Server dùng ffmpeg chuyển về WAV mono 24 kHz. Tab *Đã lưu* dùng lại giọng cũ. |
| Nhập văn bản | Ô văn bản có đếm ký tự, số câu sẽ sinh, ước lượng độ dài audio, xem trước văn bản sau chuẩn hoá. Phím tắt ⌘/Ctrl + Enter. |
| Thanh tiến độ | Màn hình dot-matrix của thiết bị phát chính là thanh tiến độ: phần trăm, giai đoạn, bước giải mã `k/N`, thời gian còn lại. Server đếm chính xác bằng forward hook trên OmniVoice (mỗi bước giải mã gọi forward một lần cho cả lô), đẩy về trình duyệt qua Server-Sent Events, có polling dự phòng. Huỷ được giữa chừng. |
| Chỉnh cấu hình | Panel tự sinh từ schema của mô hình: `num_step`, `speed`, `guidance_scale`, `language`, và nâng cao `t_shift`, `position_temperature`, `class_temperature`, `denoise`, `postprocess_output`. Ba mức cài sẵn 8, 16, 32 bước. Thêm nhóm xử lý: chuẩn hoá tiếng Việt, tách câu, khoảng nghỉ, rút ngắn khoảng lặng dài, số câu mỗi lô, seed. |
| Cache và tăng tốc | Xem bên dưới. |

### Bốn lớp cache

1. **Giọng mẫu**: định danh bằng SHA-256 của file. Tải lại cùng file thì không giải mã lại.
2. **Voice prompt**: `model.create_voice_clone_prompt()` (acoustic token + lời chép) được lưu ra đĩa (`.pt`) và giữ trong RAM (LRU 16). Lần sau bỏ qua Vọng và audio tokenizer.
3. **Câu đã sinh**: mỗi câu được lưu theo khoá `(mô hình, giọng, lời mẫu, câu, tham số ảnh hưởng âm thanh)`. Sửa một câu thì chỉ câu đó được sinh lại.
4. **Kết quả hoàn chỉnh**: cùng nội dung và cấu hình thì phát ngay, không chạm GPU.

Ngoài cache: model chỉ nạp một lần khi khởi động (fp16 trên GPU), các câu còn thiếu được sinh **theo lô** trong một lần gọi `generate()` (OmniVoice ghép batch cho cả classifier-free guidance), và `seed` cố định để kết quả lặp lại được.

Nút *Xoá cache* ở cuối trang xoá lớp 2 đến 4 và lịch sử, giữ lại giọng mẫu.

## Giọng mẫu có sẵn

Tab *Có sẵn* có 8 giọng người thật (4 nữ, 4 nam) lấy từ FLEURS tiếng Việt (Google, CC BY 4.0), nằm trong `backend/presets/` và được nạp vào kho giọng khi server khởi động. Mỗi giọng đi kèm lời đọc gốc, nên chọn xong là đọc được ngay, không qua bước chép lời.

Các giọng được chọn bằng số đo, không theo cảm tính: cao độ trung vị (pYIN), độ biến thiên cao độ (độ lệch chuẩn, tính bằng bán cung), tốc độ nói (âm tiết trên giây tiếng nói), tỉ lệ tiếng nói trên ồn nền từ 25 dB trở lên, và Whisper nghe lại khớp lời gốc (WER dưới 6%). Số đo của từng giọng ghi trong `backend/presets/presets.json`.

## Chuẩn hoá văn bản tiếng Việt (đề xuất cải tiến)

`normalize_text=True` của OmniVoice dùng WeTextProcessing, vốn nhắm tới tiếng Trung và tiếng Anh, nên đọc sai số và đơn vị trong câu tiếng Việt (thí nghiệm 5 trong notebook). `backend/textnorm_vi.py` viết lại trước khi đưa vào mô hình:

```
Nhiệt độ hôm nay là 30°C, độ ẩm 80%, giá vé 15k tại TP.HCM.
-> Nhiệt độ hôm nay là ba mươi độ xê, độ ẩm tám mươi phần trăm, giá vé mười lăm nghìn tại thành phố Hồ Chí Minh.
```

Hỗ trợ: số nguyên tới nghìn tỷ (đúng quy tắc *mốt, tư, lăm, linh*), số thập phân, dấu chấm phân cách hàng nghìn, %, °C, đơn vị đo, tiền tệ, ngày tháng, giờ, khoảng `2-3`, và các viết tắt hay gặp. Chạy `python -m backend.textnorm_vi` để xem ví dụ.

## Khoảng nghỉ chủ động

Viết thẳng vào văn bản: `[nghỉ 1.5s]`, `[nghỉ 800ms]`, `[nghỉ 2 giây]`, `[nghỉ 0,5s]` hoặc `[nghỉ]` (1 giây). Nút *Chèn khoảng nghỉ* dưới ô văn bản chèn sẵn `[nghỉ 1s]` tại con trỏ.

- Văn bản được cắt tại dấu nghỉ **trước** khi chuẩn hoá, nên "1.5s" không bị đọc thành chữ, và hai câu hai bên không bao giờ bị gộp.
- Khi ghép, phần lặng mô hình tự sinh ở mép hai câu được bỏ đi rồi chèn đúng số mili giây yêu cầu (đo được 1,503 s cho `[nghỉ 1.5s]`, 3,003 s cho `[nghỉ 3s]`). Khoảng nghỉ này không bị “Rút ngắn khoảng lặng dài” cắt.
- Đổi độ dài khoảng nghỉ không phải sinh lại: các câu lấy từ cache, chỉ ghép lại (0,01 s).
- Dấu nghỉ đứng liền nhau được cộng dồn; dấu nghỉ ở đầu hoặc cuối văn bản tạo khoảng lặng ở đầu hoặc cuối file.

Khoảng nghỉ tự nhiên do dấu câu (OmniVoice, cùng giọng mẫu, 16 bước, chưa rút ngắn): không dấu 140 ms, dấu phẩy 160 ms, dấu chấm 380 ms, xuống dòng 500 ms, `...` 620 ms.

## Khoảng lặng dài giữa câu (lỗi đã sửa)

Khi giọng mẫu thu ở nơi có tiếng ồn nền và có nhiều chỗ ngắt nghỉ, OmniVoice sinh ra khoảng lặng 3 đến 4 giây giữa câu. Có hai nguyên nhân, sửa ở hai chỗ:

1. **Ước lượng độ dài sai.** OmniVoice ước lượng tốc độ nói bằng *số chữ của lời mẫu / thời lượng mẫu*. Khoảng lặng trong giọng mẫu làm người nói trông chậm hơn thực tế, nên audio được dự trù quá dài và mô hình lấp phần thừa bằng khoảng lặng. Studio rút các khoảng lặng của giọng mẫu xuống tối đa 400 ms (`ref_tight.wav`) trước khi mã hoá voice prompt.
2. **Ngưỡng cắt lặng cố định.** Bước hậu xử lý của OmniVoice chỉ coi là lặng khi dưới -50 dBFS, nhưng tiếng ồn nền được mô hình chép lại nằm khoảng -42 dBFS. `audio.shorten_pauses` dùng ngưỡng thích nghi theo từng câu (30% từ mức ồn nền tới mức tiếng nói), coi các đốm ngắn và nhỏ (tiếng thở) là lặng, nhưng giữ lại chữ ngắn đọc nhỏ.

Với giọng mẫu `test-voice.m4a` (17 s, có ồn nền), audio thô giảm từ 12,6 s xuống 9,2 s, kết quả cuối 8,7 s, và Vọng nghe lại đủ chữ. Tắt hoặc chỉnh ở *Xử lý văn bản và hiệu năng*: “Rút ngắn khoảng lặng dài”, “Khoảng lặng dài nhất”.

## Mô hình 2: MMS-TTS

`backend/engines/mms_engine.py` dùng lại cách làm trong CLI `mms_tts/cli.py` của nhóm: cùng checkpoint `facebook/mms-tts-vie`, chuẩn hoá Unicode NFC, kiểm tra văn bản có token đọc được, seed cố định, giới hạn số luồng CPU.

- **Không cần giọng mẫu.** Khi chọn MMS-TTS, giao diện ghi rõ mô hình đọc bằng giọng có sẵn, và nút đọc chạy được dù chưa có giọng mẫu.
- **Tham số:** `speaking_rate` (tốc độ), `noise_scale` (độ biến thiên giọng), `noise_scale_duration` (độ biến thiên nhịp). Chuẩn hoá tiếng Việt, tách câu, `[nghỉ …]`, cache theo câu và rút ngắn khoảng lặng vẫn áp dụng như OmniVoice.
- **Thiết bị:** mặc định CUDA nếu có, ngược lại CPU (VITS nhẹ, CPU đủ nhanh). Đổi bằng `VONG_MMS_DEVICE`; đổi checkpoint bằng `VONG_MMS_MODEL`.
- **Đo trên M3 Pro (CPU):** câu 12,4 s audio sinh trong 1,3 s (RTF 0,10). OmniVoice 16 bước trên MPS: RTF khoảng 1,0, nhưng clone được giọng và Whisper nghe lại đúng hơn (MMS đọc "Nhiệt độ" thành "Diệt độ").

### Thêm mô hình khác

Viết một lớp con của `TTSEngine` (`backend/engines/base.py`): khai báo `PARAMS` (giao diện tự vẽ điều khiển, cache tự khoá theo chúng), viết `load()`, `prepare_voice()`, `steps_per_batch()`, `synthesize()` và gọi `on_step()` để thanh tiến độ chạy; đặt `needs_voice = False` nếu mô hình không dùng giọng mẫu. Đăng ký trong `backend/engines/__init__.py`.

## API

| Method | Đường dẫn | Mô tả |
| --- | --- | --- |
| GET | `/api/status` | trạng thái mô hình, thống kê cache, giới hạn |
| GET | `/api/models` | danh sách mô hình và schema tham số |
| GET, POST | `/api/voices` | liệt kê, tải lên giọng mẫu (multipart `file`) |
| GET, DELETE | `/api/voices/{id}`… | audio WAV của giọng, xoá giọng |
| POST | `/api/normalize` | xem trước chuẩn hoá và tách câu |
| POST | `/api/jobs` | tạo job `{engine, voice_id, text, ref_text, params}` |
| GET | `/api/jobs/{id}/events` | tiến độ qua Server-Sent Events |
| POST | `/api/jobs/{id}/cancel` | huỷ |
| GET | `/api/audio/{key}.wav` | file kết quả |
| GET | `/api/history` | các lần đọc gần đây |
| POST | `/api/cache/clear` | xoá cache |

## Giấy phép và đạo đức

Code OmniVoice theo Apache 2.0, trọng số theo CC-BY-NC: chỉ dùng cho nghiên cứu phi thương mại. Chỉ nhân bản giọng của một người khi người đó đồng ý.
