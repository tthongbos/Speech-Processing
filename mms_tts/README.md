# MMS Text-to-Speech

Demo CLI dùng Hugging Face Transformers `VitsModel` và `AutoTokenizer`, mặc định
`facebook/mms-tts-vie`. Chạy các lệnh dưới đây từ **thư mục gốc repo**.
Không cần Gradio, Streamlit, FFmpeg hay dịch vụ API.

## Cài đặt

Python **3.10+**. Nên tạo môi trường riêng; nếu đã có `python_env` phù hợp thì
chỉ cần kích hoạt môi trường đó.

```bash
python3 -m venv python_env
source python_env/bin/activate
python -m pip install --upgrade pip

# Máy chỉ dùng CPU:
python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r mms_tts/requirements.txt
```

Với NVIDIA GPU, cài Torch có CUDA phù hợp với driver theo
[hướng dẫn PyTorch](https://pytorch.org/get-started/locally/), rồi cài
`mms_tts/requirements.txt` như trên. Kiểm tra:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Một số ngôn ngữ cần romanization; cài thêm `python -m pip install uroman`.
Tiếng Việt mặc định không cần gói này. Tokenizer tự áp dụng romanization khi
checkpoint yêu cầu; xem [tài liệu VITS](https://huggingface.co/docs/transformers/model_doc/vits).

## Chạy demo

```bash
# Mặc định tiếng Việt, tự chọn CUDA nếu khả dụng, ngược lại CPU:
python -m mms_tts --text "Xin chào! Chúc bạn một ngày tốt lành."

# Ép CPU, ghi WAV vào /tmp:
python -m mms_tts --text "Xin chào Việt Nam!" --device cpu --output /tmp/xin_chao.wav

# Ép CUDA; báo lỗi rõ ràng nếu GPU/Torch CUDA không khả dụng:
python -m mms_tts --text "Đây là bản thử nghiệm trên GPU." --device cuda --output outputs/gpu.wav

# Đọc file UTF-8; chấp nhận BOM, gộp khoảng trắng và chuẩn hóa Unicode NFC:
printf 'Xin chào!\nĐây là nội dung từ tệp.\n' > /tmp/mms_input.txt
python -m mms_tts --text-file /tmp/mms_input.txt --device cpu

# Chọn checkpoint theo mã ngôn ngữ MMS, ví dụ tiếng Anh:
python -m mms_tts --lang eng --text "Hello! This is an English speech demo."

# Hoặc chỉ định model ID / thư mục checkpoint VITS cục bộ:
python -m mms_tts --model-id facebook/mms-tts-vie --text "Xin chào!" --seed 42

python -m mms_tts --help
```

- Chọn đúng một trong `--text` và `--text-file`. Input rỗng bị từ chối.
- `--model` là alias của `--model-id`; không kết hợp với `--lang`.
  `--lang` chọn `facebook/mms-tts-<code>` (ví dụ `vie`, `eng`), không dịch văn bản.
  Checkpoint tùy chọn phải tương thích VITS, không phải model TTS bất kỳ.
- `--output` mặc định là `outputs/mms_tts.wav`, tính từ thư mục đang chạy.
  Tự tạo thư mục cha; **ghi đè WAV đã có**, nhưng không cho ghi đè file input.
- `--cpu-threads` mặc định 4 để tránh quá nhiều thread trên máy nhiều lõi;
  có thể điều chỉnh theo phần cứng. Dùng float32, `eval()` và `inference_mode()`.
- `--seed` mặc định 42 vì VITS có lấy mẫu ngẫu nhiên. Cùng seed giúp tái lập
  trong cùng môi trường, không bảo đảm WAV giống từng bit giữa CPU/GPU/phiên bản.
- Demo xử lý một đoạn văn bản mỗi lần. Nên dùng câu/đoạn ngắn; chưa có chia
  văn bản dài tự động hoặc streaming. Văn bản chỉ có dấu câu/ký tự bị tokenizer
  loại bỏ sẽ báo lỗi thay vì sinh âm thanh trống. Chất lượng phụ thuộc checkpoint.

## Cache và offline

Lần đầu cần mạng để tải tokenizer/weights từ Hugging Face. Cache mặc định ở
`~/.cache/huggingface`, có thể đổi bằng biến `HF_HOME` (nên đặt ngoài repo).
Sau khi tải thành công, chạy lại hoàn toàn từ cache:

```bash
python -m mms_tts --text "Xin chào!" --local-files-only --device cpu
```

Repo ignore `outputs/`, WAV/MP3/FLAC, cache, môi trường Python và checkpoint.
Không commit weights hoặc âm thanh sinh ra.

## Log và kiểm thử

CLI log model/device, thời gian tải/chuẩn bị, đường dẫn WAV, sample rate,
thời lượng audio, inference time và RTF. Inference time chỉ tính forward của
model; CUDA được synchronize trước/sau phép đo. Không tính tải model, tokenize,
chuyển WAV về CPU hay ghi file. **RTF = inference time / audio duration**;
RTF < 1 nghĩa là phần forward nhanh hơn thời lượng âm thanh. Đây là một lần
chạy, không phải benchmark warmup hoặc độ trễ toàn bộ CLI.

```bash
# Test offline, không tải checkpoint; dùng unittest có sẵn trong Python:
python -m unittest discover -s tests -v

# Smoke thật (cần mạng hoặc cache):
python -m mms_tts --text "Xin chào! Đây là bản thử nghiệm tổng hợp tiếng nói bằng tiếng Việt." --device cpu --output /tmp/mms_smoke.wav
```

Đã kiểm tra trên Python 3.10.19, Torch 2.7.1+cpu, Transformers 4.53.0,
NumPy 1.26.4, SciPy 1.15.3: model thật tạo WAV mono PCM16, 16.000 Hz,
dài 5,744 giây; forward 0,813 giây, RTF 0,142 với 4 CPU thread.
Đây là kết quả quan sát trên máy phát triển; tốc độ máy khác sẽ khác.
Không có CUDA trong môi trường này nên chưa kiểm thử inference GPU thật.
Unit test kiểm tra chọn thiết bị với cả hai trạng thái CUDA khả dụng/không khả dụng.

Tài liệu nguồn: [model card tiếng Việt](https://huggingface.co/facebook/mms-tts-vie).
Weights được phát hành theo **CC-BY-NC-4.0**; xem điều khoản trong model card.
