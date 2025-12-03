# MIND News Recommendation with Pretrained LM + Fastformer

Pipeline huấn luyện/evaluate hệ gợi ý tin tức MIND (MINDsmall) dùng pretrained language model và Fastformer-style user encoder. Code chú thích rõ shape để thuận tiện review.

## Cấu trúc dữ liệu
Mặc định dùng `data/MINDsmall_train` và `data/MINDsmall_dev` đã có sẵn (behaviors.tsv, news.tsv).

## Cài đặt
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# cần sẵn mô hình Hugging Face (vd. bert-base-uncased) trong cache khi không có mạng
```

## Cấu hình
Chỉnh `configs/default.yaml`:
- `data.*`: đường dẫn behaviors/news, `max_history`, `max_seq_len`, `cache_dir`.
- `model.name`: key của model trong registry (hiện có `fastformer`), `pretrained_model_name/tokenizer_name`, `embed_dim`, `dropout`.
- `train.*`: batch size, lr, `neg_k` (số negative/positive), `epochs`, `fp16`, `grad_accum_steps`.
- `eval.batch_size`: batch cho dev/inference.

## Huấn luyện & đánh giá
```bash
python -m src.train --config configs/default.yaml
```
- Loader parse `news.tsv` → tokenize (title + abstract), cache (có kèm category/subcategory id cho các model như NAML), padding row = 0; `behaviors.tsv` → lịch sử + impressions, negative sampling.
- Model registry: `src/models/registry.py` xây model từ `model.name`. Fastformer hiện tại = NewsEncoder (pretrained LM + projection) + FastformerAggregator (global query/key additive attention) → user vector; scorer = dot-product.
- Collator trả tensor kèm comment shape: history `(B,H,L)`, candidates `(B,K,L)`, mask/indices, category/subcategory nếu có.
- Tiêu chí in ra: loss train và AUC trên dev. Checkpoint tốt nhất lưu ở `checkpoints/`, kèm `model_config` + `tokenizer_name`.

## Suy luận
```bash
python -m src.inference \
  --config configs/default.yaml \
  --checkpoint checkpoints/fastformer_epoch1_auc0.0xxx.pt \
  --behaviors data/MINDsmall_dev/behaviors.tsv \
  --news data/MINDsmall_dev/news.tsv \
  --output predictions.jsonl
```
Output mỗi dòng JSON: `{"impression_id": "...", "ranking": [["N123", score], ...]}` đã sắp xếp giảm dần theo điểm.

## Ghi chú triển khai
- Collate đã chèn comment về shape (history: `(B, H, L)`, candidates: `(B, K, L)`), padding news id = 0; cung cấp thêm `history_mask`, `candidate_mask`, category/subcategory cho model cần thêm feature.
- Negative sampling thực hiện online khi đọc behaviors (một positive + `neg_k` negative cho mỗi impression có click).
- Evaluator tính AUC trung bình trên từng impression (bỏ qua trường hợp toàn positive/toàn negative).
- Khuyến nghị chạy trên GPU; nếu dùng CPU, giảm `batch_size`, `max_history`, `max_seq_len`.

## Mở rộng gợi ý
- Để thêm model mới (NAML/NRMS/MINER), tạo file trong `src/models/variants_*.py` hoặc tương tự, implement `RecModelBase`, đăng ký bằng decorator `@register_model("<name>")`, và đọc batch từ collator (news encoder có sẵn category/subcategory).
- Freeze/finetune từng phần LM, thêm scheduler/warmup lớn hơn; có thể precompute news embedding cho serving.
- Bổ sung logging (TensorBoard/W&B) và script hyper-parameter search để tối ưu AUC.
