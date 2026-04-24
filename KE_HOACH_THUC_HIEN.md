# KẾ HOẠCH THỰC HIỆN BÀI TẬP LỚN - MÔN HỌC SÂU

> **Phần 1**: #10 - Phân tích quan điểm về nhà hàng / khách sạn
> **Phần 2**: #25 - Tìm hiểu và trình bày mô hình BERT
> **Combo E (NLP)**: Cả 2 phần đều thuộc lĩnh vực Xử lý ngôn ngữ tự nhiên (NLP), bổ trợ lẫn nhau — dùng BERT làm mô hình cho Phần 1, trình bày lý thuyết BERT ở Phần 2

---

## PHẦN 1: DEMO - PHÂN TÍCH QUAN ĐIỂM NHÀ HÀNG / KHÁCH SẠN

### 1.1 Mô tả bài toán

**Phân tích quan điểm (Sentiment Analysis)** là bài toán phân loại văn bản theo cảm xúc. Đây là một trong những bài toán kinh điển và quan trọng nhất trong lĩnh vực Xử lý ngôn ngữ tự nhiên (Natural Language Processing - NLP).

- **Đầu vào (Input)**: Một đoạn văn bản đánh giá (review) của khách hàng về nhà hàng hoặc khách sạn. Ví dụ: _"Đồ ăn rất ngon, nhân viên phục vụ nhiệt tình, không gian thoáng mát"_
- **Đầu ra (Output)**: Nhãn cảm xúc tương ứng:
  - **Tích cực (Positive)**: Khách hàng hài lòng, khen ngợi
  - **Tiêu cực (Negative)**: Khách hàng không hài lòng, phàn nàn
  - **Trung tính (Neutral)**: Nhận xét khách quan, không thiên về bên nào
- **Ứng dụng thực tế**:
  - Giúp chủ nhà hàng/khách sạn nắm bắt phản hồi khách hàng tự động
  - Phát hiện sớm các vấn đề cần cải thiện (vệ sinh, thái độ nhân viên, chất lượng món ăn)
  - Hỗ trợ ra quyết định kinh doanh dựa trên dữ liệu
  - Theo dõi danh tiếng thương hiệu trên mạng xã hội

### 1.2 Tại sao chọn bài toán này?

1. **Liên kết chặt chẽ với Phần 2**: Dùng chính mô hình BERT để fine-tune cho bài toán phân tích quan điểm → khi vấn đáp có thể giải thích cả lý thuyết lẫn thực hành
2. **Dataset phong phú**: Có nhiều bộ dữ liệu sẵn có trên Kaggle, không cần tự thu thập
3. **Giá trị thực tế cao**: Sentiment Analysis được ứng dụng rộng rãi trong các doanh nghiệp, startup, thương mại điện tử
4. **Kết quả trực quan**: Người dùng nhập review → hệ thống trả về kết quả ngay lập tức, dễ demo

### 1.3 Dataset gợi ý

| Dataset | Ngôn ngữ | Số lượng | Nguồn | Ghi chú |
|---|---|---|---|---|
| Yelp Reviews | Tiếng Anh | ~6.9 triệu reviews | [Kaggle Yelp Dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset) | Phổ biến nhất, có rating 1-5 sao, dung lượng lớn |
| TripAdvisor Hotel Reviews | Tiếng Anh | ~20,000 reviews | [Kaggle TripAdvisor](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews) | Nhỏ gọn, phù hợp train trên máy cá nhân |
| Restaurant Reviews (UCI) | Tiếng Anh | ~1,000 reviews | [Kaggle Restaurant Reviews](https://www.kaggle.com/datasets/d4rklucif3r/restaurant-reviews) | Đơn giản nhất, tốt để bắt đầu thử nghiệm |
| Foody Reviews | Tiếng Việt | Tự crawl | foody.vn | Cần viết script crawl, phù hợp nếu muốn làm tiếng Việt |
| VLSP Sentiment | Tiếng Việt | ~10,000 | VLSP shared task | Dataset chuẩn tiếng Việt cho nghiên cứu |

**Khuyến nghị**: Bắt đầu với **TripAdvisor Hotel Reviews** (~20k) để prototype nhanh, sau đó mở rộng sang Yelp nếu muốn kết quả tốt hơn.

### 1.4 GitHub Repo tham khảo (đã xác nhận public - HTTP 200)

| Repo | Stars | Mô tả chi tiết | Link |
|---|---|---|---|
| Restaurant-Reviews-Sentiment-Analysis | 3 | Pipeline hoàn chỉnh: EDA → tiền xử lý → so sánh nhiều mô hình (Naive Bayes, SVM, LSTM) | [GitHub](https://github.com/afreenasif/Restaurant-Reviews-Sentiment-Analysis) |
| HotelReviewNLP | 4 | Phân tích NLP cho hotel review, có EDA chi tiết với biểu đồ đẹp | [GitHub](https://github.com/DrigoDomingos/HotelReviewNLP) |
| Hotel-Review-Sentiment-analysis | 4 | Hotel sentiment với pipeline đầy đủ từ data đến model | [GitHub](https://github.com/fenilgodhani/Hotel-Review-Sentiment-analysis) |
| Zomato-LSTM-Sentiment-Analysis | 1 | BiLSTM cho sentiment + giao diện Streamlit web UI | [GitHub](https://github.com/Yuvaraja-techdev/Zomato--LSTM--Sentiment-Analysis) |
| Restaurant-Review-Sentiment-Analysis | 1 | So sánh LSTM và Transformer trên Yelp reviews | [GitHub](https://github.com/CzJLee/Restaurant-Review-Sentiment-Analysis) |

### 1.5 Kiến trúc mô hình chi tiết

```
                    KIẾN TRÚC TỔNG QUAN
                    ====================

Input: "Đồ ăn rất ngon, phục vụ tuyệt vời"
                        |
                        v
        +-------------------------------+
        |   BƯỚC 1: TIỀN XỬ LÝ VĂN BẢN  |
        +-------------------------------+
        | - Chuyển chữ thường (lowercase) |
        | - Loại bỏ HTML tags, URL        |
        | - Loại bỏ ký tự đặc biệt       |
        | - Loại bỏ stopwords (tùy chọn)  |
        +-------------------------------+
                        |
                        v
        +-------------------------------+
        |  BƯỚC 2: BERT TOKENIZER        |
        +-------------------------------+
        | - Thêm [CLS] ở đầu câu        |
        | - Thêm [SEP] ở cuối câu        |
        | - WordPiece tokenization        |
        | - Chuyển tokens → input_ids     |
        | - Tạo attention_mask            |
        | - Padding đến max_length=256    |
        +-------------------------------+
                        |
                        v
        +-------------------------------+
        |  BƯỚC 3: BERT ENCODER           |
        +-------------------------------+
        | Pre-trained model:              |
        | - bert-base-uncased (tiếng Anh) |
        | - vinai/phobert-base (tiếng Việt)|
        |                                 |
        | 12 Transformer Encoder layers   |
        | Mỗi layer gồm:                 |
        |   - Multi-Head Self-Attention   |
        |   - Feed-Forward Network        |
        |   - Layer Normalization         |
        |   - Residual Connection         |
        |                                 |
        | Output: vector 768 chiều cho    |
        | mỗi token trong câu            |
        +-------------------------------+
                        |
                        v
        +-------------------------------+
        |  BƯỚC 4: CLASSIFICATION HEAD    |
        +-------------------------------+
        | - Lấy output của token [CLS]   |
        |   (vector 768 chiều)           |
        | - Dropout (p=0.3) chống        |
        |   overfitting                  |
        | - Dense layer: 768 → 256       |
        | - ReLU activation              |
        | - Dropout (p=0.3)              |
        | - Dense layer: 256 → 3 classes |
        | - Softmax activation           |
        +-------------------------------+
                        |
                        v
        +-------------------------------+
        |  BƯỚC 5: OUTPUT                 |
        +-------------------------------+
        | Positive: 0.92                 |
        | Negative: 0.05                 |
        | Neutral:  0.03                 |
        | => Kết luận: POSITIVE          |
        +-------------------------------+
```

**Giải thích tại sao dùng token [CLS]**: Trong BERT, token [CLS] được thiết kế để tổng hợp thông tin ngữ nghĩa của toàn bộ câu. Sau khi đi qua 12 lớp Transformer, output của [CLS] chứa biểu diễn tổng hợp (aggregate representation) của cả câu, phù hợp cho bài toán phân loại văn bản.

### 1.6 Công nghệ sử dụng

| Thành phần | Công nghệ | Phiên bản | Lý do chọn |
|---|---|---|---|
| Ngôn ngữ | Python | 3.10+ | Hệ sinh thái ML/DL phong phú nhất |
| Framework DL | PyTorch | 2.0+ | Linh hoạt, debug dễ, cộng đồng lớn |
| Thư viện NLP | HuggingFace Transformers | 4.30+ | Cung cấp BERT pre-trained, API đơn giản |
| Pre-trained model | bert-base-uncased | - | 110 triệu tham số, tiếng Anh, uncased |
| Hoặc (tiếng Việt) | vinai/phobert-base | - | Pre-trained trên 20GB tiếng Việt |
| Xử lý dữ liệu | pandas, numpy | - | Chuẩn công nghiệp cho data processing |
| Chia dữ liệu | scikit-learn | - | train_test_split, metrics |
| Trực quan hóa | matplotlib, seaborn | - | Biểu đồ phân tích, confusion matrix |
| Word Cloud | wordcloud | - | Hiển thị từ khóa phổ biến |
| Demo UI | Streamlit hoặc Gradio | - | Tạo web app nhanh, không cần frontend |
| Notebook | Jupyter / Google Colab | - | Colab có GPU T4 miễn phí |

### 1.7 Các bước thực hiện chi tiết

#### Bước 1: Thu thập và chuẩn bị dataset

- [ ] Tải dataset từ Kaggle (TripAdvisor Hotel Reviews hoặc Yelp)
- [ ] Khám phá cấu trúc dữ liệu: các cột, kiểu dữ liệu, giá trị null
- [ ] Chuyển đổi rating thành nhãn sentiment:
  - Rating 1-2 sao → **Negative**
  - Rating 3 sao → **Neutral**
  - Rating 4-5 sao → **Positive**
- [ ] Chia dữ liệu: Train 80% / Validation 10% / Test 10%
- [ ] Kiểm tra cân bằng nhãn (class balance), nếu mất cân bằng thì áp dụng oversampling hoặc class weights

#### Bước 2: Phân tích khám phá dữ liệu (EDA - Exploratory Data Analysis)

- [ ] Thống kê số lượng review theo từng nhãn sentiment (biểu đồ cột)
- [ ] Phân tích độ dài trung bình của review (histogram)
- [ ] Tạo WordCloud cho từng loại sentiment (Positive, Negative, Neutral)
- [ ] Phân tích top 20 từ xuất hiện nhiều nhất trong mỗi loại
- [ ] Biểu đồ phân phối rating (1-5 sao)
- [ ] Kiểm tra và xử lý dữ liệu trùng lặp, dữ liệu rỗng

#### Bước 3: Tiền xử lý dữ liệu

- [ ] Làm sạch văn bản:
  - Chuyển về chữ thường (lowercase)
  - Loại bỏ HTML tags: `<br>`, `<p>`, v.v.
  - Loại bỏ URL, email
  - Loại bỏ ký tự đặc biệt, số (tùy trường hợp)
  - Loại bỏ khoảng trắng thừa
- [ ] Tokenization bằng BERT Tokenizer:
  - `AutoTokenizer.from_pretrained('bert-base-uncased')`
  - Thêm token đặc biệt: [CLS] ở đầu, [SEP] ở cuối
  - WordPiece: tách từ thành subword (ví dụ: "unhappy" → "un", "##happy")
  - Chuyển tokens thành input_ids (số nguyên)
  - Tạo attention_mask (1 cho token thật, 0 cho padding)
- [ ] Tạo PyTorch Dataset và DataLoader:
  - Batch size: 32 (hoặc 16 nếu thiếu RAM GPU)
  - Shuffle cho train set, không shuffle cho val/test
  - num_workers: 2-4 để tăng tốc đọc dữ liệu

#### Bước 4: Xây dựng mô hình

- [ ] Tải BERT pre-trained: `BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)`
- [ ] Hoặc tự xây dựng classification head:
  - Lấy output [CLS] từ BERT (768 chiều)
  - Thêm Dropout (p=0.3) để chống overfitting
  - Thêm Linear layer: 768 → 256 → 3
  - Softmax activation cho output
- [ ] Định nghĩa hàm loss: `CrossEntropyLoss` (có hỗ trợ class weights nếu dữ liệu mất cân bằng)
- [ ] Optimizer: `AdamW` với learning rate = 2e-5 (khuyến nghị cho fine-tuning BERT)
- [ ] Learning rate scheduler: Linear warmup rồi giảm dần (warmup_steps = 10% tổng steps)

#### Bước 5: Huấn luyện mô hình

- [ ] Huấn luyện 3-5 epochs (BERT thường hội tụ nhanh, không cần nhiều epoch)
- [ ] Mỗi epoch:
  - Forward pass: Đưa batch qua mô hình, tính output
  - Tính loss: So sánh output với nhãn thật
  - Backward pass: Tính gradient
  - Update weights: Optimizer step
  - Ghi nhận train loss, train accuracy
- [ ] Sau mỗi epoch, đánh giá trên validation set:
  - Tính val_loss, val_accuracy
  - Nếu val_loss giảm → lưu model checkpoint (best model)
  - Nếu val_loss tăng liên tiếp 2 epoch → Early stopping
- [ ] Vẽ biểu đồ learning curve: train_loss vs val_loss qua các epoch
- [ ] Theo dõi GPU memory usage, training time mỗi epoch

#### Bước 6: Đánh giá mô hình

- [ ] Đánh giá trên test set (dữ liệu chưa từng thấy):
  - **Accuracy**: Tỷ lệ dự đoán đúng tổng thể
  - **Precision**: Trong các mẫu dự đoán là Positive, bao nhiêu % thực sự Positive
  - **Recall**: Trong các mẫu thực sự Positive, mô hình tìm được bao nhiêu %
  - **F1-score**: Trung bình điều hòa của Precision và Recall
  - **Confusion Matrix**: Ma trận nhầm lẫn, trực quan hóa bằng heatmap
- [ ] So sánh với các mô hình baseline:
  - **Naive Bayes + TF-IDF**: Mô hình cơ bản nhất
  - **SVM + TF-IDF**: Mô hình truyền thống mạnh
  - **LSTM**: Mô hình deep learning không dùng pre-trained
  - **BERT fine-tuned**: Mô hình của chúng ta
- [ ] Phân tích lỗi (Error Analysis):
  - Xem các mẫu bị dự đoán sai
  - Tìm pattern: review ngắn, mỉa mai (sarcasm), review trung tính khó phân loại
  - Đề xuất hướng cải thiện

#### Bước 7: Xây dựng demo giao diện (UI)

- [ ] Tạo giao diện Streamlit hoặc Gradio:
  - Ô nhập liệu: Người dùng nhập review
  - Nút "Phân tích": Gửi review qua mô hình
  - Hiển thị kết quả: Nhãn sentiment + confidence score (%)
  - Biểu đồ thanh: Hiển thị xác suất cho 3 lớp
- [ ] Tải model checkpoint đã lưu (best model)
- [ ] Xử lý edge cases: review rỗng, review quá dài, ký tự đặc biệt
- [ ] Thêm một số review mẫu để demo nhanh
- [ ] Chạy thử và screenshot cho báo cáo

#### Bước 8: Viết báo cáo và làm slide

- [ ] Phần demo trong quyển báo cáo (~15 trang):
  - Giới thiệu bài toán và ứng dụng
  - Mô tả dataset
  - Kiến trúc mô hình (có hình vẽ)
  - Kết quả thực nghiệm (bảng so sánh, confusion matrix)
  - Giao diện demo (screenshot)
  - Kết luận và hướng phát triển
- [ ] Phần demo trong slide (~10-15 slides):
  - Demo trực tiếp trên máy (ưu tiên)
  - Hoặc quay video demo dự phòng

---

## PHẦN 2: LÝ THUYẾT - MÔ HÌNH BERT

### 2.1 Tổng quan về BERT

**BERT** viết tắt của **Bidirectional Encoder Representations from Transformers** (Biểu diễn mã hóa hai chiều từ Transformers).

- **Tác giả**: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (Google AI Language)
- **Năm công bố**: Tháng 10/2018
- **Paper gốc**: _"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"_ (arXiv:1810.04805)
- **Ý nghĩa lịch sử**: BERT đã tạo ra bước đột phá trong NLP, đạt kết quả state-of-the-art (SOTA) trên 11 bài toán NLP cùng lúc. Nó thay đổi hoàn toàn cách tiếp cận NLP: từ huấn luyện mô hình từ đầu (train from scratch) sang phương pháp pre-train rồi fine-tune.

### 2.2 Bối cảnh ra đời - Tại sao cần BERT?

#### 2.2.1 Hạn chế của các mô hình trước BERT

| Mô hình | Năm | Cách hoạt động | Hạn chế |
|---|---|---|---|
| **Word2Vec** | 2013 | Mỗi từ có 1 vector cố định | Không hiểu ngữ cảnh. Ví dụ: "bank" (ngân hàng) và "bank" (bờ sông) có cùng vector |
| **GloVe** | 2014 | Vector dựa trên ma trận đồng xuất hiện | Cùng hạn chế như Word2Vec - vector tĩnh |
| **ELMo** | 2018 | Dùng Bi-LSTM để tạo vector phụ thuộc ngữ cảnh | Hai chiều nhưng chỉ ghép nối (concatenate) trái-phải, không tương tác sâu |
| **GPT-1** | 2018 | Dùng Transformer Decoder, đọc từ trái sang phải | Chỉ một chiều (unidirectional) - khi xử lý một từ, chỉ thấy các từ bên trái |

#### 2.2.2 Ý tưởng đột phá của BERT

**Vấn đề cốt lõi**: Để hiểu một từ trong câu, ta cần nhìn cả ngữ cảnh bên trái lẫn bên phải cùng lúc.

**Ví dụ minh họa**:
- _"Tôi đến **ngân hàng** để gửi tiền"_ → "ngân hàng" = tổ chức tài chính
- _"Tôi ngồi bên **bờ sông**"_ → cần ngữ cảnh hai bên để hiểu nghĩa

**Giải pháp của BERT**: Sử dụng Transformer Encoder với cơ chế Self-Attention, cho phép mỗi từ "nhìn thấy" tất cả các từ khác trong câu (cả trái lẫn phải) đồng thời → **Bidirectional** (hai chiều thực sự).

### 2.3 Kiến trúc Transformer - Nền tảng của BERT

BERT được xây dựng trên kiến trúc **Transformer**, được giới thiệu trong paper _"Attention Is All You Need"_ (Vaswani et al., 2017). BERT chỉ sử dụng phần **Encoder** của Transformer.

#### 2.3.1 Self-Attention Mechanism (Cơ chế Tự chú ý)

Self-Attention cho phép mô hình xác định mức độ liên quan giữa mỗi cặp từ trong câu.

**Công thức**:
```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

Trong đó:
- **Q (Query)**: "Tôi đang tìm kiếm thông tin gì?" - vector truy vấn của từ hiện tại
- **K (Key)**: "Tôi chứa thông tin gì?" - vector khóa của mỗi từ trong câu
- **V (Value)**: "Thông tin thực tế của tôi là gì?" - vector giá trị cần trích xuất
- **d_k**: Số chiều của vector Key (dùng để chuẩn hóa, tránh giá trị quá lớn)
- **softmax**: Chuyển đổi thành xác suất (tổng = 1), thể hiện mức độ "chú ý" vào mỗi từ

**Ví dụ trực quan**: Với câu _"Con mèo ngồi trên thảm vì nó mệt"_
- Khi xử lý từ "nó", Self-Attention sẽ gán trọng số cao cho "mèo" → hiểu "nó" chỉ "con mèo"

#### 2.3.2 Multi-Head Attention (Chú ý đa đầu)

Thay vì chỉ có 1 bộ Attention, BERT dùng **nhiều đầu (heads)** song song:
- BERT Base: **12 heads** → mỗi head học một khía cạnh ngữ nghĩa khác nhau
- Head 1 có thể học quan hệ ngữ pháp (chủ-vị)
- Head 2 có thể học quan hệ đồng tham chiếu (nó → mèo)
- Head 3 có thể học quan hệ vị trí (từ gần nhau)
- Cuối cùng, ghép nối (concatenate) kết quả tất cả heads và nhân với ma trận trọng số W^O

**Công thức**:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O
    trong đó head_i = Attention(Q * W_i^Q, K * W_i^K, V * W_i^V)
```

#### 2.3.3 Position Encoding (Mã hóa vị trí)

Transformer không có khái niệm "thứ tự" như RNN. Để mô hình biết vị trí của từng từ, BERT sử dụng **Position Embeddings** - mỗi vị trí (0, 1, 2, ..., 511) có một vector embedding riêng được học trong quá trình pre-training.

#### 2.3.4 Feed-Forward Network (Mạng truyền thẳng)

Sau Self-Attention, mỗi vị trí được đưa qua một mạng truyền thẳng 2 lớp:
```
FFN(x) = max(0, x * W_1 + b_1) * W_2 + b_2
```
- Lớp 1: Mở rộng chiều từ 768 → 3072 (x4), dùng ReLU/GELU activation
- Lớp 2: Thu nhỏ chiều từ 3072 → 768

#### 2.3.5 Các thành phần bổ sung

- **Layer Normalization**: Chuẩn hóa output sau mỗi sub-layer, giúp huấn luyện ổn định
- **Residual Connection**: Cộng input vào output (x + Sublayer(x)), giúp gradient truyền ngược tốt hơn, tránh vanishing gradient

### 2.4 Kiến trúc BERT chi tiết

#### 2.4.1 Hai phiên bản BERT

| Thông số | BERT Base | BERT Large |
|---|---|---|
| Số lớp Transformer (L) | 12 | 24 |
| Kích thước hidden (H) | 768 | 1024 |
| Số Attention heads (A) | 12 | 16 |
| Tổng tham số | **110 triệu** | **340 triệu** |
| Kích thước mỗi head | 768/12 = 64 | 1024/16 = 64 |
| FFN inner dimension | 3072 | 4096 |

#### 2.4.2 Input Representation (Biểu diễn đầu vào)

Mỗi token đầu vào của BERT là **tổng của 3 loại embedding**:

```
Input = Token Embedding + Segment Embedding + Position Embedding
```

1. **Token Embedding**: Vector biểu diễn cho mỗi từ/subword trong vocabulary (~30,000 tokens cho tiếng Anh)

2. **Segment Embedding**: Phân biệt câu A và câu B trong các bài toán cặp câu
   - Câu A: tất cả token có Segment = 0
   - Câu B: tất cả token có Segment = 1

3. **Position Embedding**: Biểu diễn vị trí của token trong câu (0 đến 511)

**Các token đặc biệt**:
- **[CLS]** (Classification): Đặt ở đầu mỗi câu. Output của [CLS] sau 12 lớp Transformer được dùng cho bài toán phân loại
- **[SEP]** (Separator): Đặt giữa hai câu và ở cuối, dùng để phân tách câu A và câu B
- **[MASK]**: Dùng trong pre-training để che (mask) token

**Ví dụ**:
```
Input:  [CLS] Đồ ăn rất ngon [SEP] Phục vụ tuyệt vời [SEP]
Token:   101   ...  ...  ...   102   ...   ...   ...    102
Segment:  0     0    0    0     0     1     1     1      1
Position: 0     1    2    3     4     5     6     7      8
```

### 2.5 Pre-training BERT (Huấn luyện trước)

BERT được pre-train trên lượng dữ liệu khổng lồ không cần gán nhãn (unsupervised/self-supervised) với **2 nhiệm vụ đồng thời**:

#### 2.5.1 Masked Language Model (MLM) - Mô hình ngôn ngữ che từ

**Ý tưởng**: Giống trò chơi "điền vào chỗ trống"

**Cách hoạt động**:
1. Chọn ngẫu nhiên **15% tokens** trong mỗi câu để xử lý
2. Trong 15% đó:
   - **80%** được thay bằng [MASK]: _"Đồ ăn rất [MASK]"_ → mô hình dự đoán "ngon"
   - **10%** được thay bằng từ ngẫu nhiên: _"Đồ ăn rất cá"_ → mô hình vẫn phải dự đoán "ngon"
   - **10%** giữ nguyên: _"Đồ ăn rất ngon"_ → mô hình xác nhận "ngon" là đúng
3. Mô hình phải dự đoán từ gốc dựa trên ngữ cảnh hai bên

**Tại sao 80/10/10 mà không 100% [MASK]?**
- 100% [MASK] → khi fine-tune, mô hình chưa bao giờ thấy từ thật, gây mismatch
- 10% từ ngẫu nhiên → buộc mô hình phải "kiểm tra" mọi từ, không chỉ [MASK]
- 10% giữ nguyên → giúp mô hình học biểu diễn của từ thật

#### 2.5.2 Next Sentence Prediction (NSP) - Dự đoán câu tiếp theo

**Ý tưởng**: Mô hình cần hiểu mối quan hệ giữa hai câu

**Cách hoạt động**:
1. Cho cặp câu (A, B), mô hình dự đoán: "B có phải là câu tiếp theo của A không?"
2. **50% trường hợp**: B đúng là câu tiếp theo (nhãn: IsNext)
   - A: _"Nhà hàng này rất đông khách"_
   - B: _"Phải đợi 30 phút mới có bàn"_ → **IsNext**
3. **50% trường hợp**: B là câu ngẫu nhiên (nhãn: NotNext)
   - A: _"Nhà hàng này rất đông khách"_
   - B: _"Hôm nay trời mưa to"_ → **NotNext**

**Tầm quan trọng**: NSP giúp BERT hiểu mối quan hệ logic giữa các câu, hữu ích cho các bài toán như Question Answering, Natural Language Inference.

#### 2.5.3 Dữ liệu pre-training

- **BooksCorpus**: 800 triệu từ (11,038 cuốn sách)
- **English Wikipedia**: 2,500 triệu từ (chỉ lấy phần text)
- Tổng cộng: **~3.3 tỷ từ** (3.3 billion words)
- Thời gian pre-training: **4 ngày trên 64 TPU chips** (Google)

### 2.6 Fine-tuning BERT (Tinh chỉnh)

Sau khi pre-train, BERT được fine-tune cho các bài toán cụ thể bằng cách **thêm một lớp output đơn giản** phía trên.

#### 2.6.1 Phân loại văn bản (Text Classification) — áp dụng cho Phần 1

```
Input:  [CLS] Review text [SEP]
         |
    BERT Encoder (12 layers)
         |
    Output [CLS] → Dense → Softmax → Positive/Negative/Neutral
```
- Dùng output của token [CLS] làm biểu diễn cho toàn bộ câu
- Thêm lớp Dense + Softmax để phân loại

#### 2.6.2 Nhận dạng thực thể (Named Entity Recognition - NER)

```
Input:  [CLS] Hà Nội là thủ đô [SEP]
         |
    BERT Encoder (12 layers)
         |
    Output mỗi token → Dense → Entity tag
    "Hà"→B-LOC, "Nội"→I-LOC, "là"→O, "thủ"→O, "đô"→O
```
- Dùng output của **từng token** để gán nhãn thực thể

#### 2.6.3 Hỏi đáp (Question Answering)

```
Input:  [CLS] Câu hỏi [SEP] Đoạn văn chứa câu trả lời [SEP]
         |
    BERT Encoder (12 layers)
         |
    Output mỗi token → Dự đoán vị trí start và end của câu trả lời
```
- Mô hình tìm vị trí bắt đầu và kết thúc của câu trả lời trong đoạn văn

#### 2.6.4 Các lưu ý khi fine-tuning

| Hyperparameter | Giá trị khuyến nghị | Ghi chú |
|---|---|---|
| Learning rate | 2e-5, 3e-5, 5e-5 | Nhỏ hơn nhiều so với train from scratch |
| Batch size | 16, 32 | Tùy GPU memory |
| Epochs | 2-4 | BERT hội tụ rất nhanh |
| Max sequence length | 128, 256, 512 | 512 là tối đa BERT hỗ trợ |
| Warmup steps | 10% tổng training steps | Tăng dần learning rate từ 0 |
| Weight decay | 0.01 | Regularization |

### 2.7 Kết quả và ảnh hưởng của BERT

#### 2.7.1 Kết quả trên các benchmark (2018)

| Benchmark | Bài toán | Kết quả BERT | So với SOTA trước đó |
|---|---|---|---|
| GLUE | Tổng hợp 8 bài NLU | 80.5 → **BERT: 82.1** | +1.6 điểm |
| SQuAD 1.1 | Hỏi đáp | F1: 91.2 → **BERT: 93.2** | +2.0 điểm |
| SQuAD 2.0 | Hỏi đáp (có câu không trả lời được) | F1: 66.3 → **BERT: 83.1** | +16.8 điểm |
| MNLI | Suy luận ngôn ngữ tự nhiên | 86.7 → **BERT: 87.6** | +0.9 điểm |
| SST-2 | Phân tích cảm xúc | 94.9 → **BERT: 95.9** | +1.0 điểm |

#### 2.7.2 Các biến thể phát triển từ BERT

| Mô hình | Năm | Cải tiến so với BERT | Tác giả |
|---|---|---|---|
| **RoBERTa** | 2019 | Bỏ NSP, train lâu hơn, dữ liệu nhiều hơn | Facebook AI |
| **ALBERT** | 2019 | Giảm tham số bằng parameter sharing và factorized embedding | Google |
| **DistilBERT** | 2019 | Nhỏ hơn 40%, nhanh hơn 60%, giữ 97% hiệu suất | Hugging Face |
| **XLNet** | 2019 | Permutation language model, không dùng [MASK] | Google/CMU |
| **ELECTRA** | 2020 | Thay MLM bằng replaced token detection, hiệu quả hơn | Google |
| **PhoBERT** | 2020 | BERT cho tiếng Việt, pre-train trên 20GB text tiếng Việt | VinAI |
| **DeBERTa** | 2021 | Disentangled attention, enhanced mask decoder | Microsoft |

#### 2.7.3 Ảnh hưởng đến ngành AI

- BERT mở ra kỷ nguyên **"pre-train then fine-tune"** trong NLP
- Là nền tảng cho các Large Language Models (LLMs) hiện đại: GPT-3, GPT-4, Claude, LLaMA
- Thay đổi cách làm NLP: từ feature engineering thủ công sang transfer learning
- Ảnh hưởng đến cả Computer Vision (ViT - Vision Transformer áp dụng cùng kiến trúc)

### 2.8 GitHub Repo tham khảo (đã xác nhận public - HTTP 200)

| Repo | Stars | Mô tả chi tiết | Link |
|---|---|---|---|
| transformers-tutorials | 864 | Hướng dẫn fine-tune Transformers cho nhiều bài toán NLP | [GitHub](https://github.com/abhimishra91/transformers-tutorials) |
| LLM101 | 24 | Tutorial BERT, LLM, multimodal models, fine-tuning | [GitHub](https://github.com/WangRongsheng/LLM101) |
| Bert_fine_tuning_Sentence_classification | 9 | Hướng dẫn chi tiết fine-tuning BERT cho phân loại câu | [GitHub](https://github.com/Prajwalbhandary17/Bert_fine_tuning_Sentence_classification) |

### 2.9 Tài liệu học lý thuyết

| Tài liệu | Loại | Mô tả | Ưu tiên |
|---|---|---|---|
| The Illustrated BERT (Jay Alammar) | Blog | Giải thích trực quan nhất về BERT với hình vẽ minh họa | **Đọc đầu tiên** |
| The Illustrated Transformer (Jay Alammar) | Blog | Giải thích Transformer, nền tảng của BERT | **Đọc thứ hai** |
| BERT Paper gốc (arXiv:1810.04805) | Paper | Paper gốc của tác giả, chi tiết nhất | Đọc thứ ba |
| Attention Is All You Need (arXiv:1706.03762) | Paper | Paper Transformer gốc | Tham khảo |
| HuggingFace BERT docs | Docs | Tài liệu sử dụng BERT trong code | Khi code |
| CS224N Stanford - Lecture 14 | Video | Bài giảng BERT từ Stanford | Bổ sung |

### 2.10 Các bước thực hiện Phần 2

- [ ] **B1**: Đọc và hiểu tổng quan BERT
  - Đọc blog "The Illustrated BERT" của Jay Alammar
  - Đọc blog "The Illustrated Transformer"
  - Ghi chú các khái niệm chính
- [ ] **B2**: Đọc paper gốc BERT
  - Tập trung vào: kiến trúc, pre-training tasks (MLM, NSP), fine-tuning
  - Hiểu rõ bảng kết quả thực nghiệm
  - Ghi chú các điểm cần trình bày
- [ ] **B3**: Hiểu kiến trúc Transformer chi tiết
  - Self-Attention mechanism (Query, Key, Value)
  - Multi-Head Attention
  - Position Encoding
  - Feed-Forward Network
  - Layer Normalization + Residual Connection
  - Vẽ sơ đồ kiến trúc
- [ ] **B4**: Hiểu kiến trúc BERT chi tiết
  - Input Representation (3 loại embedding)
  - Các token đặc biệt: [CLS], [SEP], [MASK]
  - Pre-training: MLM + NSP
  - Fine-tuning cho các bài toán khác nhau
  - Các biến thể: RoBERTa, ALBERT, DistilBERT, PhoBERT
- [ ] **B5**: Làm slide trình bày (~15-20 slides)
  - Slide 1: Tiêu đề
  - Slide 2-3: Bối cảnh và động lực (tại sao cần BERT)
  - Slide 4-6: Kiến trúc Transformer (Self-Attention, Multi-Head)
  - Slide 7-10: Kiến trúc BERT (Input, Encoder, Output)
  - Slide 11-13: Pre-training (MLM, NSP)
  - Slide 14-15: Fine-tuning (Classification, NER, QA)
  - Slide 16-17: Kết quả và so sánh
  - Slide 18-19: Các biến thể và ảnh hưởng
  - Slide 20: Liên kết với Phần 1 (demo sentiment analysis)
- [ ] **B6**: Viết phần lý thuyết trong quyển báo cáo (~15 trang)
  - Cấu trúc tương tự slide nhưng chi tiết hơn
  - Bao gồm công thức toán, hình vẽ kiến trúc
  - Trích dẫn paper gốc

---

## TIẾN ĐỘ DỰ KIẾN

| Tuần | Công việc | Output |
|---|---|---|
| **Tuần 1** | Thu thập dataset, đọc paper BERT, EDA | Dataset sạch, hiểu tổng quan BERT |
| **Tuần 2** | Tiền xử lý, xây dựng mô hình, train | Model trained, accuracy > 85% |
| **Tuần 3** | Đánh giá, làm demo UI, hiểu sâu BERT | Demo chạy được, slide lý thuyết |
| **Tuần 4** | Hoàn thiện slide + quyển báo cáo | Slide + báo cáo hoàn chỉnh, tập vấn đáp |

---

## CẤU TRÚC THƯ MỤC DỰ KIẾN

```
UTC_DEEP_LEARNING/
|-- README.md
|-- KE_HOACH_THUC_HIEN.md              # File này
|-- DANH_GIA_DE_TAI.md                  # Đánh giá 25 đề tài
|-- DANH_SACH_CHU_DE_BAI_TAP_LON.md    # Danh sách chủ đề gốc
|
|-- phan1_sentiment/                     # PHẦN 1: Demo
|   |-- notebooks/
|   |   |-- 01_eda.ipynb                 # Phân tích khám phá dữ liệu
|   |   |-- 02_preprocessing.ipynb       # Tiền xử lý
|   |   |-- 03_training.ipynb            # Huấn luyện mô hình
|   |   |-- 04_evaluation.ipynb          # Đánh giá kết quả
|   |-- src/
|   |   |-- data_loader.py               # Đọc và xử lý dữ liệu
|   |   |-- model.py                     # Định nghĩa mô hình
|   |   |-- train.py                     # Script huấn luyện
|   |   |-- predict.py                   # Script dự đoán
|   |-- app/
|   |   |-- streamlit_app.py             # Giao diện demo
|   |-- data/
|   |   |-- raw/                         # Dataset gốc
|   |   |-- processed/                   # Dataset đã xử lý
|   |-- models/                          # Model checkpoints
|   |-- requirements.txt
|
|-- phan2_bert/                          # PHẦN 2: Lý thuyết
|   |-- slides/                          # Slide trình bày
|   |-- figures/                         # Hình vẽ kiến trúc
|   |-- references/                      # Tài liệu tham khảo
|
|-- bao_cao/                             # Quyển báo cáo tổng hợp
|   |-- bao_cao.docx
|   |-- bao_cao.pdf
```

---

## CÂU HỎI VẤN ĐÁP THƯỜNG GẶP (CHUẨN BỊ TRƯỚC)

### Về Phần 1 (Demo)

1. **Tại sao chọn BERT mà không dùng LSTM cho sentiment analysis?**
   → BERT là mô hình pre-trained trên lượng dữ liệu khổng lồ (3.3 tỷ từ), đã học được ngữ nghĩa phong phú. LSTM phải train from scratch trên dataset nhỏ nên thiếu kiến thức ngôn ngữ. BERT thường đạt accuracy cao hơn LSTM 3-5% trên các benchmark sentiment analysis.

2. **Token [CLS] là gì và tại sao dùng nó cho classification?**
   → [CLS] là token đặc biệt đặt ở đầu mỗi câu. Trong quá trình pre-training, BERT được huấn luyện để output của [CLS] chứa biểu diễn ngữ nghĩa tổng hợp của toàn bộ câu. Do đó, ta chỉ cần lấy output [CLS] rồi đưa qua lớp Dense để phân loại.

3. **Overfitting là gì và cách xử lý?**
   → Overfitting là khi mô hình học thuộc dữ liệu train nhưng không tổng quát được trên dữ liệu mới. Cách xử lý: dùng Dropout (tắt ngẫu nhiên 30% neuron), Early Stopping (dừng khi val_loss tăng), Weight Decay (phạt trọng số lớn).

4. **Tại sao learning rate cho fine-tuning BERT rất nhỏ (2e-5)?**
   → Vì BERT đã được pre-train tốt, các trọng số đã ở vùng tối ưu. Learning rate lớn sẽ phá hỏng kiến thức đã học (catastrophic forgetting). Learning rate nhỏ giúp tinh chỉnh nhẹ nhàng cho bài toán cụ thể.

### Về Phần 2 (Lý thuyết BERT)

5. **BERT khác GPT ở điểm nào?**
   → BERT dùng Transformer **Encoder** (bidirectional - hai chiều), GPT dùng Transformer **Decoder** (unidirectional - một chiều trái sang phải). BERT tốt cho bài toán hiểu ngôn ngữ (classification, NER, QA). GPT tốt cho bài toán sinh ngôn ngữ (text generation, chatbot).

6. **Masked Language Model hoạt động thế nào?**
   → Che 15% tokens trong câu, mô hình dự đoán từ bị che dựa trên ngữ cảnh hai bên. Trong 15%: 80% thay bằng [MASK], 10% thay bằng từ ngẫu nhiên, 10% giữ nguyên. Tỷ lệ 80/10/10 giúp tránh mismatch giữa pre-training và fine-tuning.

7. **Self-Attention là gì? Giải thích đơn giản?**
   → Self-Attention cho phép mỗi từ trong câu "nhìn" tất cả các từ khác để hiểu ngữ cảnh. Mỗi từ tạo 3 vector: Query (tôi tìm gì), Key (tôi chứa gì), Value (giá trị thực). Tính tích Q*K để biết mức độ liên quan, nhân với V để lấy thông tin cần thiết.

8. **BERT có bao nhiêu tham số? Tại sao cần nhiều vậy?**
   → BERT Base có 110 triệu tham số, BERT Large có 340 triệu. Nhiều tham số giúp mô hình học được các pattern ngữ nghĩa phức tạp từ 3.3 tỷ từ dữ liệu. Đây là trade-off: nhiều tham số = hiểu sâu hơn nhưng cần nhiều tài nguyên tính toán hơn.

---

## LƯU Ý QUAN TRỌNG

1. **Chương trình máy tính KHÔNG được có comment** (yêu cầu của thầy) — code phải tự giải thích qua tên biến, tên hàm rõ ràng
2. **Quyển báo cáo <= 30 trang**, bìa mềm, bao gồm cả Phần 1 và Phần 2
3. **Vấn đáp**: Chuẩn bị trả lời cả về code (cách hoạt động, tham số) lẫn lý thuyết BERT (kiến trúc, pre-training, fine-tuning)
4. **Liên kết 2 phần**: Dùng BERT làm model cho Phần 1 → giải thích lý thuyết BERT ở Phần 2 → tạo câu chuyện mạch lạc
5. **Phân bổ điểm**: Báo cáo 2.5đ + Slide 2.5đ + Vấn đáp 5đ = **10 điểm** (vấn đáp chiếm trọng số lớn nhất)
