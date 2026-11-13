import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import gzip
import base64
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
INPUT_AE_DIR = "${SERVER_DATA_ROOT}/prediction_travel_mode/trace_generate/code/output_ae_test/"
OUTPUT_PRED_FILE = "predicted_labels.parquet"
MODEL_CHECKPOINT = "urban_transformer_cls_best_ab.pth"

CENTER_FEATURES = ['start_to_primary_m', 'end_to_primary_m', 'start_to_nearest_sub_m', 'end_to_nearest_sub_m',
                   'trip_min_to_primary_m', 'trip_min_to_nearest_sub_m', 'od_ratio_primary', 'od_ratio_nearest_sub',
                   'city_total_centers']

USE_ENV = True
USE_LANDUSE = False
USE_TRAFFIC = False
USE_CENTER = False

BATCH_SIZE = 32
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 工具函数 ====================
def decompress_patch(b64_str: str) -> np.ndarray:
    try:
        compressed = base64.b64decode(b64_str)
        raw = gzip.decompress(compressed)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(64, 11, 11)
        return arr
    except Exception:
        return np.zeros((64, 11, 11), dtype=np.float32)

def extract_patches_from_row(b64_field: str):
    if not isinstance(b64_field, str) or b64_field.strip() == '':
        return np.zeros((1, 64, 11, 11), dtype=np.float32)
    patch_strs = [p.strip() for p in b64_field.split('|') if p.strip()]
    if not patch_strs:
        return np.zeros((1, 64, 11, 11), dtype=np.float32)
    patches = [decompress_patch(p) for p in patch_strs]
    return np.stack(patches)  # [T, 64, 11, 11]

# ==================== 模型定义（与训练一致）====================
# （此处省略重复的类定义，保持与你提供的完全一致）
class AttentionPool(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.attn = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.proj = nn.Linear(in_channels, embed_dim)
    def forward(self, x):
        B_T, C, H, W = x.shape
        weights = torch.sigmoid(self.attn(x))
        weighted = x * weights
        pooled = weighted.mean(dim=[2, 3])
        return self.proj(pooled)

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.max_len = max_len
    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.pos_embed(positions)

class SimpleResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(8, channels)
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class PatchEncoder(nn.Module):
    def __init__(self, input_channels=64, embed_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(8, 64)
        self.res_block = SimpleResidualBlock(64)
        self.pool = nn.MaxPool2d(2)
        self.attn_pool = AttentionPool(64, embed_dim)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.res_block(x)
        x = self.pool(x)
        x = self.attn_pool(x)
        return x

class TokenEmbedder(nn.Module):
    def __init__(self, num_tokens=5, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens + 1, embed_dim, padding_idx=0)
    def forward(self, x):
        return self.embedding(x)

class UrbanMultiModalClassifier(nn.Module):
    def __init__(self, num_classes=4, input_channels=64, embed_dim=128, nhead=8, num_layers=6,
                 use_env=True, use_landuse=True, use_traffic=True, use_center=True):
        super().__init__()
        self.use_env = use_env
        self.use_landuse = use_landuse
        self.use_traffic = use_traffic
        self.use_center = use_center
        self.embed_dim = embed_dim

        if self.use_env:
            self.patch_encoder = PatchEncoder(input_channels, embed_dim)
            self.self_attn_img = nn.MultiheadAttention(embed_dim, num_heads=nhead // 2, batch_first=True)
        if self.use_landuse:
            self.token_embedder = TokenEmbedder(num_tokens=5, embed_dim=embed_dim)
        if self.use_traffic:
            self.traffic_embedder = TokenEmbedder(num_tokens=3, embed_dim=embed_dim)
        if self.use_center:
            self.center_encoder = nn.Sequential(
                nn.Linear(len(CENTER_FEATURES), embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
        self.embed_norm = nn.LayerNorm(embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_encoder = LearnablePositionalEmbedding(embed_dim)
        self.final_norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=512
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_img, x_land, x_traffic, x_center, lengths):
        B = 0
        if x_land is not None: B = x_land.size(0)
        elif x_traffic is not None: B = x_traffic.size(0)
        elif x_img is not None: B = x_img.size(0)
        elif x_center is not None: B = x_center.size(0)
        else: raise ValueError("At least one input modality must be provided.")

        embeddings_img = None
        if self.use_env and x_img is not None:
            T = x_img.shape[1]
            x_flat = x_img.view(B * T, x_img.shape[2], x_img.shape[3], x_img.shape[4])
            embeddings_img = self.patch_encoder(x_flat)
            embeddings_img = embeddings_img.view(B, T, -1)
            embeddings_img = self.embed_norm(embeddings_img)

        embeddings_land = None
        if self.use_landuse and x_land is not None:
            embeddings_land = self.token_embedder(x_land)
            embeddings_land = self.embed_norm(embeddings_land)

        embeddings_traffic = None
        if self.use_traffic and x_traffic is not None:
            embeddings_traffic = self.traffic_embedder(x_traffic)
            embeddings_traffic = self.embed_norm(embeddings_traffic)

        has_center = self.use_center and x_center is not None
        center_emb = None
        if has_center:
            center_emb = self.center_encoder(x_center).unsqueeze(1)
            center_emb = self.embed_norm(center_emb)

        embeddings_list = []
        if embeddings_img is not None: embeddings_list.append(embeddings_img)
        if embeddings_land is not None: embeddings_list.append(embeddings_land)
        if embeddings_traffic is not None: embeddings_list.append(embeddings_traffic)
        if has_center: embeddings_list.append(center_emb)

        if not embeddings_list:
            raise ValueError("No embeddings generated!")

        combined = torch.cat(embeddings_list, dim=1)

        if self.use_env and embeddings_img is not None and len(embeddings_list) == 1:
            combined, _ = self.self_attn_img(embeddings_img, embeddings_img, embeddings_img)
        combined = self.final_norm(combined)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        embeddings = torch.cat([cls_tokens, combined], dim=1)
        embeddings = self.pos_encoder(embeddings)

        total_seq_len = embeddings.shape[1] - 1
        if total_seq_len > 0:
            if lengths is not None:
                main_len = lengths if not has_center else lengths + 1
                pad_mask = torch.arange(total_seq_len, device=embeddings.device).unsqueeze(0) >= main_len.unsqueeze(1)
            else:
                pad_mask = torch.zeros(B, total_seq_len, dtype=torch.bool, device=embeddings.device)
        else:
            pad_mask = torch.empty(B, 0, dtype=torch.bool, device=embeddings.device)

        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=embeddings.device)
        full_mask = torch.cat([cls_mask, pad_mask], dim=1)

        encoded = self.transformer(embeddings, src_key_padding_mask=full_mask)
        cls_out = encoded[:, 0, :]
        logits = self.classifier(cls_out)
        return logits, cls_out

# ==================== 推理主函数（支持无 seg_id）====================
def main():
    print(f"Using device: {DEVICE}")

    model = UrbanMultiModalClassifier(
        num_classes=NUM_CLASSES,
        use_env=USE_ENV,
        use_landuse=USE_LANDUSE,
        use_traffic=USE_TRAFFIC,
        use_center=USE_CENTER
    ).to(DEVICE)

    state_dict = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded successfully.")

    parquet_files = sorted(glob.glob(os.path.join(INPUT_AE_DIR, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {INPUT_AE_DIR}")

    all_results = []
    global_row_id = 0  # 全局唯一 ID

    with torch.no_grad():
        for file_path in tqdm(parquet_files, desc="Processing files"):
            df = pd.read_parquet(file_path)
            if 'ae_patch_11x11x64_b64' not in df.columns:
                print(f"⚠️ Skipping {file_path}: missing 'ae_patch_11x11x64_b64'")
                continue

            patches_list = []
            local_indices = []  # 记录有效行的索引（用于生成 ID）

            for idx, row in df.iterrows():
                b64_str = row['ae_patch_11x11x64_b64']
                patches = extract_patches_from_row(b64_str)
                if patches.size == 0 or np.all(patches == 0):
                    continue
                patches_list.append(torch.from_numpy(patches).float())
                local_indices.append(idx)

            if not patches_list:
                continue

            # 批量推理
            for i in range(0, len(patches_list), BATCH_SIZE):
                batch_patches = patches_list[i:i + BATCH_SIZE]
                batch_local_indices = local_indices[i:i + BATCH_SIZE]

                padded = torch.nn.utils.rnn.pad_sequence(batch_patches, batch_first=True, padding_value=0.0)
                padded = padded.to(DEVICE)
                lengths = torch.tensor([p.shape[0] for p in batch_patches], dtype=torch.long).to(DEVICE)

                logits, _ = model(
                    x_img=padded,
                    x_land=None,
                    x_traffic=None,
                    x_center=None,
                    lengths=lengths
                )
                preds = logits.argmax(dim=1).cpu().numpy()

                # 生成唯一 ID：可以用 (文件名, 行号) 或全局 ID
                for local_idx, pred in zip(batch_local_indices, preds):
                    # 方式1：全局递增 ID（推荐，简单唯一）
                    unique_id = str(global_row_id)
                    global_row_id += 1

                    # 方式2（可选）：用文件名+行号 → f"{Path(file_path).stem}_{local_idx}"
                    # unique_id = f"{Path(file_path).stem}_{local_idx}"

                    all_results.append({'seg_id': unique_id, 'pred_label': int(pred)})

    result_df = pd.DataFrame(all_results)
    result_df.to_parquet(OUTPUT_PRED_FILE, index=False, compression='snappy')
    print(f"\n✅ Inference completed. Results saved to {OUTPUT_PRED_FILE}")
    print(f"Total predictions: {len(result_df)}")

if __name__ == "__main__":
    main()