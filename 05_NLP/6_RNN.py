# title: RNN Language Model
# author: Taewook Kang (laputa99999@gmail.com)
# description: Simple RNN-based language model implemented in PyTorch.
# limitation: 
#   1. Insufficient Data
#     The training sequence used only 11 sentences, a very small number. RNN language models require more data to generalize and learn patterns.
#     With insufficient data, the model quickly memorizes all the patterns, leaving no room for further improvement.
#
#   2. Data Imbalance Compared to Model Capacity
#     The parameters of the model (VanillaRNNLM) (emb_dim=64, hidden_dim=128) are large compared to the data size.
#     With insufficient data, the model may not learn sufficiently, or the loss may stop decreasing after overfitting.
#
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(7)

# 1) 학습 문장 준비(*주: 실용적으로 사용하기 위해서는 훨씬 많은 데이터가 필요) 
sentences = [
    "오늘 날씨가 좋아서 친구와 공원으로 산책을 갔다",
    "어제 비가 많이 와서 우산을 쓰고 출근했다",
    "아침에 커피를 마시고 책을 조금 읽었다",
    "점심에는 동료들과 따뜻한 국수를 먹었다",
    "퇴근 후에 헬스장에서 가볍게 운동을 했다",
    "주말에는 가족과 함께 바다에 다녀왔다",
    "오랜만에 영화를 보고 늦게 집에 돌아왔다",
    "도서관에서 과제를 끝내고 조용히 쉬었다",
    "친구 생일이라 작은 케이크를 선물했다",
    "저녁에 산책하며 노을을 천천히 감상했다",
    "새로 산 이어폰으로 음악을 들으며 이동했다",
]

# 특수 토큰
BOS, EOS, PAD = "<BOS>", "<EOS>", "<PAD>"

def tokenize(s):  # 매우 단순: 공백 기준
    return s.strip().split()

# 2) 토큰화 & 사전 구축  
all_tokens = []
tok_sentences = []
for s in sentences:
    toks = [BOS] + tokenize(s) + [EOS]
    tok_sentences.append(toks)
    all_tokens.extend(toks)
vocab = sorted(set(all_tokens + [PAD]))
stoi = {t:i for i,t in enumerate(vocab)}
itos = {i:t for t,i in stoi.items()}
vocab_size = len(vocab)
pad_id = stoi[PAD]

# 3) 인코딩 + 입력/타깃 쌍 만들기  
encoded = [[stoi[t] for t in toks] for toks in tok_sentences]
# 입력: [BOS, w1, w2, ...]  -> 타깃: [w1, w2, ..., EOS]
inputs = [seq[:-1] for seq in encoded]
targets = [seq[1:]  for seq in encoded]
max_len = max(len(x) for x in inputs)

def pad_to_max(ids, max_len, pad=pad_id):
    return ids + [pad]*(max_len - len(ids))

X = [pad_to_max(x, max_len) for x in inputs]
Y = [pad_to_max(y, max_len) for y in targets]

# 4) 데이터셋/로더  
class LMDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]

ds = LMDataset(X, Y)
dl = DataLoader(ds, batch_size=4, shuffle=True)

# 5) 바닐라 RNN 언어모델  
class VanillaRNNLM(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.RNN(emb_dim, hidden_dim, batch_first=True, nonlinearity='tanh')
        self.fc  = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h0=None):
        e = self.emb(x)               # (B,T,E)
        out, h = self.rnn(e, h0)      # (B,T,H)
        logits = self.fc(out)         # (B,T,V)
        return logits, h

model = VanillaRNNLM(vocab_size)
criterion = nn.CrossEntropyLoss(ignore_index=pad_id)  # PAD는 무시
optimizer = optim.Adam(model.parameters(), lr=2e-3)

# 6) 학습 루프  
epochs = 50
for ep in range(1, epochs+1):
    model.train()
    total = 0.0
    for x, y in dl:
        optimizer.zero_grad()
        logits, _ = model(x)  # (B,T,V)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total += loss.item()
    if ep % 5 == 0:
        print(f"[epoch {ep:02d}] loss={total/len(dl):.4f}")

# 7) 문장 생성 함수  
@torch.no_grad()
def generate(max_steps=30, greedy=True, seed=None):
    model.eval()
    if seed is None:
        cur = torch.tensor([[stoi[BOS]]], dtype=torch.long)
    else:
        # seed 문장을 토큰화해 BOS와 함께 시작
        toks = [BOS] + tokenize(seed)
        cur = torch.tensor([[stoi.get(t, stoi[PAD]) for t in toks]], dtype=torch.long)
    h = None
    out = []
    for _ in range(max_steps):
        logits, h = model(cur, h)
        last = logits[:, -1, :]  # 마지막 위치 분포
        if greedy:
            next_id = last.argmax(dim=-1)
        else:
            probs = last.softmax(dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).squeeze(1)
        token = itos[next_id.item()]
        if token == EOS or token == PAD:
            break
        out.append(token)
        cur = torch.tensor([[next_id.item()]], dtype=torch.long)
    return " ".join(out)

# 8) 테스트: 생성  
print("\n[generation from <BOS>]")
print(generate())

print("\n[generation with seed: '저녁에']")
print(generate(seed="저녁에"))

print("\n[generation with seed: '오늘']")
print(generate(seed="오늘"))
