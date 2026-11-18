from pathlib import Path
import torch
from config import *
from tokenizer import CharTokenizer
from model import VLanguageModel

torch.manual_seed(SEED)
data_path = Path("data/train.txt")
if not data_path.exists():
    raise FileNotFoundError("Создай data/train.txt")
text = data_path.read_text(encoding="utf-8")
if len(text) < BLOCK_SIZE * 3:
    raise ValueError("Добавь больше текста в датасет")

tokenizer = CharTokenizer(text)
Path("checkpoints").mkdir(exist_ok=True)
tokenizer.save("checkpoints/tokenizer.json")
tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
split = int(len(tokens) * 0.9)
train_data, val_data = tokens[:split], tokens[split:]

def get_batch(source):
    starts = torch.randint(0, len(source) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([source[i:i+BLOCK_SIZE] for i in starts])
    y = torch.stack([source[i+1:i+BLOCK_SIZE+1] for i in starts])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def evaluate(model):
    model.eval()
    result = {}
    for name, source in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(20):
            xb, yb = get_batch(source)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        result[name] = sum(losses) / len(losses)
    model.train()
    return result

model = VLanguageModel(tokenizer.vocab_size).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
print(f"v 0.0.1 | PyTorch {torch.__version__} | vocab {tokenizer.vocab_size}")
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

best_val = float("inf")
for step in range(MAX_ITERS):
    if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
        metrics = evaluate(model)
        print(f"step {step}/{MAX_ITERS} | train {metrics['train']:.4f} | val {metrics['val']:.4f}")
        if metrics["val"] < best_val:
            best_val = metrics["val"]
            torch.save({"model": model.state_dict(), "vocab_size": tokenizer.vocab_size},
                       "checkpoints/best_model.pt")
    xb, yb = get_batch(train_data)
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Обучение завершено. Лучшая модель сохранена в checkpoints/best_model.pt")
