
import torch
from config import DEVICE, BLOCK_SIZE
from tokenizer import CharTokenizer
from model import VLanguageModel

# загрузка модели
tokenizer = CharTokenizer.load("checkpoints/tokenizer.json")
checkpoint = torch.load(
    "checkpoints/best_model.pt",
    map_location=DEVICE
)

model = VLanguageModel(checkpoint["vocab_size"]).to(DEVICE)
model.load_state_dict(checkpoint["model"])
model.eval()

history = []

print("v запущена. /exit — выход, /reset — очистить историю.\n")

while True:
    prompt = input("ты: ").strip()

    if prompt == "/exit":
        print("v: отключаюсь.")
        break

    if prompt == "/reset":
        history.clear()
        print("v: история очищена.\n")
        continue

    if not prompt:
        continue

    try:
        user_ids = tokenizer.encode(prompt)
    except ValueError as exc:
        print(f"v: неизвестные символы: {exc}\n")
        continue

    # добавляем разделитель между сообщениями
    try:
        separator = tokenizer.encode("\n")
    except ValueError:
        separator = []

    history.extend(separator + user_ids)

    # ограничиваем контекст размером модели
    context_ids = history[-BLOCK_SIZE:]

    context = torch.tensor(
        [context_ids],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():
        output = model.generate(
            context,
            max_new_tokens=250
        )

    # получаем только новую генерацию
    response_ids = output[0, len(context_ids):].tolist()
    response = tokenizer.decode(response_ids)

    print(f"v: {response}\n")

    # сохраняем ответ в истории
    history = output[0].tolist()
