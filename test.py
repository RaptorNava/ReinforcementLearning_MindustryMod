import torch

ckpt = torch.load("mindustry_agent.pt", map_location="cpu")

print("Ключи чекпоинта:", ckpt.keys())
print(f"Эпизод: {ckpt.get('episode')}")
print(f"Всего шагов: {ckpt.get('total_steps')}")
print(f"Лучшая награда: {ckpt.get('best_reward')}")

model_keys = list(ckpt['model'].keys())
print(f"Количество тензоров в модели: {len(model_keys)}")
print("Примеры слоев:", model_keys[:5])
