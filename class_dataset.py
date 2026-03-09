import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DotaDataSet(Dataset):
    def __init__(self, matches_data, augment=False):
        self.matches_data = torch.tensor(matches_data,dtype=torch.long)
        self.augment = augment
        
    def __len__(self):
        # Каждый матч дает нам 10 обучающих примеров
        return len(self.matches_data) * 10

    def __getitem__(self, idx):
        # 1. Получаем индексы матча и позиции
        match_idx = idx // 10
        pos_idx = idx % 10

        matches_heroes = self.matches_data[match_idx]
        
        # Логика по сторонам (Radiant / Dire)
        if pos_idx < 5:
            # Предсказываем за Radiant
            target_hero = matches_heroes[pos_idx]
            position = pos_idx
            # Берем всех Radiant (индексы 0-4), кроме самого себя (target)
            allies = np.delete(matches_heroes[0:5], pos_idx) # ИСПРАВЛЕНО
            # Враги - все 5 героев Dire
            enemies = matches_heroes[5:10].clone()
        else:
            # Предсказываем за Dire
            target_hero = matches_heroes[pos_idx]
            position = pos_idx - 5 # Возвращаем к 0-4
            # Берем всех Dire (индексы 5-9), кроме самого себя
            allies = np.delete(matches_heroes[5:10], position) # ИСПРАВЛЕНО (pos_idx - 5 это и есть position)
            # Враги - все 5 героев Radiant
            enemies = matches_heroes[0:5].clone()
            
        if self.augment:
            num_hidden_allies = np.random.randint(0, 4) # Нужно оставить минимум 1 союзника
            if num_hidden_allies > 0:
                
                hide_ally_idx = np.random.choice(4, num_hidden_allies, replace=False)
                allies[hide_ally_idx] = 0 # 0 = PAD
                
            num_hidden_enemies = np.random.randint(0, 5)
            if num_hidden_enemies > 0:
                
                hide_enemy_idx = np.random.choice(5, num_hidden_enemies, replace=False)
                enemies[hide_enemy_idx] = 0 # 0 = PAD
                
        return {
            "allies": torch.tensor(allies, dtype=torch.long),
            "enemies": torch.tensor(enemies, dtype=torch.long),
            "position": torch.tensor(position, dtype=torch.long),
            "target": torch.tensor(target_hero, dtype=torch.long)
        }
    # ТЕСТ И ПОДГОТОВКА ДАННЫХ (SPLIT / DATALOADER) 
def creator_loader(csv_file,batch_size=32,test_size=0.2):
    df=pd.read_csv(csv_file)

    # 4. Train / Val split ПО МАТЧАМ (Data Leakage prevention)
    r_cols = ['r_pos1', 'r_pos2', 'r_pos3', 'r_pos4', 'r_pos5']
    d_cols = ['d_pos1', 'd_pos2', 'd_pos3', 'd_pos4', 'd_pos5']

    # Преобразуем датафрейм в массив [N_matches, 10]
    matches_array=df[r_cols+d_cols].values.astype(np.int64)

    
    train_matches, val_matches = train_test_split(matches_array, test_size=test_size, random_state=42)
    
    print(f"Всего матчей: {len(matches_array)}. Train: {len(train_matches)}, Val: {len(val_matches)}")
    print(f"Примеров для обучения: {len(train_matches) * 10}")
    
    # 5. Создаем Dataset
    train_dataset = DotaDataSet(train_matches, augment=True) # Аугментация нужна только на трейне!
    val_dataset = DotaDataSet(val_matches, augment=False)    # Валидация идет на полных драфтах
    
    # 6. Создаем DataLoaders
    # shuffle=True только для трейна
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
# Допустим, мы берем очень маленький батч (batch_size=2), 
# чтобы вывод в консоли не был гигантским.
if __name__ == "__main__":

    train_loader, val_loader = creator_loader(
        "dota_ml_current_patch.csv",
        batch_size=2
    )

    batch = next(iter(train_loader))

    print("Ключи словаря:", batch.keys())

    print("\n--- Союзники (allies) ---")
    print("Shape:", batch["allies"].shape)
    print(batch["allies"])

    print("\n--- Враги (enemies) ---")
    print("Shape:", batch["enemies"].shape)
    print(batch["enemies"])

    print("\n--- Позиция (position) ---")
    print("Shape:", batch["position"].shape)
    print(batch["position"])

    print("\n--- Целевой герой (target) ---")
    print("Shape:", batch["target"].shape)
    print(batch["target"])