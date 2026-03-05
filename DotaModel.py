import torch
import torch.nn as nn
import torch.nn.functional as F

class DotaModel(nn.Module):
    def __init__(self, num_heros=155,emb_dim=64,pos_dim=16,hidden_dim=256):
        """
        num_heroes: Максимальный ID героя + 1 (с запасом)
        emb_dim: Размер вектора героя (64)
        pos_dim: Размер вектора позиции (16)
        """
        super(DotaModel,self).__init__()
        # 1. Эмбеддинги
        # padding_idx=0 гарантирует, что вектор для ID=0 всегда будет нулевым и не будет обучаться.
        self.hero_embedding=nn.Embedding(num_heros,emb_dim,padding_idx=0)
        self.pos_embedding=nn.Embedding(5,pos_dim)

        # 2. MLP (Многослойный перцептрон)
        # Входной размер: Ally_Vector(64) + Enemy_Vector(64) + Pos_Vector(16) = 144
        input_dim=emb_dim+ emb_dim+pos_dim

        self.net=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),# 144 -> 256 (Расширение признаков)
            nn.ReLU(),# Нелинейность для синергии
            nn.Dropout(0.2),# Защита от зубрежки (Drop 20%)
            nn.Linear(hidden_dim,hidden_dim//2),# 256 -> 128 (Сжатие сути)
            nn.ReLU(),
            nn.Linear(hidden_dim//2,num_heros)# 128 -> 160 (Логиты для каждого героя)
        )

        # Инициализация весов (Optional, но полезно для стабильного старта)
        self._init_weights()
    def _init_weights(self):
        # Kaiming He init отлично подходит для ReLU слоев
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self,allies,enemies,possition):
        #1.Получение векторов (Lookup)
        a_emd=self.hero_embedding(allies)# [batch, 4, 64]
        e_emd=self.hero_embedding(enemies)# [batch, 5, 64]
        p_emd=self.pos_embedding(possition)# [batch, 16]
        #2. Masked Mean Pooling
        # Создаем маски: 1 если герой есть, 0 если это паддинг (0)
        # unsqueeze(-1) нужен, чтобы размерность стала [batch, N, 1] для умножения на эмбеддинг
        a_mask=(allies!=0).float().unsqueeze(-1)
        e_mask=(enemies!=0).float().unsqueeze(-1)

        # Суммируем векторы
        a_sum=(a_emd*a_mask).sum(dim=1)
        e_sum=(e_emd*e_mask).sum(dim=1)

        # Считаем количество реальных героев (избегаем деления на 0 с помощью clamp или eps)
        a_count=a_mask.sum(dim=1).clamp(min=1.0)
        e_count = e_mask.sum(dim=1).clamp(min=1.0)

        # Получаем честное среднее
        a_mean = a_sum / a_count  # [batch, 64]
        e_mean = e_sum / e_count  # [batch, 64]

        #Шаг 3: Объединение

        combined=torch.cat([a_mean,e_mean,p_emd],dim=1)# [batch, 144]

        #Шаг 4: Предсказание (Prediction)

        logits=self.net(combined)# [batch, 160]
        return logits
"""""    
# Тестовый прогон
if __name__ == "__main__":
    # Создаем модель
    model = DotaModel(num_heros=160)
    print("Модель создана!")
    
    # Генерируем фейковый батч (как будто от DataLoader)
    batch_size = 2
    dummy_allies = torch.tensor([[10, 20, 0, 0], [5, 0, 0, 0]]) # 2 примера
    dummy_enemies = torch.tensor([[99, 100, 101, 102, 103], [1, 2, 3, 4, 5]])
    dummy_pos = torch.tensor([0, 4]) # Керри и Саппорт
    
    # Прогоняем через модель
    output = model(dummy_allies, dummy_enemies, dummy_pos)
    
    print(f"\nРазмер выхода: {output.shape}") 
    # Ожидаем: torch.Size([2, 160])
    
    print("Проверка пройдена: Логиты сгенерированы.")
    """