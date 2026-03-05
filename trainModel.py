import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm # Красивый прогресс-бар

from class_dataset import DotaDataSet
from DotaModel import DotaModel 
# 1. Вспомогательная функция для метрик
def calculate_accuracy(logits, targets, k=5):

    with torch.no_grad():
        batch_size=targets.size(0)
        # Получаем индексы топ-k предсказаний
        _,pred=logits.topk(k,1,True,True)
        pred=pred.t()
       # Сравниваем с правильным ответом (расширяем target под размер pred)
        correct=pred.eq(targets.view(1,-1).expand_as(pred))
        # Top-1: Правильный ответ в первой строке
        correct_1=correct[:1].reshape(-1).float().sum(0,keepdim=True)
        # Top-k: Правильный ответ в любой из k строк
        correct_k=correct[:k].reshape(-1).float().sum(0,keepdim=True)
        return correct_1.mul_(100.0/batch_size),correct_k.mul_(100.0/batch_size)
# 2. Функция одной эпохи обучения
def train_epoch(model,loader,criterrion,optimizer,device):
    model.train()

    running_loss=0.0
    running_top1=0.0
    running_top5=0.0

    pbar = tqdm(loader, desc="Training", leave=False)

    for batch in pbar:
        # 1. Распаковка данных и перенос на GPU/CPU
        allies=batch['allies'].to(device)
        enemies = batch['enemies'].to(device)
        pos = batch['position'].to(device)      
        targets = batch['target'].to(device)

        # 2. Обнуляем градиенты
        optimizer.zero_grad()
        # 3. Forward pass
        logits=model(allies,enemies,pos)

        # 4. Расчет ошибки
        loss=criterrion(logits,targets)

        # 5. Backward pass & Step
        loss.backward()

        # Защита от взрыва градиентов
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        
        optimizer.step()
        # Метрики и логирование 
        acc1,acc5=calculate_accuracy(logits,targets,k=5)

        running_loss+=loss.item()
        running_top1+=acc1.item()
        running_top5+=acc5.item()

        pbar.set_postfix({'Loss': loss.item(), 'Top1': acc1.item(), 'Top5': acc5.item()})

    # Средние значения за эпоху
    epoch_loss=running_loss/len(loader)
    epoch_top1=running_top1/len(loader)
    epoch_top5=running_top5/len(loader)

    return epoch_loss, epoch_top1, epoch_top5
# --- 3. Основной запуск ---
def run_training():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {DEVICE}")

    # 1. Данные 
    dataset = DotaDataSet('dota_games.csv') 
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 2. Инициализация
    model=DotaModel(num_heros=156).to(DEVICE)

    # 3.Опитимизатор 
    optimizer=optim.Adam(model.parameters(),lr=0.001)

    # Loss Function
    criterion=nn.CrossEntropyLoss()

     # 3. Цикл по эпохам
    print("Начинаем обучение...")

    for epoch in range(10):
        loss,acc1,acc5=train_epoch(model,train_loader,criterion,optimizer,DEVICE)

        print(f"Epoch {epoch+1}/{10}")
        print(f"  Loss: {loss:.4f} | Top-1 Acc: {acc1:.2f}% | Top-5 Acc: {acc5:.2f}%")
        print("-" * 30)

        torch.save(model.state_dict(),f"dota_model_epoch_{epoch+1}.pth")
    print("Обучение закончено!")

if __name__ == "__main__":
    run_training()