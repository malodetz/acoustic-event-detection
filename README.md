# Acoustic event detection task for Digital Signal Processing course

1. Изменения по сравнению с бэйзлайном в модели:

Изменены размеры mel-спектрограммы, чтобы "изображение" имело размеры 128x256

Чуть-чуть измены последние слои классификатора, чтобы они были степенями двойки

Добавлена нормализация по батчу

Сделан небольшой рефакторинг:

```python
class SoundModel(nn.Module):
    def __init__(self, sample_rate=16000, n_classes=41):
        super().__init__()
        self.ms = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, hop_length=126
        )

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=3, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            EfficientNet.from_pretrained("efficientnet-b0"),
            nn.ReLU(),
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.layers(self.ms(x))

    def inference(self, x):
        x = self.forward(x)
        x = F.softmax(x, dim=1)
        return x
```
2. Изменения в процессе обучения

Модель обучается в режиме кросс-валидации, 
т.е. тренировочные данные разбиваются на 5 примерно равных фолдов, но не случайным образом, 
а так, чтобы в каждом из них процент примеров каждого класса был таким же, как в целой выборке.
По очереди каждый из фолдов становится валидационный выборкой, а остальные 4 - тренировочной.
В каждом случае проходит 100 эпох с использованием scheduler-a, аналогичного оригинальному ходу обучения.
Все 5 уже обученных модей есть в репозитории.

3. Получение предсказаний

Полученные 5 моделей (каждая имеет лучший f1 на своей валидационной выборке) используются в виде ансамбля:
Для каждой считаются вероятности того, что объект принадлежит каждомку из 41 классов, 
затем эти 5 векторов вероятностей складываются и уже из этого вектора выбирается класс с наибольшим значением.

