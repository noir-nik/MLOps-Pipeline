# GitHub Actions Workflow Documentation
 
## Обзор

The workflow automates the continuous integration and deployment process for the ML pipeline, enabling:

1. Автоматический запуск при push на master
2. Запланированное инкрементное обучение модели
3. Сохранение состояния модели между запусками
4. Сбор и хранение артефактов (логи, модели, отчеты)

## Триггеры Workflow

Workflow запускается следующими событиями:

```yaml
on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main
  schedule:
    - cron: '0 0 * * *'
```

- **Push**: Запускается при пуше кода в основную ветку
- **Pull Request**: Запускается при открытии pull request на основной ветки
- **Schedule**: Запускается ежедневно в полночь (UTC) для инкрементного обновления модели

## Переменные среды

```yaml
env:
  TRAINING_ITERATIONS: 3
```

Эта переменная позволяет контролировать процесс обучения без изменения файла workflow:

- `TRAINING_ITERATIONS`: Число итераций обучения

## Шаги выполнения

### 1. Environment Setup

```yaml
- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version: '3.12'
    
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -r requirements.txt
```

Эти шаги готовят окружение выполнения:
- Извлекает код репозитория
- Настраивает Python 3.12
- Устанавливает зависимости из requirements.txt

### 2. Настройка Kaggle API

```yaml
- name: Configure Kaggle API
  env:
    KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
    KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}
  run: |
    mkdir -p ~/.kaggle
    echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
```

Этот шаг:
- Создает файл конфигурации API Kaggle, используя секреты репозитория
- Устанавливает соответствующие разрешения на файл
- Позволяет workflow загружать датасеты с Kaggle

### 3. Сохранение состояний модели и данных

```yaml
    - name: Download model state from previous runs
      uses: actions/download-artifact@v4
      with:
        name: model-state
        path: runtime/model/
      continue-on-error: true
    
    - name: Download data state from previous runs
      uses: actions/download-artifact@v4
      with:
        name: data-state
        path: runtime/data/
      continue-on-error: true
```

Этот шаг:
- Пытается получить состояние модели и данных из предыдущих запусков workflow
- Помещает файлы в директорию runtime
- Продолжается, даже если предыдущего состояния не существует

### 4. Инкрементное обучение

```yaml
- name: Run iterative training
  run: |
    for i in $(seq 1 $TRAINING_ITERATIONS); do
      echo "Starting training iteration $i of $TRAINING_ITERATIONS"
      python src/run.py -mode "update"
      echo "Completed training iteration $i"
    done
```
Этот шаг:
- Запускает сценарий обучения для указанного количества итераций
- Обучение происходит на основе предыдущего состояния модели

### 6. Загрузка артефактов

```yaml
- name: Upload model artifacts
	uses: actions/upload-artifact@v4
	with:
	name: model-state
	path: |
		runtime/model/best_model.pkl
	retention-days: 5
	
- name: Upload data state
	uses: actions/upload-artifact@v4
	with:
	name: data-state
	path: |
		runtime/data/data_state.json
	retention-days: 5

- name: Upload training logs
	uses: actions/upload-artifact@v4
	with:
	name: training-logs
	path: |
		runtime/logs/
	retention-days: 5
```
Эти шаги:
- Сохраняют обученную модель и ее состояние для будущих запусков workflow
- Сохраняют логи обучения для анализа
- Делает отчет о работе доступным для скачивания
- Устанавливают 5-дневный период хранения артефактов

## Управление артефактами

В рабочем процессе управляются несколькими типами артефактов:

1. **Состояние модели**:
   - Файлы сериализованных моделей (best_model.pkl)
   - Отслеживание состояния данных (data_state.json)
   - Используется для продолжения обучения в последующих запусках рабочей среды

2. **Логи обучения**:
   - Файлы журналов из каждой итерации
   - Полезны для отладки и мониторинга производительности модели

## Требования и зависимости

Для использования этого рабочего процесса необходимы:

1. Репозиторий GitHub с кодом MLOps-пайплайна
2. Учетные данные API Kaggle, сохраненные как секреты репозитория
4. Файл requirements.txt со всеми необходимыми зависимостями

## Настройка

Можно настроить workflow, изменив:

1. Количество итераций обучения
2. Расписание для инкрементного обучения
3. Добавление дополнительных шагов для специализированных задач
4. Периоды хранения артефактов
5. Уведомления о завершении или сбое рабочего процесса

