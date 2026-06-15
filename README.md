# Drone Navigation RL

Лабораторная работа по обучению с подкреплением.

Реализованы три агента:

* Q-Learning (рациональный агент);
* Prospect Theory Agent (поведенческий агент);
* Risk-Sensitive Agent (риск-чувствительный агент).

## Запуск

```bash
pip install numpy scipy matplotlib
python main.py
```

## Метрики

* Success (%)
* Collision Rate (%)
* Mean Success Time
* Mean Energy Consumption

## Визуализации

* Learning Curve
* CDF Success Time
* Final Battery Heatmap
* Trajectories

Результаты сохраняются в папку `outputs/`.
