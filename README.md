# Job Shop Scheduling using Reinforcement Learning for Optimization of Workload Allocation

## Overview

This project explores the application of **Reinforcement Learning (RL)** to solve the classic **Job-Shop Scheduling Problem (JSSP)**—an NP-hard optimization task often found in manufacturing and operations management. The goal is to **minimize the makespan** and **improve machine utilization** using intelligent agents trained via the **Proximal Policy Optimization (PPO)** algorithm.

A **custom Gym environment** is developed that simulates a realistic manufacturing setup and visualized scheduling outcomes using **Gantt charts**.

---

## Features

-  Custom OpenAI Gym environment: `JobShopEnv`
-  Support for structured JSON job instance files
-  Visualization via **Matplotlib, Seaborn and Plotly Gantt Charts**
-  Implemented PPO and DQN agents
- Performance Metrics (Rewards, Episode Length, Value Loss, etc.)
- Results validated across varying scales: 5x2, 10x5, 20x10, 50x15 (Jobs x Machines)

---

## Problem Statement

**Objective**: Minimize total makespan in a job-shop scheduling system using RL-based agents.
  
Each job comprises sequential tasks on specific machines. The agent selects the next job to schedule, considering current machine availability and job progress.

---

## Markov Decision Process (MDP) Formulation

| Component | Description |
|----------|-------------|
| **States (S)** | Vector showing earliest time each job's next task can start |
| **Actions (A)** | Selecting a job whose next task is to be scheduled |
| **Reward (R)** | `-makespan` at episode end, `0` for valid moves, `-1` for invalid |
| **Discount Factor (γ)** | `0.99` to favor long-term performance |

---

## Environment Description

- Jobs loaded from a JSON file with machine-task pairs
- Simulates:
  - Machine availability
  - Job progression
  - Real-time scheduling constraints
- Outputs a state vector as observation
- Gantt chart rendering for visual task allocation

---

## RL Algorithms Used

### PPO (Proximal Policy Optimization)
- On-policy, policy-gradient method
- Chosen for stability and sample efficiency
- Uses MLP policy from Stable-Baselines3
- Trained on various instance sizes

### DQN (Deep Q-Network)
- Off-policy, value-based method
- Used for performance comparison in smaller state spaces

---

## Performance Metrics (PPO)

![image](https://github.com/user-attachments/assets/a70ebc22-452d-4781-92db-23e76f1ba5e6)

| Metric | Observation |
|--------|-------------|
| `ep_rew_mean` | Improves over time (more negative to less negative) |
| `ep_len_mean` | Decreases as policy becomes more efficient |
| `value_loss` | Drops and stabilizes—indicating convergence |
| `explained_variance` | Increases, showing better value predictions |

---

## Visualization

A custom Gantt chart is generated using Matplotlib & Seaborn:

```python
env.render_gantt_numerical()
```

![image](https://github.com/user-attachments/assets/f39d90e8-8e8c-4ebf-82ea-eb0277a8756b)
![image](https://github.com/user-attachments/assets/d94e5c79-57ac-47dd-933c-e46d8c46aab9)
![image](https://github.com/user-attachments/assets/f0f4a226-8339-4038-a4f2-0504d18c2a59)
![image](https://github.com/user-attachments/assets/94933cc4-c055-4952-b428-5c39cca73b55)
![image](https://github.com/user-attachments/assets/d73a13ba-c454-49b3-a377-f91633f9bea3)
![image](https://github.com/user-attachments/assets/edc27074-b4d4-4cb4-bde9-1cf13dd87ca1)


Displays:
- Task allocations per job
- Machine-wise scheduling
- Time span per task

---
## Future Scope

- Multi-agent scheduling environments
- Online/dynamic job arrival simulation
- Integration with industrial IoT systems for real-time deployment
