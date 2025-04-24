import gym
from gym import spaces
import numpy as np
import json
import plotly.express as px
import pandas as pd

class JobShopEnv(gym.Env):
    def __init__(self, instance_path):
        super(JobShopEnv, self).__init__()
        self.jobs = self.load_jobs_from_file(instance_path)
        self.num_jobs = len(self.jobs)
        self.num_machines = max(max(task[0] for task in job) for job in self.jobs) + 1

        self.action_space = spaces.Discrete(self.num_jobs)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.num_jobs,), dtype=np.float32)

        self.reset()

    def load_jobs_from_file(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def reset(self):
        self.current_time = 0
        self.job_task_index = [0] * self.num_jobs
        self.machine_available_time = [0] * self.num_machines
        self.job_available_time = [0] * self.num_jobs
        self.job_schedule = {job_id: [] for job_id in range(self.num_jobs)}
        
        # Make sure initial observation reflects job availability
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros((self.num_jobs,), dtype=np.float32)
        for job_id in range(self.num_jobs):
            if self.job_task_index[job_id] < len(self.jobs[job_id]):
                machine_id, _ = self.jobs[job_id][self.job_task_index[job_id]]
                obs[job_id] = self.machine_available_time[machine_id]
            else:
                obs[job_id] = self.job_available_time[job_id]  # Track job progress by the last machine time
        return obs

    def step(self, action):
        job_id = int(action)

        # Check if the job has no more tasks to process
        if self.job_task_index[job_id] >= len(self.jobs[job_id]):
            return self._get_obs(), -1.0, False, {}

        machine_id, duration = self.jobs[job_id][self.job_task_index[job_id]]
        start_time = max(self.job_available_time[job_id], self.machine_available_time[machine_id])
        end_time = start_time + duration

        # Update the machine and job available times
        self.machine_available_time[machine_id] = end_time
        self.job_available_time[job_id] = end_time
        self.job_schedule[job_id].append({
            'Machine': machine_id,
            'Start': start_time,
            'Finish': end_time
        })
        self.job_task_index[job_id] += 1

        # Check if all jobs are complete
        done = all(idx >= len(self.jobs[j]) for j, idx in enumerate(self.job_task_index))
        reward = -max(self.job_available_time) if done else 0

        return self._get_obs(), reward, done, {}

    
    def render_gantt_plotly(self):
        import plotly.graph_objects as go

        fig = go.Figure()

        machine_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
            "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
        ]

        for job_id, tasks in self.job_schedule.items():
            for task in tasks:
                fig.add_trace(go.Bar(
                    x=[task['Finish'] - task['Start']],
                    y=[f"Job {job_id}"],
                    base=task['Start'],
                    orientation='h',
                    name=f"Machine {task['Machine']}",
                    marker=dict(color=machine_colors[task['Machine'] % len(machine_colors)]),
                    hovertemplate=(
                        f"Job {job_id}<br>"
                        f"Machine {task['Machine']}<br>"
                        f"Start: {task['Start']}<br>"
                        f"Finish: {task['Finish']}<extra></extra>"
                    ),
                    showlegend=False  # We'll add custom legend
                ))

        # Add machine legend separately
        unique_machines = sorted({task['Machine'] for tasks in self.job_schedule.values() for task in tasks})
        for m in unique_machines:
            fig.add_trace(go.Bar(
                x=[None], y=[None], name=f"Machine {m}",
                marker=dict(color=machine_colors[m % len(machine_colors)])
            ))

        fig.update_layout(
            title="Gantt Chart",
            barmode='stack',
            xaxis_title="Time",
            yaxis_title="Job",
            yaxis=dict(autorange="reversed"),  # Jobs from top to bottom
            plot_bgcolor='rgba(245, 245, 255, 1)',
            bargap=0.3,
            height=600,
            legend_title="Machine"
        )

        fig.show()


