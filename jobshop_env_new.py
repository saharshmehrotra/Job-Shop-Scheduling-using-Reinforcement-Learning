import gym
from gym import spaces
import numpy as np
import json
import plotly.graph_objects as go
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
        
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros((self.num_jobs,), dtype=np.float32)
        for job_id in range(self.num_jobs):
            if self.job_task_index[job_id] < len(self.jobs[job_id]):
                machine_id, _ = self.jobs[job_id][self.job_task_index[job_id]]
                obs[job_id] = self.machine_available_time[machine_id]
            else:
                obs[job_id] = self.job_available_time[job_id]
        return obs

    def step(self, action):
        job_id = int(action)

        if self.job_task_index[job_id] >= len(self.jobs[job_id]):
            return self._get_obs(), -1.0, False, {}

        machine_id, duration = self.jobs[job_id][self.job_task_index[job_id]]
        start_time = max(self.job_available_time[job_id], self.machine_available_time[machine_id])
        end_time = start_time + duration

        self.machine_available_time[machine_id] = end_time
        self.job_available_time[job_id] = end_time
        self.job_schedule[job_id].append({
            'Machine': machine_id,
            'Start': start_time,
            'Finish': end_time
        })
        self.job_task_index[job_id] += 1

        done = all(idx >= len(self.jobs[j]) for j, idx in enumerate(self.job_task_index))
        reward = -max(self.job_available_time) if done else 0

        return self._get_obs(), reward, done, {}

    def render_gantt_numerical(self):
        records = []
        for job_id, tasks in self.job_schedule.items():
            for task in tasks:
                records.append({
                    "Job": f"Job {job_id}",
                    "Machine": f"Machine {int(task['Machine'])}",
                    "Start": int(task['Start']),
                    "Duration": int(task['Finish']) - int(task['Start'])
                })

        fig = go.Figure()

        for rec in records:
            fig.add_trace(go.Bar(
                x=[rec["Duration"]],
                y=[rec["Job"]],
                base=rec["Start"],
                orientation='h',
                name=rec["Machine"],
                hoverinfo="x+name",
                marker=dict(line=dict(width=1))
            ))

        fig.update_layout(
            title="Gantt Chart (Numerical Time)",
            xaxis_title="Time",
            yaxis_title="Job",
            barmode='stack',
            height=600,
            plot_bgcolor='rgba(245, 245, 255, 1)',
            legend_title="Machine",
            yaxis=dict(autorange="reversed")
        )

        fig.show()
