import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from models.schedule import MaskingSchedule

# make folder
os.makedirs('visualisations/schedule', exist_ok=True)

min_t = 0.01
batch_size = 200
time_steps = 100

colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
order = [0, 3, 1, 4, 2, 5]

def plot_schedules():
    # different schedules
    schedules = {
        'cosine': MaskingSchedule('cosine'),
        'linear': MaskingSchedule('linear'), 
        'poly2': MaskingSchedule('poly2'),
        'poly0.5': MaskingSchedule('poly0.5'),
        'sigmoid0.7': MaskingSchedule('sigmoid_3.0_3.0_0.7'),
        'sigmoid1.1': MaskingSchedule('sigmoid_3.0_3.0_1.1'),
    }
    
    # time steps
    t_tensor = torch.linspace(1/time_steps, 1, time_steps)
    s_tensor = t_tensor - 1.0 / time_steps
    
    # plot masking probability
    plt.figure()
    for name, schedule in schedules.items():
        alpha_t = schedule.alpha(t_tensor)
        masking_prob = 1 - alpha_t
        plt.plot(t_tensor, masking_prob, label=name, linewidth=1)
    
    plt.xlabel('t')
    plt.ylabel('Masking Prob')
    plt.title('Masking Probability')
    plt.legend()
    plt.grid()
    plt.savefig('visualisations/schedule/masking_prob.png')
    plt.show()
    
    # plot unmasking probability 
    # in discrete case, we don't use time clamp: min_t = 1/time_steps
    plt.figure()
    for name, schedule in schedules.items():
        alpha_t = schedule.alpha(t_tensor)
        alpha_s = schedule.alpha(s_tensor)
        unmask_prob = (alpha_s - alpha_t) / (1 - alpha_t)
        plt.plot(t_tensor, unmask_prob, label=name, linewidth=1)
    
    plt.xlabel('t')
    plt.ylabel('Unmask Prob')
    plt.title('Unmasking Probability')
    plt.legend()
    plt.grid()
    plt.savefig('visualisations/schedule/unmasking_prob.png')
    plt.show()

    # plot cumulative unmasking probability  
    plt.figure()
    for i, (name, schedule) in enumerate(schedules.items()):
        alpha_t = schedule.alpha(t_tensor)
        plt.plot(t_tensor, alpha_t, label=name, color=colors[order[i]], linewidth=1)
    plt.xlabel('t')
    plt.ylabel('Cumulative Unmask Prob')
    plt.title('Cumulative Unmasking Probability')
    plt.legend()
    plt.grid()
    plt.savefig('visualisations/schedule/cumulative_unmasking_prob.png')
    plt.show()
    
    # plot loss weights continuous time
    plt.figure()
    for i, (name, schedule) in enumerate(schedules.items()):
        # use time clamp for visual purposes
        t_tensor = torch.linspace(min_t, 1, time_steps)
        # continuous weights
        loss_weights = - schedule.dgamma_times_alpha(t_tensor)
        plt.plot(t_tensor, loss_weights, label=name, linewidth=1, color=colors[order[i]])
    
    plt.xlabel('t')
    plt.ylabel('Loss Weight')
    plt.title('Loss Weights')
    plt.ylim(0.0, 80)
    plt.legend()
    plt.grid()
    plt.savefig('visualisations/schedule/loss_weights.png')
    plt.show()


    # plot batch examples continuous time
    # random timesteps like in training
    # use time clamp for visual purposes
    random_t = torch.rand(batch_size) * (1.0 - min_t) + min_t
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, (name, schedule) in enumerate(schedules.items()):
        ax = axes[order[i]]
        
        # calculate loss weights
        weights = - schedule.dgamma_times_alpha(random_t)
        expected_masked = 1 - schedule.alpha(random_t)  
        weight_expected_masked = weights * expected_masked
        ax.bar(range(batch_size), sorted(weights, reverse=True), alpha=0.7, color='blue')
        mean_weight = weights.mean()
        weight_total_mean = weight_expected_masked.mean()
        max_weight = weights.max()
        ax.axhline(mean_weight, color=colors[i], linestyle='--', 
                label=f'Mean = {mean_weight:.3f}', linewidth=1)
        
        ax.set_xlabel('Batch Sample (sorted by t)', fontsize=14)
        ax.set_ylabel('Loss Weight', fontsize=14)
        ax.set_title(name, fontweight='bold', fontsize=16)
        ax.legend([f'Mean = {mean_weight:.3f}', f'Max = {max_weight:.3f}', f'Expected Total = {weight_total_mean:.3f}'], 
                 loc='upper left', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualisations/schedule/batch_examples.png', dpi=300, bbox_inches='tight')
    plt.show()

 
    # plot effect of time_clamp using cosine schedule
    min_t_vals = [0.0001, 0.001, 0.01, 0.02]
    schedule = schedules['cosine']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, min_t_val in enumerate(min_t_vals):
        ax = axes[i]
        random_t = torch.rand(batch_size) * (1.0 - min_t_val) + min_t_val
        # calculate loss weights
        weights = -schedule.dgamma_times_alpha(random_t)
        expected_masked = 1 - schedule.alpha(random_t)  # B, 1
        weight_expected_masked = weights * expected_masked
        ax.bar(range(batch_size), sorted(weights, reverse=True), alpha=0.7, color='blue')
        mean_weight = weights.mean()
        weight_total_mean = weight_expected_masked.mean()
        max_weight = weights.max()
        ax.axhline(mean_weight, color=colors[i], linestyle='--', 
                label=f'Mean = {mean_weight:.3f}', linewidth=1)
        
        ax.set_xlabel('Batch Sample (sorted by t)', fontsize=14)
        ax.set_ylabel('Loss Weight', fontsize=14)
        ax.set_title('min_t = ' + str(min_t_val), fontweight='bold', fontsize=16)
        ax.legend([f'Mean = {mean_weight:.3f}', f'Max = {max_weight:.3f}', f'Expected Total = {weight_total_mean:.3f}'], 
                 loc='upper left', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualisations/schedule/time_clamp.png', dpi=300, bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    plot_schedules()
    print("Done!")
