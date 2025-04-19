import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV data
data = np.loadtxt('/Users/kunwarnir/projects/envPoisoning/ScalingEnvironmentPoisoning/metrics/1744946116.664236/accuracy_buffer.csv', delimiter=',')
# data = np.loadtxt('/Users/kunwarnir/projects/envPoisoning/master/ScalingEnvironmentPoisoning/accuracy_buffer.csv', delimiter=',')

# Convert to DataFrame
df = pd.DataFrame(data, columns=['episode', 'timestep', 'accuracy'])

# Calculate mean accuracy per episode
episode_accuracy = df.groupby('episode')['accuracy'].mean().reset_index()

# Remove outliers (points beyond 3 standard deviations from the mean)
mean = episode_accuracy['accuracy'].mean()
std = episode_accuracy['accuracy'].std()
episode_accuracy = episode_accuracy[
    (episode_accuracy['accuracy'] >= mean - 3 * std) &
    (episode_accuracy['accuracy'] <= mean + 3 * std)
]

# Apply smoothing using rolling average
window_size = 5  # Adjust this value to control smoothing amount
episode_accuracy['smoothed_accuracy'] = episode_accuracy['accuracy'].rolling(
    window=window_size, center=True, min_periods=1).mean()

# Create the plot
plt.figure(figsize=(12, 6))

# Plot with matplotlib
plt.plot(episode_accuracy['episode'], episode_accuracy['smoothed_accuracy'],
         linewidth=2, color='blue')

# Customize the plot
plt.title('Average Accuracy per Episode', fontsize=14)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('accuracy_per_episode3.png', dpi=300, bbox_inches='tight')
plt.close()
