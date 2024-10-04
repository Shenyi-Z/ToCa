import json
import numpy as np
import matplotlib.pyplot as plt

with open('timing_info.json', 'r') as f:
    data = json.load(f)

attn_times = []
cross_attn_times = []
mlp_times = []
block_times = []

for entry in data:
    timing_info = entry['timing_info']
    attn_times.extend(timing_info['attn_time'])
    cross_attn_times.extend(timing_info['cross_attn_time'])
    mlp_times.extend(timing_info['mlp_time'])
    block_times.extend(timing_info['block_time'])

average_attn_time = np.mean(attn_times)
average_cross_attn_time = np.mean(cross_attn_times)
average_mlp_time = np.mean(mlp_times)
average_block_time = np.mean(block_times)

print(f"Average Attention Time: {average_attn_time:.4f} ms")
print(f"Average Cross Attention Time: {average_cross_attn_time:.4f} ms")
print(f"Average MLP Time: {average_mlp_time:.4f} ms")
print(f"Average Block Time: {average_block_time:.4f} ms")

labels = ['Attention', 'Cross Attention', 'MLP', 'Block']
avg_times = [average_attn_time, average_cross_attn_time, average_mlp_time, average_block_time]

plt.bar(labels, avg_times, color=['blue', 'green', 'red', 'orange'])
plt.ylabel('Average Time (ms)')
plt.title('Average Time per Module')

plt.savefig('module_average_times.png')
