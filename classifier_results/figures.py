import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set the style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Define the path variable for the csv file
csv_file_path = "training_results.csv"

# Read the csv file
results_df = pd.read_csv(csv_file_path)

# Create directory if it doesn't exist
output_dir = "final_SAM_graphs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# Set up the plotting parameters for publication-quality figures
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# 1. Training Loss Progression
plt.figure(figsize=(12, 8))
plt.plot(results_df['epoch'], results_df['train_loss'], 
         marker='o', linewidth=3, markersize=8, color='#2E86AB', 
         markerfacecolor='white', markeredgewidth=2, markeredgecolor='#2E86AB')
plt.title('Training Loss Progression', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Training Loss', fontsize=14)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(range(1, len(results_df) + 1, 2))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_loss_progression.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 2. Training vs Validation Accuracy Comparison
plt.figure(figsize=(12, 8))
plt.plot(results_df['epoch'], results_df['train_acc'], 
         marker='o', linewidth=3, markersize=8, color='#A23B72', 
         label='Training Accuracy', markerfacecolor='white', 
         markeredgewidth=2, markeredgecolor='#A23B72')
plt.plot(results_df['epoch'], results_df['val_acc'], 
         marker='s', linewidth=3, markersize=8, color='#F18F01', 
         label='Validation Accuracy', markerfacecolor='white', 
         markeredgewidth=2, markeredgecolor='#F18F01')
plt.title('Training vs Validation Accuracy', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend(loc='lower right', fontsize=14)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(range(1, len(results_df) + 1, 2))
plt.ylim(0.8, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 3. Learning Rate Schedule
plt.figure(figsize=(12, 8))
plt.plot(results_df['epoch'], results_df['lr'], 
         marker='D', linewidth=3, markersize=10, color='#C73E1D', 
         markerfacecolor='white', markeredgewidth=2, markeredgecolor='#C73E1D')
plt.title('Learning Rate Schedule', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Learning Rate', fontsize=14)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(range(1, len(results_df) + 1, 2))
plt.yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'learning_rate_schedule.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 4. Comprehensive Training Overview (2x2 subplot)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('SAM Model Training Overview', fontsize=20, fontweight='bold')

# Training Loss
axes[0, 0].plot(results_df['epoch'], results_df['train_loss'], 
                marker='o', linewidth=3, color='#2E86AB')
axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].grid(True, alpha=0.3)

# Accuracy Comparison
axes[0, 1].plot(results_df['epoch'], results_df['train_acc'], 
                marker='o', linewidth=3, color='#A23B72', label='Train')
axes[0, 1].plot(results_df['epoch'], results_df['val_acc'], 
                marker='s', linewidth=3, color='#F18F01', label='Validation')
axes[0, 1].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Learning Rate
axes[1, 0].plot(results_df['epoch'], results_df['lr'], 
                marker='D', linewidth=3, color='#C73E1D')
axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Learning Rate')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, alpha=0.3)

# Generalization Gap (Train Acc - Val Acc)
gap = results_df['train_acc'] - results_df['val_acc']
axes[1, 1].plot(results_df['epoch'], gap, 
                marker='^', linewidth=3, color='#8B5A3C')
axes[1, 1].set_title('Generalization Gap', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Train Acc - Val Acc')
axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_overview.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 5. Performance Metrics Summary
plt.figure(figsize=(14, 8))
epochs = results_df['epoch']
metrics = ['train_acc', 'val_acc']
colors = ['#A23B72', '#F18F01']
markers = ['o', 's']

for i, metric in enumerate(metrics):
    plt.plot(epochs, results_df[metric], 
             marker=markers[i], linewidth=3, markersize=8, 
             color=colors[i], label=metric.replace('_', ' ').title(),
             markerfacecolor='white', markeredgewidth=2, 
             markeredgecolor=colors[i])

# Add annotations for best performance
best_train_idx = results_df['train_acc'].idxmax()
best_val_idx = results_df['val_acc'].idxmax()

plt.annotate(f'Best Train: {results_df.iloc[best_train_idx]["train_acc"]:.4f}',
             xy=(results_df.iloc[best_train_idx]['epoch'], results_df.iloc[best_train_idx]['train_acc']),
             xytext=(10, 10), textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.annotate(f'Best Val: {results_df.iloc[best_val_idx]["val_acc"]:.4f}',
             xy=(results_df.iloc[best_val_idx]['epoch'], results_df.iloc[best_val_idx]['val_acc']),
             xytext=(10, -20), textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.title('Model Performance Metrics with Best Scores', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend(loc='lower right', fontsize=14)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(range(1, len(results_df) + 1, 2))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'performance_metrics_annotated.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 6. Training Dynamics Heatmap
plt.figure(figsize=(12, 8))
metrics_matrix = results_df[['train_loss', 'train_acc', 'val_acc', 'lr']].T
epochs_labels = [f'E{i}' for i in results_df['epoch']]

# Normalize each metric to 0-1 scale for better visualization
metrics_normalized = metrics_matrix.copy()
for i in range(len(metrics_normalized)):
    row = metrics_normalized.iloc[i]
    metrics_normalized.iloc[i] = (row - row.min()) / (row.max() - row.min())

sns.heatmap(metrics_normalized, 
            xticklabels=epochs_labels, 
            yticklabels=['Train Loss', 'Train Acc', 'Val Acc', 'Learning Rate'],
            cmap='RdYlBu_r', center=0.5, annot=False, 
            cbar_kws={'label': 'Normalized Value'})
plt.title('Training Dynamics Heatmap (Normalized)', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Metrics', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_dynamics_heatmap.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 7. Model Convergence Analysis
plt.figure(figsize=(12, 8))
# Calculate moving averages for smoother trend visualization
window = 3
train_loss_smooth = results_df['train_loss'].rolling(window=window, center=True).mean()
train_acc_smooth = results_df['train_acc'].rolling(window=window, center=True).mean()
val_acc_smooth = results_df['val_acc'].rolling(window=window, center=True).mean()

plt.subplot(2, 1, 1)
plt.plot(results_df['epoch'], results_df['train_loss'], 'o-', alpha=0.6, color='#2E86AB', label='Original')
plt.plot(results_df['epoch'], train_loss_smooth, '-', linewidth=3, color='#1B5E82', label='Smoothed Trend')
plt.title('Training Loss Convergence', fontsize=14, fontweight='bold')
plt.ylabel('Training Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(results_df['epoch'], results_df['train_acc'], 'o-', alpha=0.6, color='#A23B72', label='Train (Original)')
plt.plot(results_df['epoch'], results_df['val_acc'], 's-', alpha=0.6, color='#F18F01', label='Val (Original)')
plt.plot(results_df['epoch'], train_acc_smooth, '-', linewidth=3, color='#7A2B54', label='Train (Smoothed)')
plt.plot(results_df['epoch'], val_acc_smooth, '-', linewidth=3, color='#C17001', label='Val (Smoothed)')
plt.title('Accuracy Convergence', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.suptitle('Model Convergence Analysis', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'convergence_analysis.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Print summary statistics
print("=== SAM Model Training Summary ===")
print(f"Total Epochs: {len(results_df)}")
print(f"Final Training Loss: {results_df['train_loss'].iloc[-1]:.4f}")
print(f"Best Training Accuracy: {results_df['train_acc'].max():.4f} (Epoch {results_df['train_acc'].idxmax() + 1})")
print(f"Best Validation Accuracy: {results_df['val_acc'].max():.4f} (Epoch {results_df['val_acc'].idxmax() + 1})")
print(f"Final Generalization Gap: {(results_df['train_acc'].iloc[-1] - results_df['val_acc'].iloc[-1]):.4f}")
print(f"\nAll graphs saved successfully in '{output_dir}' directory!")

# List all generated files
generated_files = [
    'training_loss_progression.png',
    'accuracy_comparison.png', 
    'learning_rate_schedule.png',
    'training_overview.png',
    'performance_metrics_annotated.png',
    'training_dynamics_heatmap.png',
    'convergence_analysis.png'
]

print(f"\nGenerated {len(generated_files)} visualization files:")
for file in generated_files:
    print(f"  - {file}")

