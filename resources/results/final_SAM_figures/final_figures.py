import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Variable to place the path of the CSV file
csv_filepath = 'metrics.csv'
# Set style for professional-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Create directory for saving graphs
graph_dir = 'final_SAM_graphs'
if not os.path.exists(graph_dir):
    os.makedirs(graph_dir)
    print(f"Created directory: {graph_dir}")

# Load and clean the CSV file
def load_and_clean_data(filepath):
    # Read raw file and clean inconsistencies
    cleaned_lines = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(',')
            parts = parts[:6]  # Take only first 6 columns
            cleaned_lines.append(','.join(parts) + '\n')
    
    # Save cleaned file temporarily
    temp_filepath = 'temp_cleaned_metrics.csv'
    with open(temp_filepath, 'w') as file:
        file.writelines(cleaned_lines)
    
    # Load with pandas
    df = pd.read_csv(temp_filepath)
    os.remove(temp_filepath)  # Clean up temp file
    return df

# Load data
print("Loading and cleaning data...")
metrics_df = load_and_clean_data(csv_filepath)
print(f"Loaded {len(metrics_df)} rows of data")

# 1. Individual Metric Plots with Enhanced Styling - FIXED LEGEND LOCATIONS
def create_individual_plots(df):
    metrics = [
        ('mean_loss', 'Mean Loss', 'Loss Value', 'upper right'),     # FIXED: changed from 'lower'
        ('precision', 'Precision', 'Precision Score', 'lower right'), # FIXED: changed from 'upper'
        ('recall', 'Recall', 'Recall Score', 'lower right'),        # FIXED: changed from 'upper'
        ('accuracy', 'Accuracy', 'Accuracy Score', 'lower right')   # FIXED: changed from 'upper'
    ]
    
    for metric, title, ylabel, legend_loc in metrics:
        plt.figure(figsize=(12, 8))
        
        for phase in df['phase'].unique():
            phase_data = df[df['phase'] == phase].sort_values('epoch')
            color = '#2E86AB' if phase == 'train' else '#A23B72'
            marker = 'o' if phase == 'train' else 's'
            
            plt.plot(phase_data['epoch'], phase_data[metric], 
                    marker=marker, linewidth=2.5, markersize=8,
                    label=f'{phase.capitalize()}', color=color, alpha=0.8)
        
        plt.title(f'{title} Over Training Epochs', fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontweight='bold')
        plt.ylabel(ylabel, fontweight='bold')
        plt.legend(loc=legend_loc, frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        filename = f'{metric}_over_epochs.png'
        plt.savefig(os.path.join(graph_dir, filename), bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"Saved: {filename}")

# 2. Combined Metrics Dashboard
def create_combined_dashboard(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Training Metrics Dashboard', fontsize=20, fontweight='bold', y=0.95)
    
    metrics = [
        ('mean_loss', 'Mean Loss', axes[0,0], 'Loss'),
        ('precision', 'Precision', axes[0,1], 'Score'),
        ('recall', 'Recall', axes[1,0], 'Score'),
        ('accuracy', 'Accuracy', axes[1,1], 'Score')
    ]
    
    colors = ['#2E86AB', '#A23B72']
    
    for metric, title, ax, ylabel in metrics:
        for i, phase in enumerate(df['phase'].unique()):
            phase_data = df[df['phase'] == phase].sort_values('epoch')
            ax.plot(phase_data['epoch'], phase_data[metric], 
                   marker=['o', 's'][i], linewidth=2, markersize=6,
                   label=phase.capitalize(), color=colors[i], alpha=0.8)
        
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    filename = 'metrics_dashboard.png'
    plt.savefig(os.path.join(graph_dir, filename), bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {filename}")

# 3. Training vs Validation Comparison
def create_train_val_comparison(df):
    if 'val' not in df['phase'].values:
        print("No validation data found, skipping train-val comparison")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training vs Validation Performance Comparison', 
                fontsize=18, fontweight='bold', y=0.95)
    
    metrics = ['mean_loss', 'precision', 'recall', 'accuracy']
    titles = ['Mean Loss', 'Precision', 'Recall', 'Accuracy']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i//2, i%2]
        
        # Plot training data
        train_data = df[df['phase'] == 'train'].sort_values('epoch')
        ax.plot(train_data['epoch'], train_data[metric], 
               'o-', label='Training', color='#2E86AB', alpha=0.7, linewidth=2)
        
        # Plot validation data
        val_data = df[df['phase'] == 'val'].sort_values('epoch')
        ax.plot(val_data['epoch'], val_data[metric], 
               's-', label='Validation', color='#A23B72', linewidth=2.5, markersize=8)
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Value', fontweight='bold')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    filename = 'train_val_comparison.png'
    plt.savefig(os.path.join(graph_dir, filename), bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {filename}")

# 4. Performance Summary Heatmap
def create_performance_heatmap(df):
    # Create summary statistics for validation data (or training if no validation)
    if 'val' in df['phase'].values:
        summary_data = df[df['phase'] == 'val'].groupby('epoch')[['precision', 'recall', 'accuracy']].mean()
    else:
        summary_data = df[df['phase'] == 'train'].groupby('epoch')[['precision', 'recall', 'accuracy']].mean()
    
    plt.figure(figsize=(14, 8))
    
    # Create heatmap
    sns.heatmap(summary_data.T, annot=True, cmap='RdYlBu_r', center=0.5,
                fmt='.3f', linewidths=0.5, cbar_kws={'label': 'Score'})
    
    plt.title('Performance Metrics Heatmap Across Epochs', 
             fontweight='bold', fontsize=16, pad=20)
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Metrics', fontweight='bold')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    filename = 'performance_heatmap.png'
    plt.savefig(os.path.join(graph_dir, filename), bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {filename}")

# 5. Model Improvement Trend Analysis
def create_improvement_analysis(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss improvement
    train_loss = df[df['phase'] == 'train'].sort_values('epoch')
    if len(train_loss) > 1:
        loss_improvement = train_loss['mean_loss'].iloc[0] - train_loss['mean_loss'].iloc[-1]
        
        ax1.plot(train_loss['epoch'], train_loss['mean_loss'], 'o-', 
                color='#E63946', linewidth=2.5, markersize=6)
        ax1.fill_between(train_loss['epoch'], train_loss['mean_loss'], 
                        alpha=0.3, color='#E63946')
        ax1.set_title(f'Loss Reduction Trend\n(Improvement: {loss_improvement:.4f})', 
                     fontweight='bold')
        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Mean Loss', fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    # Accuracy improvement
    train_acc = df[df['phase'] == 'train'].sort_values('epoch')
    if len(train_acc) > 1:
        acc_improvement = train_acc['accuracy'].iloc[-1] - train_acc['accuracy'].iloc[0]
        
        ax2.plot(train_acc['epoch'], train_acc['accuracy'], 'o-', 
                color='#2A9D8F', linewidth=2.5, markersize=6)
        ax2.fill_between(train_acc['epoch'], train_acc['accuracy'], 
                        alpha=0.3, color='#2A9D8F')
        ax2.set_title(f'Accuracy Improvement Trend\n(Improvement: {acc_improvement:.4f})', 
                     fontweight='bold')
        ax2.set_xlabel('Epoch', fontweight='bold')
        ax2.set_ylabel('Accuracy', fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = 'improvement_analysis.png'
    plt.savefig(os.path.join(graph_dir, filename), bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {filename}")

# 6. Final Performance Summary Bar Chart
def create_final_summary(df):
    # Get final epoch performance
    final_train = df[df['phase'] == 'train'].iloc[-1]
    
    if 'val' in df['phase'].values:
        final_val = df[df['phase'] == 'val'].iloc[-1]
        phases = ['Training', 'Validation']
        precision_scores = [final_train['precision'], final_val['precision']]
        recall_scores = [final_train['recall'], final_val['recall']]
        accuracy_scores = [final_train['accuracy'], final_val['accuracy']]
    else:
        phases = ['Training']
        precision_scores = [final_train['precision']]
        recall_scores = [final_train['recall']]
        accuracy_scores = [final_train['accuracy']]
    
    x = np.arange(len(phases))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars1 = ax.bar(x - width, precision_scores, width, label='Precision', 
                   color='#264653', alpha=0.8)
    bars2 = ax.bar(x, recall_scores, width, label='Recall', 
                   color='#2A9D8F', alpha=0.8)
    bars3 = ax.bar(x + width, accuracy_scores, width, label='Accuracy', 
                   color='#E9C46A', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Phase', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Final Model Performance Summary', fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    filename = 'final_performance_summary.png'
    plt.savefig(os.path.join(graph_dir, filename), bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {filename}")

# Generate all visualizations
print("\nGenerating visualizations...")
create_individual_plots(metrics_df)
create_combined_dashboard(metrics_df)
create_train_val_comparison(metrics_df)
create_performance_heatmap(metrics_df)
create_improvement_analysis(metrics_df)
create_final_summary(metrics_df)

print(f"\n✅ All visualizations saved successfully in '{graph_dir}' directory!")
print(f"📊 Generated {len(os.listdir(graph_dir))} professional graphs for your paper.")

# Display summary of generated files
print("\n📁 Generated files:")
for file in sorted(os.listdir(graph_dir)):
    if file.endswith('.png'):
        print(f"   • {file}")

