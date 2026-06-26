import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate():
    # Read the data
    df = pd.read_csv('success_results.csv')
    
    # Set seaborn style for academic look
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. Accuracy Chart (Peak Semantic Score)
    plt.figure(figsize=(8, 6))
    ax1 = sns.barplot(x="Method", y="Peak_Relevance_Score", data=df, errorbar=None)
    plt.title("Ablation Study: Retrieval Accuracy in Sparse Action Scenarios", fontsize=14)
    plt.ylabel("Peak Semantic Score (CLIP)", fontsize=12)
    plt.xlabel("")
    # Add values on top
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=300)
    print("Saved accuracy_comparison.png")
    plt.close()
    
    # 2. Resource Chart (Peak VRAM & Latency)
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    color1 = 'tab:blue'
    ax1.set_title("Edge Hardware Constraints: Peak VRAM vs Latency", fontsize=14)
    ax1.set_ylabel('Peak VRAM (MB)', color=color1, fontsize=12)
    ax1.bar(df['Method'], df['Peak_VRAM_MB'], color=color1, alpha=0.7, width=0.4, align='center', label='VRAM')
    ax1.tick_params(axis='y', labelcolor=color1)
    # Target line for 4GB (4096 MB) limits
    ax1.axhline(y=4096, color='red', linestyle='--', label='Hardware Limit (GTX 1650 4GB)')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  
    color2 = 'tab:red'
    ax2.set_ylabel('Latency (Seconds)', color=color2, fontsize=12)  
    ax2.plot(df['Method'], df['Latency_s'], color=color2, marker='o', linewidth=2, markersize=8, label='Latency')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim([0, 2]) # Scale latency
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('resource_comparison.png', dpi=300)
    print("Saved resource_comparison.png")
    plt.close()

if __name__ == "__main__":
    generate()
