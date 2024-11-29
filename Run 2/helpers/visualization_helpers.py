# helpers/visualization_helpers.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# custom palatte matching the tripadvisor colors 
custom_palette = [
    '#32d99c',  # Bright turquoise green
    '#020c09',  # Very dark green
    '#ffffff',  # Pure white
    '#66aa8d',  # Soft sage green
    '#858585'   # Medium gray
]

def plot_class_distribution(ratings_series, title="Class Distribution", save_path=None):
    class_counts = ratings_series.value_counts(sort=False).sort_index()
    plt.figure(figsize=(8, 5))
    palette = custom_palette[:len(class_counts)] # use custom pallate

    
    sns.barplot(
        x=class_counts.index, 
        y=class_counts.values, 
        palette=palette,
        edgecolor='black'
    )
    
    plt.title(title, fontsize=16, fontweight='bold',palette=custom_palette)
    plt.xlabel("Rating", fontsize=14)
    plt.ylabel("Number of Reviews", fontsize=14)
    plt.xticks(ticks=range(len(class_counts)), labels=class_counts.index, fontsize=12)
    plt.yticks(fontsize=12)
    
    if save_path:
        # Ensure the directory exists before saving
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, title="Confusion Matrix", save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8,6))
    
    heatmap_palette = custom_palette[:5]  # palette for the heatmap - 5 colors only
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap=sns.color_palette(heatmap_palette),
        xticklabels=classes, 
        yticklabels=classes, 
        linewidths=.5, 
        linecolor='gray'
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("Actual", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    if save_path: # make sure directory exists before saving
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.tight_layout()
    plt.show()

def display_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred)
    print("Classification Report:\n", report)
