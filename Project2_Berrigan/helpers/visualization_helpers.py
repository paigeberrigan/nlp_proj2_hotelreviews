# helpers/visualization_helpers.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# custom palatte matching the tripadvisor colors 
custom_palette = [
    '#ffffff',  
    '#858585',   
    '#32d99c',  
    '#66aa8d', 
    '#020c09', 
]

def plot_class_distribution(ratings_series, title="Class Distribution", save_path=None):
    # Count the occurrences of each rating
    class_counts = ratings_series.value_counts(sort=False).sort_index()
    
    # if more or less colors in paltte raise warning
    if len(class_counts) > len(custom_palette):
        raise ValueError("Not enough colors in the custom_palette to match the number of unique classes.")
    plt.figure(figsize=(8, 5))
    
    # Plot w seaborn
    sns.barplot(
        x=class_counts.index,
        y=class_counts.values,
        edgecolor='black',
        palette=custom_palette[:len(class_counts)],  # only use first 5 colors
    )

    # add plot visuals
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Rating", fontsize=14)
    plt.ylabel("Number of Reviews", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # save to path
    if save_path:
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
    
    heatmap_palette = [ # similar to custom but is in strength order
    '#32b99c',
    '#28947c',
    '#195c4e',
    '#144a3e',
    '#144a3e',
    ]
    
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
    
    # add plot visuals
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

# print the classification stats for each embedding type
def print_classification_report(y_true, y_pred, embedding_type):
    report = classification_report(y_true, y_pred)
    print(f"\nClassification Report for {embedding_type} Embedding:\n")
    print(report)
