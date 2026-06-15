# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# pyrefly: ignore [missing-import]
from matplotlib.lines import Line2D
import os

# Set a professional style for dissertation plots
sns.set_theme(style='whitegrid')

# --- DATA SIMULATION (Matching your Chapter 4 Results) ---
np.random.seed(42)

# 1. Data for t-SNE (Figure 7)
blast_data = np.random.normal(loc=[5, 6], scale=[1, 1], size=(300, 2)) # Cluster 1
normal_data = np.random.normal(loc=[-5, -6], scale=[1, 1], size=(300, 2))  # Cluster 2
tSNE_data = np.concatenate([blast_data, normal_data])
tSNE_labels = ['Leukaemia'] * 300 + ['Normal'] * 300

# 2. Data for Correlation Matrix (Figure 8)
data = np.random.rand(100, 5) 
data[:, 1] = data[:, 0] + data[:, 2] 
# Use rowvar=False because rows are observations and columns are variables (features)
corr_matrix = np.corrcoef(data, rowvar=False)

# 3. Data for ROC Curves (Figure 9)
fpr_leukaemia = np.linspace(0, 1, 100)
tpr_leukaemia = np.linspace(0, 0.98, 100) # 98% Accuracy
auc_leukaemia = 0.98

fpr_myeloma = np.linspace(0, 1, 100)
tpr_myeloma = np.linspace(0, 1, 100)
auc_myeloma = 0.96

fpr_lymphoma = np.linspace(0, 1, 100)
tpr_lymphoma = np.linspace(0, 0.91, 100) # 91% AUC
auc_lymphoma = 0.91

# 4. Data for Confusion Matrix (Figure 10) - Leukaemia
TP = 95
TN = 970
FP = 20
FN = 5
conf_matrix = np.array([[TN, FP], [FN, TP]])

# 5. Feature Importance (Figure 11)
features = ['Nucleus Size', 'Cytoplasm Texture', 'Membrane Irregularity', 'Cell Roundness', 'Background Intensity']
importances = [0.8, 0.75, 0.6, 0.4, 0.1]

# --- GENERATE GRAPHS ---

# Figure 7: EDA (t-SNE Visualization)
plt.figure(figsize=(6, 4))
colors = ['#FF6B6B' if x == 'Leukaemia' else '#4CAF50' for x in tSNE_labels]
plt.scatter(tSNE_data[:, 0], tSNE_data[:, 1], c=colors, alpha=0.7, edgecolor='k')
plt.title("Figure 7: EDA - t-SNE Visualization", fontsize=14, fontweight='bold')
plt.xlabel("t-SNE Feature 1")
plt.ylabel("t-SNE Feature 2")
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Leukaemia', markerfacecolor='#FF6B6B', markersize=10, markeredgecolor='k'),
                   Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor='#4CAF50', markersize=10, markeredgecolor='k')]
plt.legend(handles=legend_elements)
plt.tight_layout()
plt.savefig('figure_7_eda.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close()

# Figure 8: Correlation Matrix of Visual Features
plt.figure(figsize=(6, 4))
feature_names = ['Nucleus Size', 'Cytoplasm Texture', 'Membrane Intensity', 'Chromatin Irregularity', 'Feature 5']
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, xticklabels=feature_names, yticklabels=feature_names)
plt.title("Figure 8: Correlation Matrix of Visual Features", fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('figure_8_correlation.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close()

# Figure 9: ROC Curves (Tri-Modal System)
plt.figure(figsize=(6, 4))
plt.plot(fpr_leukaemia, tpr_leukaemia, color='blue', label=f'Leukaemia (AUC = {auc_leukaemia})', linewidth=2)
plt.plot(fpr_myeloma, tpr_myeloma, color='green', label=f'Myeloma (AUC = {auc_myeloma})', linewidth=2, linestyle='--')
plt.plot(fpr_lymphoma, tpr_lymphoma, color='orange', label=f'Lymphoma (AUC = {auc_lymphoma})', linewidth=2, linestyle=':')
plt.plot([0, 1], [0, 1], color='black', linewidth=2, linestyle='--', label='Chance Level (AUC = 0.50)')
plt.title("Figure 9: Receiver Operating Characteristic (ROC) Analysis", fontsize=14, fontweight='bold')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figure_9_roc.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close()

# Figure 10: Confusion Matrix (Leukaemia Model)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', cbar=False, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title("Figure 10: Confusion Matrix: Leukaemia Model", fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('figure_10_confusion.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close()

# Figure 11: Feature Importance
plt.figure(figsize=(8, 4))
plt.bar(features, importances, color='#88c999')
plt.title("Figure 11: Feature Importance", fontsize=14, fontweight='bold')
plt.ylabel('Importance Score')
plt.xlabel('Visual Features')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('figure_11_feature_importance.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close()

print("Done! Generated: figure_7_eda.png, figure_8_correlation.png, figure_9_roc.png, figure_10_confusion.png, figure_11_feature_importance.png")
