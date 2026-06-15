import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects

# --- SETUP ---
def draw_box(ax, x, y, width, height, text, color='#FFFFFF', alpha=1.0, edgecolor='black'):
    ax.add_patch(patches.Rectangle((x, y), width, height, fill=color, alpha=alpha, edgecolor=edgecolor, boxstyle='round,pad=0.1'))
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=10, fontweight='bold')

def draw_arrow(ax, x, y, dx, dy):
    ax.arrow(x, y, x + dx, y + dy, width=1.5, head_width=8, head_length=10, fc='k', ec='k')

def save_diagram(fig_num, title):
    plt.title(title, fontsize=16, fontweight='bold', y=1.05)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'figure_{fig_num}.png', dpi=300)
    plt.show()

# --- DIAGRAM 1: SYSTEM ARCHITECTURE ---
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
ax1.set_xlim(0, 12)
ax1.set_ylim(0, 8)

# Clinician
draw_box(ax1, 0.5, 5, 2, "CLINICIAN\n(Laptop)", '#E8F6F3')
draw_arrow(ax1, 2.5, 5, 0.5, 1) # Clinician to Browser

# Web Browser
draw_box(ax1, 1, 4, 3, 1.5, "WEB BROWSER\n(Frontend)", '#A8DADC')
draw_arrow(ax1, 1, 5.5, 1, 3, 1) # Browser to Server

# Flask Server (Container)
draw_box(ax1, 5, 3, 5.5, 3.5, "FLASK SERVER\n(Python Backend)", '#A2E5CC')
draw_arrow(ax1, 1, 7.5, 4, 3.5) # Browser to Server

# Models (Inside Server)
draw_box(ax1, 5.5, 3.5, 1.5, 1, "Blood Model", '#90CAF9')
draw_box(ax1, 7.5, 3.5, 1.5, 1, "Tissue Model", '#90CAF9')
draw_box(ax1, 6.5, 5, 1.5, 1, "Myeloma Model", '#90CAF9')
# Server to Models
ax1.plot([7.25, 7.5], [7.5, 4.5], color='black', linestyle=':', linewidth=2)

# Database
draw_box(ax1, 5, 6.5, 5.5, 1, "SQLITE DATABASE\n(hematology.db)", '#B0E0E6')
# Server to Database
draw_arrow(ax1, 7.5, 6.5, 0, -1.5)

save_diagram(1, "Figure 1: System Architecture")


# --- DIAGRAM 2: DATA PREPROCESSING ---
fig2, ax2 = plt.subplots(1, 1, figsize=(4, 8))
ax2.set_xlim(0, 6)
ax2.set_ylim(0, 8)

y_pos = 7.5

# Steps
draw_box(ax2, 1, y_pos, 4, 1, "INPUT IMAGE", '#FF9999')
draw_arrow(ax2, 3, y_pos, 0, -0.8)

draw_box(ax2, 1, y_pos-1, 4, 1, "RESIZING\n(224x224)", '#FFCC99')
draw_arrow(ax2, 3, y_pos-1, 0, -0.8)

draw_box(ax2, 1, y_pos-2, 4, 1, "STAIN NORM.\n(Macenko)", '#FFCC99')
draw_arrow(ax2, 3, y_pos-2, 0, -0.8)

draw_box(ax2, 1, y_pos-3, 4, 1, "CONVERT TENSOR", '#FFCC99')
draw_arrow(ax2, 3, y_pos-3, 0, -0.8)

draw_box(ax2, 1, y_pos-4, 4, 1, "READY FOR MODEL", '#99FF99')

save_diagram(2, "Figure 2: Data Preprocessing")


# --- DIAGRAM 3: RESNET18 ARCHITECTURE ---
fig3, ax3 = plt.subplots(1, 1, figsize=(3, 8))
ax3.set_xlim(0, 5)
ax3.set_ylim(0, 9)

y_pos = 8.5

# Input
draw_box(ax3, 1, y_pos, 3, 1, "INPUT\n224x224", '#D9E1F2')

# Layers
draw_box(ax3, 1, y_pos-1.5, 3, 1, "CONV 1\n(Detect Edges)", '#FF9999')
draw_arrow(ax3, 2, y_pos, 0, -1.5)

draw_box(ax3, 1, y_pos-3, 3, 1, "CONV 2\n(Detect Shapes)", '#FF9999')
draw_arrow(ax3, 2, y_pos-3, 0, -1.5)

draw_box(ax3, 1, y_pos-4.5, 3, 1, "CONV 3\n(Detect Texture)", '#FF9999')
draw_arrow(ax3, 2, y_pos-4.5, 0, -1.5)

draw_box(ax3, 1, y_pos-6, 3, 1, "RES BLOCK\n(Deep Learning)", '#FF9999')
draw_arrow(ax3, 2, y_pos-6, 0, -1.5)

draw_box(ax3, 1, y_pos-7.5, 3, 1, "DENSE LAYER\n(Combine Features)", '#E8F6F3')
draw_arrow(ax3, 2, y_pos-7.5, 0, -1.5)

draw_box(ax3, 1, y_pos-9, 3, 1, "DENSE LAYER\n(Final Classification)", '#E8F6F3')
draw_arrow(ax3, 2, y_pos-9, 0, -1.5)

draw_box(ax3, 1, y_pos-10.5, 3, 1, "SOFTMAX\n(Prediction)", '#90CAF9')
draw_arrow(ax3, 2, y_pos-10.5, 0, -1.5)

save_diagram(3, "Figure 3: ResNet18 Architecture")


# --- DIAGRAM 4: DECISION RULES ---
fig4, ax4 = plt.subplots(1, 1, figsize=(6, 4))
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 6)

# Input
draw_box(ax4, 1, 3, 2, 1, "AI PREDICTION", '#D9E1F2')

# Decision 1 (Diamond)
decision1 = patches.Polygon(((4.5, 3), (6, 2), (7.5, 3), (6, 4)), closed=True, facecolor='#D1D8A7', edgecolor='black')
ax4.add_patch(decision1)
ax4.text(6, 3, "Is Cancer?", ha='center', va='center', fontsize=9)
draw_arrow(ax4, 3, 3, 1, -0.5)

# Branch 1 (NO)
draw_box(ax4, 8, 3, 2, 1, "STABLE", '#4CAF50')
draw_arrow(ax4, 7, 2.5, 1, 3.5) # To input
draw_arrow(ax4, 5.5, 2.5, 1.5, 2) # From decision

# Branch 2 (YES)
draw_box(ax4, 8, 4.5, 2, 1, "URGENT", '#FF6B6B')
draw_arrow(ax4, 5.5, 3.5, 1.5, 4) # From decision
draw_arrow(ax4, 7.5, 3, 2.5, 4.5) # From decision

save_diagram(4, "Figure 4: Clinical Decision Logic")


# --- DIAGRAM 5: DATABASE SCHEMA ---
fig5, ax5 = plt.subplots(1, 1, figsize=(6, 4))
ax5.set_xlim(0, 8)
ax5.set_ylim(0, 6)

# Table 1
draw_box(ax5, 0.5, 4, 2.5, "PATIENTS TABLE", '#A2E5CC')
ax5.text(1, 5.5, "id (PK)", fontsize=9)
ax5.text(1, 5, "hosp_no", fontsize=9)
ax5.text(1, 4.5, "age", fontsize=9)
ax5.text(1, 3.5, "dob", fontsize=9)
ax5.text(1, 2.5, "address", fontsize=9)

# Table 2
draw_box(ax5, 4.5, 4, 2.5, "DIAGNOSTICS TABLE", '#A2E5CC')
ax5.text(5, 5.5, "id (PK)", fontsize=9)
ax5.text(5, 5, "p_id (FK)", fontsize=9)
ax5.text(5, 4.5, "test_type", fontsize=9)
ax5.text(5, 3.5, "prediction", fontsize=9)
ax5.text(5, 2.5, "timestamp", fontsize=9)

# Connector
draw_arrow(ax5, 2.5, 4, 1, -0.5) # From T1 to T2 (Foreign Key)
ax5.text(3, 3.2, "p_id (FK)", fontsize=9, fontweight='bold', color='red')

save_diagram(5, "Figure 5: Database Schema")