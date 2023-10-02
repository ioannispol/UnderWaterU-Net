import sys
import io
import matplotlib.pyplot as plt
from torchsummary import summary
import networkx as nx
from matplotlib.patches import FancyBboxPatch
from underwater_unet.model import AttentionUNet

# Assuming the AttentionUNet model is defined in the script
model = AttentionUNet(n_channels=3, n_classes=1)


def plot_model(model, input_size, filename):
    # Generate the model summary string output
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    summary(model, input_size=input_size)
    output = new_stdout.getvalue()
    sys.stdout = old_stdout

    # Parse the model summary string to generate data for plotting
    lines = output.split('\n')
    node_info = []
    connections = []
    prev_name = None
    for line in lines:
        if '---' in line or '==' in line or not line:
            continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 3:  # Check to ensure there are at least 3 parts
            continue
        name = parts[0]
        output_shape = parts[2]

        node_info.append((name, output_shape))
        if prev_name:
            connections.append((prev_name, name))
        prev_name = name

    # Generate plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')

    G = nx.DiGraph()
    G.add_edges_from(connections)
    pos = nx.spring_layout(G)

    for node, (x, y) in pos.items():
        labels = [n for n, _ in node_info if n in node]
        shapes = [s for _, s in node_info if s in node]
        label = f"{labels[0]}\n{shapes[0]}"
        bbox = FancyBboxPatch((x, y), boxstyle="square,pad=0.3", facecolor="skyblue", edgecolor="black")
        ax.add_patch(bbox)
        ax.text(x, y, label, va='center', ha='center', fontsize=10)

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        arrowprops = dict(facecolor='gray', edgecolor='gray', arrowstyle='-|>', lw=1)
        ax.annotate('', (x1, y1), (x0, y0), arrowprops=arrowprops)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    return filename


# Generate the model visualization
image_path = plot_model(model, (3, 256, 256), "unet_attention_model_viz.png")
image_path
