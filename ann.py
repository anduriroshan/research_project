import os
import graphviz

# Make sure Graphviz bin is on PATH
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

from graphviz import Digraph

# Create a new directed graph with left-to-right orientation
dot = Digraph(comment='PyTorch ANN Architecture', format='png')
dot.attr(rankdir='LR', size='12,8', nodesep='0.3', ranksep='0.8')  # Reduced ranksep for shorter arrows
dot.attr('node', fontsize='14')  # Increased fontsize for nodes
dot.attr('edge', arrowsize='0.7')  # Smaller arrowheads

# Input layer
dot.node('Input', 'Input\n(2 features)', shape='circle', style='filled', fillcolor='lightblue', fontsize='16')  # Larger text

# First hidden layer cluster - Dense 64
with dot.subgraph(name='cluster_h1') as c:
    c.attr(label='Dense (64) + BatchNorm1d + LeakyReLU + Dropout(0.2)', 
           style='rounded', 
           color='gray', 
           fontsize='18',  # Larger cluster label
           labelloc='t')   # Position label at top
    # Show 8 representative neurons
    for i in range(8):
        c.node(f'h1_{i}', '', shape='circle', style='filled', fillcolor='lightblue', width='0.6', height='0.6')
    # Add ellipsis to show there are more
    c.node('h1_more', '...', shape='none', fontsize='16')  # Larger ellipsis

# Second hidden layer cluster - Dense 32
with dot.subgraph(name='cluster_h2') as c:
    c.attr(label='Dense (32) + BatchNorm1d + LeakyReLU', 
           style='rounded', 
           color='gray', 
           fontsize='18',  # Larger cluster label
           labelloc='t')   # Position label at top
    # Show 6 representative neurons
    for i in range(6):
        c.node(f'h2_{i}', '', shape='circle', style='filled', fillcolor='lightblue', width='0.6', height='0.6')
    # Add ellipsis to show there are more
    c.node('h2_more', '...', shape='none', fontsize='16')  # Larger ellipsis

# Output layer
dot.node('Output', 'Output\n(1 value)', shape='circle', style='filled', fillcolor='lightblue', fontsize='16')  # Larger text

# Connect input to first hidden layer - use weight constraint for shorter arrows
for i in range(8):
    dot.edge('Input', f'h1_{i}', weight='2')  # Weight helps control edge length

# Connect first hidden to second hidden layer (reduce visual clutter)
for i in range(8):
    for j in range(6):
        if (i+j) % 3 == 0:  # Only show 1/3 of connections to reduce clutter
            dot.edge(f'h1_{i}', f'h2_{j}', weight='2')  # Weight helps control edge length

# Connect second hidden to output layer
for i in range(6):
    dot.edge(f'h2_{i}', 'Output', weight='2')  # Weight helps control edge length

# Export to PNG with higher DPI for better quality in thesis
dot.attr(dpi='350')  # Higher DPI for thesis quality
dot.render('pytorch_ann_architecture', format='png', cleanup=True)
print("Thesis-ready visualization saved as pytorch_ann_architecture.png")