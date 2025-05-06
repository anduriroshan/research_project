import os
import graphviz

# Make sure Graphviz bin is on PATH
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

from graphviz import Digraph

# Create a new directed graph
dot = Digraph(comment='Stacking Ensemble Architecture', format='png')
dot.attr(rankdir='LR', size='12,8', nodesep='0.3', ranksep='0.9')
dot.attr('node', fontsize='14')
dot.attr('edge', arrowsize='0.7')

# Input layer
dot.node('Input', 'Input\n(2 features)', shape='circle', style='filled', fillcolor='lightblue', fontsize='16')

# Base models
dot.node('RF', 'Random Forest\nRegressor', shape='box', style='filled', fillcolor='lightgreen', fontsize='14')
dot.node('LGBM', 'LightGBM\nRegressor', shape='box', style='filled', fillcolor='#90EE90', fontsize='14')
dot.node('NN', 'PyTorch ANN\nRegressor', shape='box', style='filled', fillcolor='lightblue', fontsize='14')

# Base model outputs
dot.node('RF_out', 'RF Predictions', shape='box', style='filled', fillcolor='#E6E6E6', fontsize='14')
dot.node('LGBM_out', 'LGBM Predictions', shape='box', style='filled', fillcolor='#E6E6E6', fontsize='14')
dot.node('NN_out', 'ANN Predictions', shape='box', style='filled', fillcolor='#E6E6E6', fontsize='14')

# Meta-learner
with dot.subgraph(name='cluster_meta') as c:
    c.attr(label='Meta-learner: Gradient Boosting Regressor', style='rounded', color='gray', fontsize='18', labelloc='t')
    
    # Create a few representative trees for GBR
    for t in range(3):
        # Root node
        c.node(f'gbm_root_{t}', '', shape='circle', style='filled', fillcolor='#FFA07A', width='0.5', height='0.5')
        # Leaf nodes
        c.node(f'gbm_leaf_{t}_1', '', shape='circle', style='filled', fillcolor='#FFA07A', width='0.4', height='0.4')
        c.node(f'gbm_leaf_{t}_2', '', shape='circle', style='filled', fillcolor='#FFA07A', width='0.4', height='0.4')
        
        # Connect within GBR trees
        c.edge(f'gbm_root_{t}', f'gbm_leaf_{t}_1')
        c.edge(f'gbm_root_{t}', f'gbm_leaf_{t}_2')
    
    # Add ellipsis for more trees
    c.node('gbm_more', '...', shape='none', fontsize='16')

# Final output
dot.node('Output', 'Final Output\n(1 value)', shape='circle', style='filled', fillcolor='lightblue', fontsize='16')

# Connect input to base models
dot.edge('Input', 'RF', weight='2')
dot.edge('Input', 'LGBM', weight='2')
dot.edge('Input', 'NN', weight='2')

# Connect base models to their outputs
dot.edge('RF', 'RF_out', weight='2')
dot.edge('LGBM', 'LGBM_out', weight='2')
dot.edge('NN', 'NN_out', weight='2')

# Connect outputs to meta-learner trees
for t in range(3):
    dot.edge('RF_out', f'gbm_root_{t}', style='dashed', weight='2')
    dot.edge('LGBM_out', f'gbm_root_{t}', style='dashed', weight='2')
    dot.edge('NN_out', f'gbm_root_{t}', style='dashed', weight='2')

# Connect GBR leaf nodes to output
for t in range(3):
    dot.edge(f'gbm_leaf_{t}_1', 'Output', weight='2')
    dot.edge(f'gbm_leaf_{t}_2', 'Output', weight='2')

# Stack parameters as a note
params = """Stacking Ensemble:
- Base Models:
  • Random Forest (100 trees)
  • LightGBM (200 trees)
  • PyTorch ANN
- Meta-learner:
  • Gradient Boosting (100 trees)"""

dot.node('stack_params', params, shape='note', style='filled', fillcolor='white', fontsize='14')

# Export to PNG with higher DPI for better quality in thesis
dot.attr(dpi='350')
dot.render('stacking_ensemble_architecture', format='png', cleanup=True)
print("Thesis-ready Stacking Ensemble visualization saved as stacking_ensemble_architecture.png")