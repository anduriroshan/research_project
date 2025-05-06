import os
import graphviz

# Make sure Graphviz bin is on PATH
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

# Create a new directed graph
dot = graphviz.Digraph(comment='Random Forest Architecture')
dot.attr(rankdir='TB')  # Top to bottom layout
dot.attr(size='9,11')  # Increased canvas size for parameter box
dot.attr('node', fontname='Arial', shape='box', style='filled', fillcolor='white')
dot.attr('edge', fontname='Arial')

# Create invisible nodes to help with alignment
with dot.subgraph() as s:
    s.attr(rank='same')
    s.node('invis1', style='invis', shape='point', width='0')
    s.node('dataset', 'Dataset', fillcolor='white')
    s.node('invis2', style='invis', shape='point', width='0')
    s.edge('invis1', 'dataset', style='invis')
    s.edge('dataset', 'invis2', style='invis')

# Create several decision trees
n_trees = 3  # Number of trees to visualize

# Add tree roots in the same rank
with dot.subgraph() as s:
    s.attr(rank='same')
    for i in range(1, n_trees + 1):
        tree_root = f'tree_root_{i}'
        s.node(tree_root, '', shape='circle', width='0.5', height='0.5', fillcolor='#e74c3c', style='filled')
    
    # Add ellipsis for trees
    s.node('ellipsis', '...', shape='plaintext')
    s.edge('tree_root_1', 'tree_root_2', style='invis')
    s.edge('tree_root_2', 'ellipsis', style='invis')
    s.edge('ellipsis', 'tree_root_3', style='invis')

# Connect dataset to tree roots
for i in range(1, n_trees + 1):
    dot.edge('dataset', f'tree_root_{i}')

# Add tree structures
for i in range(1, n_trees + 1):
    # Root node for each tree already created
    tree_root = f'tree_root_{i}'
    
    # First level nodes
    if i == 1:  # First tree
        colors = ['#e74c3c', '#3498db', '#3498db']
    elif i == 2:  # Second tree
        colors = ['#e74c3c', '#e74c3c', '#3498db']
    else:  # Third tree (Tree-N)
        colors = ['#e74c3c', '#3498db', '#3498db']
    
    # Create first level nodes (with same rank)
    with dot.subgraph() as s:
        s.attr(rank='same')
        nodes = []
        for j, color in enumerate(colors):
            node_id = f'tree_{i}_node_{j}'
            s.node(node_id, '', shape='circle', width='0.4', height='0.4', fillcolor=color, style='filled')
            nodes.append(node_id)
        
        # Add invisible edges to keep nodes aligned horizontally
        if len(nodes) > 1:
            for j in range(len(nodes)-1):
                s.edge(nodes[j], nodes[j+1], style='invis')
    
    # Connect root to first level nodes
    for j, node_id in enumerate(nodes):
        dot.edge(tree_root, node_id)
    
    # Create leaf nodes (lowest level)
    leaf_nodes = []
    with dot.subgraph() as s:
        s.attr(rank='same')
        
        # Configure leaf node colors based on the tree
        if i == 1:  # First tree
            leaf_colors = ['#3498db', '#3498db', '#3498db', '#3498db', '#e74c3c']
        elif i == 2:  # Second tree
            leaf_colors = ['#e74c3c', '#3498db', '#3498db', '#3498db', '#3498db']
        else:  # Third tree (Tree-N)
            leaf_colors = ['#3498db', '#3498db', '#3498db', '#3498db', '#3498db']
        
        # Create leaf nodes
        for k, color in enumerate(leaf_colors):
            leaf_id = f'tree_{i}_leaf_{k}'
            s.node(leaf_id, '', shape='circle', width='0.3', height='0.3', fillcolor=color, style='filled')
            leaf_nodes.append(leaf_id)
        
        # Add invisible edges to keep leaf nodes aligned horizontally
        if len(leaf_nodes) > 1:
            for k in range(len(leaf_nodes)-1):
                s.edge(leaf_nodes[k], leaf_nodes[k+1], style='invis')
    
    # Connect first level nodes to leaf nodes
    # Distribute leaf nodes among parent nodes
    if i == 1:  # First tree
        dot.edge(nodes[0], leaf_nodes[0])
        dot.edge(nodes[0], leaf_nodes[1])
        dot.edge(nodes[1], leaf_nodes[2])
        dot.edge(nodes[1], leaf_nodes[3])
        dot.edge(nodes[1], leaf_nodes[4])
    elif i == 2:  # Second tree
        dot.edge(nodes[0], leaf_nodes[0])
        dot.edge(nodes[0], leaf_nodes[1])
        dot.edge(nodes[1], leaf_nodes[2])
        dot.edge(nodes[1], leaf_nodes[3])
        dot.edge(nodes[2], leaf_nodes[4])
    else:  # Third tree (Tree-N)
        dot.edge(nodes[0], leaf_nodes[0])
        dot.edge(nodes[1], leaf_nodes[1])
        dot.edge(nodes[1], leaf_nodes[2])
        dot.edge(nodes[2], leaf_nodes[3])
        dot.edge(nodes[2], leaf_nodes[4])

# Add tree labels in same rank
with dot.subgraph() as s:
    s.attr(rank='same')
    s.node('tree_label_1', 'Decision Tree-1', shape='plaintext')
    s.node('tree_label_2', 'Decision Tree-2', shape='plaintext')
    s.node('tree_label_3', 'Decision Tree-100', shape='plaintext')
    s.edge('tree_label_1', 'tree_label_2', style='invis')
    s.edge('tree_label_2', 'tree_label_3', style='invis')

# Connect leaf nodes to tree labels with invisible edges for alignment
dot.edge(f'tree_1_leaf_2', 'tree_label_1', style='invis')
dot.edge(f'tree_2_leaf_2', 'tree_label_2', style='invis')
dot.edge(f'tree_3_leaf_2', 'tree_label_3', style='invis')

# Add result nodes in same rank
with dot.subgraph() as s:
    s.attr(rank='same')
    for i in range(1, n_trees + 1):
        result_id = f'result_{i}'
        label = f'Result-{i if i < n_trees else "100"}'
        s.node(result_id, label)
    s.edge('result_1', 'result_2', style='invis')
    s.edge('result_2', 'result_3', style='invis')

# Connect tree labels to result nodes
for i in range(1, n_trees + 1):
    dot.edge(f'tree_label_{i}', f'result_{i}')

# Add majority voting/averaging node
dot.node('voting', 'Majority Voting / Averaging', fillcolor='white')
for i in range(1, n_trees + 1):
    dot.edge(f'result_{i}', 'voting')

# Add final result
dot.node('final', 'Final Result')
dot.edge('voting', 'final')

# Add model parameters box
dot.node('params', '''RandomForestRegressor Parameters:
n_estimators = 100 trees
max_depth = 8
min_samples_split = 5
random_state = 42
n_jobs = n_cores''', 
        shape='note', style='filled', fillcolor='#f0f8ff', fontsize='10')

# Position parameters box to the right
with dot.subgraph() as s:
    s.attr(rank='same')
    s.node('invis_param', style='invis', shape='point', width='0')
    s.edge('dataset', 'invis_param', style='invis')
    s.edge('invis_param', 'params', style='invis')


# Render the graph
dot.attr(dpi='300')  # Higher resolution
dot.render('random_forest_architecture', format='png', cleanup=True)
print("Random Forest visualization created as 'random_forest_architecture.png'")