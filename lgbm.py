import os
import graphviz

# Make sure Graphviz bin is on PATH
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"
dot = graphviz.Digraph(comment='LightGBM Architecture')
dot.attr(rankdir='TB')  # Top to bottom layout
dot.attr(size='9,11')  # Canvas size with space for parameter box
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

# Create several LightGBM trees
n_trees = 3  # Number of trees to visualize

# Add tree roots in the same rank
with dot.subgraph() as s:
    s.attr(rank='same')
    for i in range(1, n_trees + 1):
        tree_root = f'tree_root_{i}'
        s.node(tree_root, '', shape='circle', width='0.5', height='0.5', fillcolor='#90EE90', style='filled')
    
    # Add ellipsis for trees
    s.node('ellipsis', '...', shape='plaintext')
    s.edge('tree_root_1', 'tree_root_2', style='invis')
    s.edge('tree_root_2', 'ellipsis', style='invis')
    s.edge('ellipsis', 'tree_root_3', style='invis')

# Connect dataset to tree roots
for i in range(1, n_trees + 1):
    dot.edge('dataset', f'tree_root_{i}')

# For each tree, create a different leaf-wise growth pattern (LightGBM's specialty)
for i in range(1, n_trees + 1):
    tree_root = f'tree_root_{i}'
    
    # First tree - deeper on left side (leaf-wise growth)
    if i == 1:
        # First level
        node1 = f'tree_{i}_node_1'
        node2 = f'tree_{i}_node_2'
        dot.node(node1, '', shape='circle', width='0.4', height='0.4', fillcolor='#90EE90', style='filled')
        dot.node(node2, '', shape='circle', width='0.4', height='0.4', fillcolor='#90EE90', style='filled')
        
        # Add nodes to same rank
        with dot.subgraph() as s:
            s.attr(rank='same')
            s.node(node1)
            s.node(node2)
            s.edge(node1, node2, style='invis')
        
        # Connect root to first level
        dot.edge(tree_root, node1)
        dot.edge(tree_root, node2)
        
        # Second level - only expand the left node (leaf-wise)
        node3 = f'tree_{i}_node_3'
        leaf1 = f'tree_{i}_leaf_1'
        dot.node(node3, '', shape='circle', width='0.4', height='0.4', fillcolor='#90EE90', style='filled')
        dot.node(leaf1, '', shape='circle', width='0.3', height='0.3', fillcolor='#90EE90', style='filled')
        
        # Connect first to second level
        dot.edge(node1, node3)
        dot.edge(node1, leaf1)
        
        # Add leaf nodes for node2
        leaf2 = f'tree_{i}_leaf_2'
        leaf3 = f'tree_{i}_leaf_3'
        dot.node(leaf2, '', shape='circle', width='0.3', height='0.3', fillcolor='#90EE90', style='filled')
        dot.node(leaf3, '', shape='circle', width='0.3', height='0.3', fillcolor='#90EE90', style='filled')
        
        # Connect node2 to its leaves
        dot.edge(node2, leaf2)
        dot.edge(node2, leaf3)
        
        # Third level - expand node3 (leaf-wise growth continues)
        leaf4 = f'tree_{i}_leaf_4'
        leaf5 = f'tree_{i}_leaf_5'
        dot.node(leaf4, '', shape='circle', width='0.3', height='0.3', fillcolor='#90EE90', style='filled')
        dot.node(leaf5, '', shape='circle', width='0.3', height='0.3', fillcolor='#90EE90', style='filled')
        
        # Connect node3 to its leaves
        dot.edge(node3, leaf4)
        dot.edge(node3, leaf5)
        
    # Second tree - different growth pattern
    elif i == 2:
        # First level
        node1 = f'tree_{i}_node_1'
        node2 = f'tree_{i}_node_2'
        dot.node(node1, '', shape='circle', width='0.4', height='0.4', fillcolor='#90EE90', style='filled')
        dot.node(node2, '', shape='circle', width='0.4', height='0.4', fillcolor='#90EE90', style='filled')
        
        # Add nodes to same rank
        with dot.subgraph() as s:
            s.attr(rank='same')
            s.node(node1)
            s.node(node2)
            s.edge(node1, node2, style='invis')
        
        # Connect root to first level
        dot.edge(tree_root, node1)
        dot.edge(tree_root, node2)
        
        # Expand node2 (different leaf-wise pattern)
        node3 = f'tree_{i}_node_3'
        leaf1 = f'tree_{i}_leaf_1'
        dot.node(node3, '', shape='circle', width='0.4', height='0.4', fillcolor='#90EE90', style='filled')
        dot.node(leaf1, '', shape='circle', width='0.3', height='0.3', fillcolor='#90EE90', style='filled')
        
        # Connect node2 to next level
        dot.edge(node2, node3)
        dot.edge(node2, leaf1)
        
        # Leaves for node1
        leaf2 = f'tree_{i}_leaf_2'
        leaf3 = f'tree_{i}_leaf_3'
        dot.node(leaf2, '', shape='circle', width='0.3', height='0.3', fillcolor='#90EE90', style='filled')
        dot.node(leaf3, '', shape='circle', width='0.3', height='0.3', fillcolor='#90EE90', style='filled')
        
        # Connect node1 to leaves
        dot.edge(node1, leaf2)
        dot.edge(node1, leaf3)
        
        # Third level - expand node3
        leaf4 = f'tree_{i}_leaf_4'
        leaf5 = f'tree_{i}_leaf_5'
        dot.node(leaf4, '', shape='circle', width='0.3', height='0.3', fillcolor='#90EE90', style='filled')
        dot.node(leaf5, '', shape='circle', width='0.3', height='0.3', fillcolor='#90EE90', style='filled')
        
        # Connect node3 to leaves
        dot.edge(node3, leaf4)
        dot.edge(node3, leaf5)
        
    # Third tree (Tree-N) - simpler structure
    else:
        # First level
        node1 = f'tree_{i}_node_1'
        node2 = f'tree_{i}_node_2'
        dot.node(node1, '', shape='circle', width='0.4', height='0.4', fillcolor='#90EE90', style='filled')
        dot.node(node2, '', shape='circle', width='0.4', height='0.4', fillcolor='#90EE90', style='filled')
        
        # Add nodes to same rank
        with dot.subgraph() as s:
            s.attr(rank='same')
            s.node(node1)
            s.node(node2)
            s.edge(node1, node2, style='invis')
        
        # Connect root to first level
        dot.edge(tree_root, node1)
        dot.edge(tree_root, node2)
        
        # Leaf nodes for node1
        leaf1 = f'tree_{i}_leaf_1'
        leaf2 = f'tree_{i}_leaf_2'
        dot.node(leaf1, '', shape='circle', width='0.3', height='0.3', fillcolor='#90EE90', style='filled')
        dot.node(leaf2, '', shape='circle', width='0.3', height='0.3', fillcolor='#90EE90', style='filled')
        
        # Connect node1 to leaves
        dot.edge(node1, leaf1)
        dot.edge(node1, leaf2)
        
        # Leaf nodes for node2
        leaf3 = f'tree_{i}_leaf_3'
        leaf4 = f'tree_{i}_leaf_4'
        dot.node(leaf3, '', shape='circle', width='0.3', height='0.3', fillcolor='#90EE90', style='filled')
        dot.node(leaf4, '', shape='circle', width='0.3', height='0.3', fillcolor='#90EE90', style='filled')
        
        # Connect node2 to leaves
        dot.edge(node2, leaf3)
        dot.edge(node2, leaf4)

# Add tree labels in same rank
with dot.subgraph() as s:
    s.attr(rank='same')
    s.node('tree_label_1', 'LightGBM Tree-1', shape='plaintext')
    s.node('tree_label_2', 'LightGBM Tree-2', shape='plaintext')
    s.node('tree_label_3', 'LightGBM Tree-200', shape='plaintext')
    s.edge('tree_label_1', 'tree_label_2', style='invis')
    s.edge('tree_label_2', 'tree_label_3', style='invis')

# Connect tree labels with invisible edges for alignment
dot.edge(f'tree_1_leaf_3', 'tree_label_1', style='invis')
dot.edge(f'tree_2_leaf_3', 'tree_label_2', style='invis')
dot.edge(f'tree_3_leaf_3', 'tree_label_3', style='invis')

# Add result nodes in same rank
with dot.subgraph() as s:
    s.attr(rank='same')
    for i in range(1, n_trees + 1):
        result_id = f'result_{i}'
        label = f'Result-{i if i < n_trees else "200"}'
        s.node(result_id, label)
    s.edge('result_1', 'result_2', style='invis')
    s.edge('result_2', 'result_3', style='invis')

# Connect tree labels to result nodes
for i in range(1, n_trees + 1):
    dot.edge(f'tree_label_{i}', f'result_{i}')

# Add boosted aggregation node
dot.node('boosting', 'Boosted Aggregation', fillcolor='white')
for i in range(1, n_trees + 1):
    dot.edge(f'result_{i}', 'boosting')

# Add final result
dot.node('final', 'Final Result')
dot.edge('boosting', 'final')

# Add model parameters box
dot.node('params', '''LightGBM Parameters:
n_estimators = 200 trees
learning_rate = 0.05
max_depth = 5
num_leaves = 20
reg_alpha = 0.1
reg_lambda = 0.1''', 
        shape='note', style='filled', fillcolor='#f0f8ff', fontsize='10')

# Position parameters box to the right
with dot.subgraph() as s:
    s.attr(rank='same')
    s.node('invis_param', style='invis', shape='point', width='0')
    s.edge('dataset', 'invis_param', style='invis')
    s.edge('invis_param', 'params', style='invis')



# Render the graph
dot.attr(dpi='300')  # Higher resolution
dot.render('lightgbm_architecture', format='png', cleanup=True)
print("LightGBM visualization created as 'lightgbm_architecture.png'")