import healpy as hp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import isrf.circle as circle
import isrf.plot_utils as plot_utils
import isrf.ray as ray
import isrf.stars as stars
import isrf.testing_utils as testing_utils
import isrf.utils as utils




# def plot_rays_interactive(star_positions, directions, ray_length=1.0):
#     """
#     Create interactive 3D plot of rays using Plotly.
    
#     Parameters:
#     -----------
#     star_positions : array, shape (N, 3)
#     directions : array, shape (M, N, 3) or (N, 3)
#     ray_length : float, length of rays to plot
#     """
    
#     # Handle different direction shapes
#     if directions.ndim == 3:  # (M, N, 3)
#         if directions.shape[0] > 1:
#             print(f"Multiple ISRF positions detected ({directions.shape[0]}). Using first one.")
#         directions = directions[0]  # Take first ISRF position
    
#     # Limit number of rays for performance
#     n_stars = len(star_positions)
#     indices = np.random.choice(len(star_positions), n_stars, replace=False)
    
#     stars = star_positions[indices]
#     dirs = directions[indices] if directions.ndim == 2 else directions
    
#     # Normalize directions
#     dirs_norm = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    
#     # Calculate end points
#     end_points = stars + ray_length * dirs_norm
    
#     fig = go.Figure()
    
#     # Add stars
#     fig.add_trace(go.Scatter3d(
#         x=stars[:, 0], y=stars[:, 1], z=stars[:, 2],
#         mode='markers',
#         marker=dict(size=5, color='red', opacity=0.8),
#         name='Stars',
#         hovertemplate='Star<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
#     ))
    
#     # Add rays (as lines)
#     for i in range(len(stars)):
#         fig.add_trace(go.Scatter3d(
#             x=[stars[i, 0], end_points[i, 0]],
#             y=[stars[i, 1], end_points[i, 1]], 
#             z=[stars[i, 2], end_points[i, 2]],
#             mode='lines',
#             line=dict(color='blue', width=2),
#             showlegend=False,
#             hoverinfo='skip'
#         ))
    
#     # Add end points
#     fig.add_trace(go.Scatter3d(
#         x=end_points[:, 0], y=end_points[:, 1], z=end_points[:, 2],
#         mode='markers',
#         marker=dict(size=3, color='green', opacity=0.6),
#         name='Ray Ends',
#         hovertemplate='Ray End<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
#     ))
    
#     # Update layout
#     fig.update_layout(
#         title=f'Star Directions - Interactive View ({n_stars} rays)',
#         scene=dict(
#             xaxis_title='X',
#             yaxis_title='Y',
#             zaxis_title='Z',
#             aspectmode='cube'
#         ),
#         width=900,
#         height=700
#     )
    
#     return fig


def plot_rays_interactive(star_positions, directions, ray_length=1.0):
    """
    Create interactive 3D plot of rays using Plotly with support for multiple ISRF positions.
    All ISRF positions are displayed on the same graph with different colors.
    
    Parameters:
    -----------
    star_positions : array, shape (N, 3)
    directions : array, shape (M, N, 3) or (N, 3)
    ray_length : float, length of rays to plot
    """
    
    # Handle different direction shapes
    if directions.ndim == 2:  # Single ISRF position (N, 3)
        directions = directions[np.newaxis, :]  # Make it (1, N, 3)
    
    n_isrf_positions = directions.shape[0]
    n_stars = len(star_positions)
    
    # Define colors for different ISRF positions
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'pink']
    if n_isrf_positions > len(colors):
        # Generate more colors if needed
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab20')
        colors = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' 
                  for c in cmap(np.linspace(0, 1, n_isrf_positions))]
    
    fig = go.Figure()
    
    # Add stars (same for all ISRF positions, so only add once)
    fig.add_trace(go.Scatter3d(
        x=star_positions[:, 0], 
        y=star_positions[:, 1], 
        z=star_positions[:, 2],
        mode='markers',
        marker=dict(size=5, color='red', opacity=0.8),
        name='Stars',
        hovertemplate='Star<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
    ))
    
    # Add rays for each ISRF position
    for isrf_idx in range(n_isrf_positions):
        dirs = directions[isrf_idx]
        
        
        # Calculate end points
        end_points = star_positions + dirs
        
        # Color for this ISRF position
        color = colors[isrf_idx % len(colors)]
        
        # Add rays as a single trace with None separators for better performance
        ray_x = []
        ray_y = []
        ray_z = []
        
        for i in range(n_stars):
            ray_x.extend([star_positions[i, 0], end_points[i, 0], None])
            ray_y.extend([star_positions[i, 1], end_points[i, 1], None])
            ray_z.extend([star_positions[i, 2], end_points[i, 2], None])
        
        fig.add_trace(go.Scatter3d(
            x=ray_x, y=ray_y, z=ray_z,
            mode='lines',
            line=dict(color=color, width=2),
            name=f'ISRF {isrf_idx}',
            hoverinfo='skip'
        ))
        
        # Add end points
        fig.add_trace(go.Scatter3d(
            x=end_points[:, 0], 
            y=end_points[:, 1], 
            z=end_points[:, 2],
            mode='markers',
            marker=dict(size=3, color=color, opacity=0.6),
            name=f'ISRF {isrf_idx} Ends',
            showlegend=False,  # Hide from legend to reduce clutter
            hovertemplate=f'ISRF {isrf_idx} Ray End<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>'
        ))
    
    # Update layout
    title = f'Star Directions - {n_isrf_positions} ISRF Position{"s" if n_isrf_positions > 1 else ""} ({n_stars} rays each)'
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        width=900,
        height=700,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    return fig

def create_test_data():

    star_positions_galactic =  testing_utils.get_example_stars()
    isrf_positions_galactic = testing_utils.get_example_ISRF()

    star_field = stars.StarField(star_positions_galactic)
    ax = star_field.plot_stars()
    star_field.get_isrf_locations(isrf_positions_galactic)
    star_field.plot_isrf(ax)

    star_field.get_isrf_directions()
    # print("Direction:", star_field.directions)
    star_positions = star_field.stars_cartesian
    isrf_positions = star_field.isrf_cartesian
    print(star_positions_galactic, star_positions_galactic + star_field.directions[0] - isrf_positions_galactic[0])
    print("Directions to ISRF 0", star_field.directions[0])


    print("Star +plus direction:", star_positions + star_field.directions[0])
    print("ISRF position 0:", isrf_positions[0])

    # plot_utils.add_healpixels(ax, nside=2,alpha=0.05, centers = True, s=1, radius=2, color='blue')
    # plt.show(block=True)

    # """Create sample data for testing."""
    # np.random.seed(42)
    
    # # Create random star positions
    # n_stars = 100
    # star_positions = star_
    
    # # Create random ISRF positions
    # n_isrf = 3
    # isrf_positions = np.random.randn(n_isrf, 3) * 20
    
    # # Calculate directions
    # directions = isrf_positions[:, np.newaxis, :] - star_positions[np.newaxis, :, :]
    
    return star_positions, star_field.directions

# Test the functions
if __name__ == "__main__":
    # Create test data
    stars, dirs = create_test_data()
    
    print(f"Stars shape: {stars.shape}")
    print(f"Directions shape: {dirs.shape}")
    
    # Create interactive plot
    fig_3d = plot_rays_interactive(stars, dirs)
    fig_3d.show()
    
