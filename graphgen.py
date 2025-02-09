import pandas as pd
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import colorsys
from matplotlib.colors import LinearSegmentedColormap
import time

def is_usdc_column(col_name):
    """Check if column contains USDC values"""
    return '[usdc]' in col_name.lower()

def clean_and_filter_data(df, col1, col2, col3):
    """Clean data with special handling for USDC columns and aggressive outlier removal"""
    df_clean = df[[col1, col2, col3]].copy()
    
    # Identify which columns are USDC
    cols_is_usdc = [is_usdc_column(col) for col in [col1, col2, col3]]
    
    # Filter for each column
    for col, is_usdc in zip([col1, col2, col3], cols_is_usdc):
        if is_usdc:
            # For USDC columns, filter out low values and extreme outliers
            df_clean = df_clean[df_clean[col].fillna(0) >= 50]
            # Remove extreme outliers (keep 5th to 85th percentile for USDC)
            lower = df_clean[col].quantile(0.05)
            upper = df_clean[col].quantile(0.85)
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        else:
            # For non-USDC columns, remove zeros, NaNs, and outliers
            df_clean = df_clean[df_clean[col] != 0]
            df_clean = df_clean.dropna(subset=[col])
            # Remove outliers (keep 10th to 90th percentile for non-USDC)
            lower = df_clean[col].quantile(0.10)
            upper = df_clean[col].quantile(0.90)
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    
    # Randomly sample 25% of the data
    sample_size = len(df_clean) // 4
    df_clean = df_clean.sample(n=sample_size, random_state=42)
    
    return df_clean

def get_random_numerical_columns(df, n=3):
    """Get three random numerical columns from the dataframe"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_cols) < n:
        return None, None, None
    
    return random.sample(numerical_cols, n)

def create_scatter_data(x, y, z, col1, col2, col3):
    """Prepare data for 3D scatter plot with appropriate scaling for different data types"""
    def normalize(arr, col_name):
        if is_usdc_column(col_name):
            # For USDC columns, use percentile filtering
            arr_min, arr_max = np.percentile(arr, [5, 95])
        elif '[%]' in col_name.lower():
            # For percentage columns, use full 0-100 range
            arr_min, arr_max = 0, 100
        else:
            # For other columns, use actual min/max
            arr_min, arr_max = np.min(arr), np.max(arr)
        
        # Prevent division by zero
        if arr_max == arr_min:
            return np.zeros_like(arr)
        
        return np.clip((arr - arr_min) / (arr_max - arr_min), 0, 1)
    
    x_norm = normalize(x, col1)
    y_norm = normalize(y, col2)
    z_norm = normalize(z, col3)
    
    return x_norm, y_norm, z_norm

def create_animation(file_number):
    """Create animation with sequential file numbering"""
    # Read the data
    try:
        df = pd.read_csv('data.csv')
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    # Get random columns and process data
    col1, col2, col3 = get_random_numerical_columns(df)
    if col1 is None:
        print("Not enough numerical columns found in the data")
        return

    # Clean and filter data
    df_clean = clean_and_filter_data(df, col1, col2, col3)
    
    # Get coordinates
    x = df_clean[col1].values
    y = df_clean[col2].values
    z = df_clean[col3].values
    
    # Randomly sample 25% of the points
    sample_size = len(x) // 4
    indices = np.random.choice(len(x), sample_size, replace=False)
    x = x[indices]
    y = y[indices]
    z = z[indices]
    
    # Prepare scatter plot data
    x_norm, y_norm, z_norm = create_scatter_data(x, y, z, col1, col2, col3)

    # Create figure and 3D axes
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    def init():
        ax.clear()
        
        # Define colors for the gradient (light to dark pinks)
        colors = [
            '#FFE4E1',  # Misty Rose (lightest pink)
            '#FFB6C1',  # Light Pink
            '#FF69B4',  # Hot Pink
            '#DB7093',  # Pale Violet Red
            '#C71585'   # Medium Violet Red (darkest pink)
        ]
        
        n_bins = 100  # Number of color gradients
        pink_cmap = LinearSegmentedColormap.from_list("custom_pinks", colors, N=n_bins)
        
        # Create scatter plot with pink gradient
        scatter = ax.scatter(x, y, z,
                            c=np.linspace(0, 1, len(x)),
                            s=50,  # point size
                            alpha=0.8,
                            cmap=pink_cmap)
        
        # Set dark forest green theme
        forest_green = '#0B2815'  # Darker forest green
        ax.set_facecolor(forest_green)
        fig.patch.set_facecolor(forest_green)
        
        # Customize grid and axis colors
        ax.grid(True, alpha=0.2, color='#ffffff')  # White grid
        
        # Set pane colors with normalized RGB values (0-1 range)
        pane_color = (0.043, 0.157, 0.082, 0.3)  # Slightly lighter forest green with transparency
        ax.xaxis.set_pane_color(pane_color)
        ax.yaxis.set_pane_color(pane_color)
        ax.zaxis.set_pane_color(pane_color)
        
        # Format axis ticks for better readability
        def format_ticks(x, pos):
            if abs(x) >= 1e6:
                return f'{x/1e6:.1f}M'
            elif abs(x) >= 1e3:
                return f'{x/1e3:.1f}K'
            return f'{x:.1f}'
        
        # Style axis labels and ticks in white
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.label.set_color('#ffffff')
            axis.set_tick_params(colors='#ffffff')
            axis.set_major_formatter(plt.FuncFormatter(format_ticks))
        
        # Set static axis labels that rotate with the plot
        ax.set_xlabel(col1.replace('_', ' ').title(), color='#ffffff', fontsize=10, labelpad=20)
        ax.set_ylabel(col2.replace('_', ' ').title(), color='#ffffff', fontsize=10, labelpad=20)
        ax.set_zlabel(col3.replace('_', ' ').title(), color='#ffffff', fontsize=10, labelpad=20)
        
        # Fix the axis labels to stay oriented correctly
        ax.xaxis._axinfo['label']['space_factor'] = 2.0
        ax.yaxis._axinfo['label']['space_factor'] = 2.0
        ax.zaxis._axinfo['label']['space_factor'] = 2.0
        
        # Rotate labels to maintain readability
        ax.xaxis.set_rotate_label(False)
        ax.yaxis.set_rotate_label(False)
        ax.zaxis.set_rotate_label(False)
        
        plt.title("")
        
        # Add white edges to the plot box
        ax.xaxis._axinfo["grid"]['color'] = "#ffffff"
        ax.yaxis._axinfo["grid"]['color'] = "#ffffff"
        ax.zaxis._axinfo["grid"]['color'] = "#ffffff"
        
        return [scatter]

    def animate(frame):
        # Update view angle
        ax.view_init(elev=20, azim=frame)
        
        # Keep labels oriented correctly
        ax.set_xlabel(col1.replace('_', ' ').title(), color='#ffffff', fontsize=10, labelpad=20)
        ax.set_ylabel(col2.replace('_', ' ').title(), color='#ffffff', fontsize=10, labelpad=20)
        ax.set_zlabel(col3.replace('_', ' ').title(), color='#ffffff', fontsize=10, labelpad=20)
        
        return []

    # Create sequential filename
    filename = f'scatter_{file_number:03d}.mp4'  # Changed back to .mp4
    
    # Create the animation
    print(f"Creating animation {file_number}/50...")
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=240,  # 240 frames for 10 seconds at 24fps
                                 interval=20,  # Keep smooth interval
                                 blit=True)
    
    # Save as MP4
    writer = animation.FFMpegWriter(fps=24, bitrate=2000)
    anim.save(filename, writer=writer)
    
    print(f"Created 3D scatter plot of:")
    print(f"X: {col1}")
    print(f"Y: {col2}")
    print(f"Z: {col3}")
    print(f"Number of points: {len(x)}")
    print(f"\nAnimation has been saved as '{filename}'")
    
    plt.close()  # Clean up the figure

def main():
    """Generate 50 visualizations"""
    for i in range(1, 51):
        try:
            create_animation(i)
            if i < 50:  # Don't wait after the last one
                print("Waiting 5 seconds before next generation...")
                time.sleep(5)
        except Exception as e:
            print(f"Error generating visualization {i}: {e}")
            time.sleep(30)  # Wait longer on error
            continue
    
    print("\nCompleted generating all 50 visualizations!")

if __name__ == "__main__":
    main()
