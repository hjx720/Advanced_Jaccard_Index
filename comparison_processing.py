import sqlite3
from comparison import compare_geopackages
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import os
import warnings
import numpy as np

# Suppress specific category of warnings
warnings.filterwarnings("ignore", category=UserWarning)

def fetch_building_data_by_base_name(base_name):
    # Connect to the SQLite database
    conn = sqlite3.connect('building_data.db')
    cursor = conn.cursor()

    # Fetch data from the database based on base_name
    cursor.execute('''SELECT * FROM building_data WHERE base_name = ? AND manipulation_type = "None"''', (base_name, ))
    reference = cursor.fetchall()
    cursor.execute('''SELECT * FROM building_data WHERE base_name = ?''', (base_name, ))
    data = cursor.fetchall()
    # Close the connection
    conn.close()

    return reference, data

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.widgets import CheckButtons
import matplotlib.pyplot as plt


def plot(results, key, limits, multiple = False):
    # Extracting GeoDataFrames and other attributes from the results dictionary
    gdf_a = results["Before Transformation"]["data A"]
    gdf_b = results["Before Transformation"]["data B"]
    gdf_b_translation = results["After Translation"]["data B"]
    gdf_b_rotate = results["After Rotation"]["data B"]
    gdf_b_scaling = results["After Scaling"]["data B"]

    jaccard_index_original = results["Before Transformation"]["Jaccard_Coefficient"]
    jaccard_index_translation = results["After Translation"]["Jaccard_Coefficient"]
    jaccard_index_rotation = results["After Rotation"]["Jaccard_Coefficient"]
    jaccard_index_scaling = results["After Scaling"]["Jaccard_Coefficient"]

    translation_note_x_axis = results["After Translation"]["Translation Note X Axis"]
    determined_x_translation = results["After Translation"]["Determined X Translation"]
    used_x_translation = results["After Translation"]["X Translation used"]
    translation_note_y_axis = results["After Translation"]["Translation Note Y Axis"]
    determined_y_translation = results["After Translation"]["Determined Y Translation"]
    used_y_translation = results["After Translation"]["Y Translation used"]
    rotation_note = results["After Rotation"]["Rotation Note"]
    determined_rotation_angle = results["After Rotation"]["Determined Rotation Angle"]
    used_rotation_angle = results["After Rotation"]["Rotation Angle used"]
    scaling_note = results["After Scaling"]["Scaling Note"]
    determined_scaling_factor = results["After Scaling"]["Determined Scaling Factor"]
    used_scaling_factor = results["After Scaling"]["Scaling Factor used"]

    if multiple == False:
        # Plotting combined GeoDataFrames
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot gdf_a
        gdf_a.boundary.plot(ax=ax, color="green", edgecolor=None)

        # Plot gdf_b with boundary
        gdf_b.boundary.plot(ax=ax, color=None, edgecolor='#F0F0F0')

        # Plot gdf_b_translation with boundary
        gdf_b_translation.boundary.plot(ax=ax, color=None, edgecolor='#CCCCCC')

        # Plot gdf_b_rotate with boundary
        gdf_b_rotate.boundary.plot(ax=ax, color=None, edgecolor='#666666')

        # Plot gdf_b_scaling with boundary
        gdf_b_scaling.boundary.plot(ax=ax, color=None, edgecolor='#333333')

        # Setting title
        subtitle = os.path.basename(key)
        ax.set_title("Advanced Jaccard Index Process")
        ax.text(0.5, 0.95, subtitle, transform=ax.transAxes, fontsize=12, ha='center')

        # Adding legend with custom handles
        handles = [Patch(edgecolor='green', facecolor= 'none', alpha=1), 
                Patch(edgecolor='#F0F0F0', facecolor= 'none', alpha=1), 
                Patch(edgecolor='#CCCCCC', facecolor= 'none', alpha=1), 
                Patch(edgecolor='#666666', facecolor= 'none', alpha=1), 
                Patch(edgecolor='#333333', facecolor= 'none', alpha=1)]
        
        labels = ['Reference Footprint', 
                'Comparing Original Footprint', 
                'Comparing Footprint after translation', 
                'Comparing Footprint after translation and rotation', 
                'Comparing Footprint after complete Transformation']
        ax.legend(handles, labels, loc='lower right')

        # Annotating Jaccard indices below the plot
        table_data = [
            ["Before Transformation:", None, None, None, None, round(jaccard_index_original, 4)],
            ["X-Axis Translation:", round(determined_x_translation, 4), limits['translation_limit_1'], limits['translation_limit_2'], round(used_x_translation, 4), None],
            ["Y-Axis Translation:", round(determined_y_translation, 4), limits['translation_limit_1'], limits['translation_limit_2'], round(used_y_translation, 4), round(jaccard_index_translation, 4)],
            ["Rotation:",  round(determined_rotation_angle, 4), limits['rotation_limit_1'], limits['rotation_limit_2'], round(used_rotation_angle, 4), round(jaccard_index_rotation, 4)],
            ["Scaling:",  round(determined_scaling_factor, 4), limits['scale_limit_1'], limits['scale_limit_2'], round(used_scaling_factor, 4), round(jaccard_index_scaling, 4)]
        ]

        columns = ['Transformation', 'Determined Value', 'Limit 1', 'Limit 2', 'Used Value', 'Jaccard Index']
        ax.text(0, 0, 'Process Protocol', transform=ax.transAxes, fontsize=12, ha='center')
        table = ax.table(cellText=table_data, colLabels=columns, loc='bottom')
        #plt.subplots_adjust(top=0.8)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)  # Adjust the scale if needed

        # Hide axes
        ax.axis('off')
        # Show plot
        plt.tight_layout()
        plt.show()
    elif multiple == True:
        # Function to update layer visibility
        def update_visibility(label):
            layer_visible[label] = not layer_visible[label]
            update_plot()

        # Function to update the plot based on layer visibility
        def update_plot():
            ax.clear()
            for label in layer_visibility:
                gdf, edgecolor, color, alpha = layer_visibility[label]
                if layer_visible[label]:
                    gdf.plot(ax=ax, color=color, edgecolor=edgecolor, alpha=alpha, linewidth=2)
            ax.set_title("Advanced Jaccard Index Process")
            subtitle = os.path.basename(key)
            ax.text(0.5, 0.95, subtitle, transform=ax.transAxes, fontsize=12, ha='center')
            ax.legend(handles, labels, loc='lower right', frameon=False)
            ax.axis('off')
            ax.set_xlim(0, 70)  # Set x-axis limits from 0 to 6
            ax.set_ylim(-20, 70)
            ax.set_aspect('equal')
            plt.tight_layout()
            plt.draw()

        # Plotting combined GeoDataFrames
        fig, ax = plt.subplots(figsize=(15, 10))

        # Create a separate axis for checkboxes
        ax_checkboxes = plt.axes([0.38, 0.01, 0.02, 0.17])
        # Adding legend with custom handles
        handles = [Patch(edgecolor='none', facecolor= 'green', alpha=0.25), 
                Patch(edgecolor='#cccccc', facecolor= 'none', alpha=1), 
                Patch(edgecolor='#969696', facecolor= 'none', alpha=1), 
                Patch(edgecolor='#636363', facecolor= 'none', alpha=1), 
                Patch(edgecolor='#252525', facecolor= 'none', alpha=1)]

        labels = ['Reference Footprint', 
                'Comparing Original Footprint', 
                'Comparing Footprint after translation', 
                'Comparing Footprint after translation and rotation', 
                'Comparing Footprint after complete Transformation']
        # Create dictionary to store visibility and colors
        layer_visibility = {
            ' ': (gdf_a, 'none', 'green', 0.25),
            '  ': (gdf_b, '#cccccc', 'none', 1),
            '   ': (gdf_b_translation, '#969696', 'none', 1),
            '    ': (gdf_b_rotate, '#636363', 'none', 1),
            '     ': (gdf_b_scaling, '#252525', 'none', 1)
        }

        # Initialize visibility status for each layer
        layer_visible = {label: True for label in layer_visibility}


        # Create checkboxes
        checkbox = CheckButtons(ax_checkboxes, layer_visibility.keys(), [True] * len(layer_visibility))
        checkbox.on_clicked(update_visibility)

        # Hide axes spines (borders) around the checkbox axes
        ax_checkboxes.spines['top'].set_visible(False)
        ax_checkboxes.spines['right'].set_visible(False)
        ax_checkboxes.spines['bottom'].set_visible(False)
        ax_checkboxes.spines['left'].set_visible(False)

        # Hide ticks and labels for the checkbox axes
        #ax_checkboxes.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False)


        # Plot GeoDataFrames and set up legend
        for label, (gdf, edgecolor, color, alpha) in layer_visibility.items():
            gdf.plot(ax=ax, color=color, edgecolor=edgecolor, alpha=alpha, linewidth=2)

        # Setting title
        subtitle = os.path.basename(key)
        ax.set_title("Advanced Jaccard Index Process")
        ax.text(0.5, 0.95, subtitle, transform=ax.transAxes, fontsize=12, ha='center')

        

        ax.legend(handles, labels, loc='lower right', frameon=False)

        # Hide axes
        ax.axis('off')

        # Show plot
        plt.tight_layout()
        plt.show()




if __name__ == "__main__":
    # Example usage
    base_name = 'Building2'
    limits = {
    "translation_limit_1": 1,
    "translation_limit_2": 0.5,
    "rotation_limit_1": 2,
    "rotation_limit_2": 1,
    "scale_limit_1": 1.1,
    "scale_limit_2": 1.05
}
    from shape_context import compare_polygons
    from hausdorff import hausdorff_distance
    reference, building_manipulated = fetch_building_data_by_base_name(base_name)
    all_results = {}
    for row in building_manipulated:
        
        results = compare_geopackages(reference[0][6],row[6],limits)
        shape_context_dist = compare_polygons(results['Before Transformation']['data A'],results['Before Transformation']['data B'])
        hausdorff_dist = hausdorff_distance(results['Before Transformation']['data A'],results['Before Transformation']['data B'])
        results.update({"Shape Context": shape_context_dist,
                        "Hausdorff Distance": hausdorff_dist})
        all_results[row[6]] = results
    
    #print(all_results)
    excel_data = [['key','original Jaccard Index', 'Determined X Translation','X Translation used','Determined Y Translation','Y Translation used','Jaccard Index after Translation','Determined Rotation Angle','Rotation Angle used','Jaccard Index after Rotation','Determined Scaling Factor','Scaling Factor used','final Jaccard Index', 'Shape Context Distance', "Hausdorff Distance"]]
    for key in all_results:
        #plot(all_results[key], key, limits)
        #print(all_results[key])
        #print(all_results[key]['Before Transformation']['Jaccard_Coefficient'])
        #print(all_results[key]['After Translation']['Jaccard_Coefficient'])
        #print(all_results[key]['After Rotation']['Jaccard_Coefficient'])
        #print(all_results[key]['After Scaling']['Jaccard_Coefficient'])
        result = all_results[key]
        comparison_dataname = key.split('\\')
        excel_data.append([
            comparison_dataname[-1],
            result['Before Transformation']['Jaccard_Coefficient'],
            result['After Translation']['Determined X Translation'],
            result['After Translation']['X Translation used'],
            result['After Translation']['Determined Y Translation'],
            result['After Translation']['Y Translation used'],
            result['After Translation']['Jaccard_Coefficient'],
            result['After Rotation']['Determined Rotation Angle'],
            result['After Rotation']['Rotation Angle used'],
            result['After Rotation']['Jaccard_Coefficient'],
            result['After Scaling']['Determined Scaling Factor'],
            result['After Scaling']['Scaling Factor used'],
            result['After Scaling']['Jaccard_Coefficient'],
            result['Shape Context'],
            result['Hausdorff Distance']
        ])

    # Convert the list of lists to a DataFrame
    df = pd.DataFrame(excel_data[1:], columns=excel_data[0])

    # Write DataFrame to Excel file
    df.to_excel('output.xlsx', index=False)
    
    #all_results = {}
    #results = compare_geopackages('Building_2.gpkg','Building_2_vis.gpkg',limits)
    #all_results['row[6]'] = results
    #plot(results, '', limits, True)