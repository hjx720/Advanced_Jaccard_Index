import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad

def radians_to_degrees(angles):
    return [np.degrees(angle) for angle in angles]

def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def angle_between(p1, p2, p3):
    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0]) - math.atan2(p3[1] - p2[1], p3[0] - p2[0])
    angle = (angle + math.pi) % (2 * math.pi) - math.pi
    return angle

def compute_turning_function(vertices):
    num_vertices = len(vertices)
    edge_lengths = []
    angles = []
    for i in range(num_vertices):
        next_i = (i + 1) % num_vertices
        if i == 0:
            angle_0 = math.atan2(vertices[next_i][1] - vertices[i][1], vertices[next_i][0] - vertices[i][0])
            angles.append(angle_0)
            edge_length = distance(vertices[i], vertices[i+1])
            edge_lengths.append(edge_length)


        else:
            edge_length = distance(vertices[i], vertices[next_i])
            edge_lengths.append(edge_length)

            angle = angle_between(vertices[i-1], vertices[i], vertices[next_i])
            angle = (angles[i-1] - angle)
            angles.append(angle)
    for i in range(len(angles)):
        angles[i] = angles[i] - angle_0
    total_length = sum(edge_lengths)
    normalized_lengths = [sum(edge_lengths[:i]) / total_length for i in range(num_vertices)]
    lengths_final = []
    angles_final = []
    for i in range(len(normalized_lengths)):
        if i > 0:
            lengths_final.append(normalized_lengths[i-1])
            lengths_final.append(normalized_lengths[i])
            angles_final.append(angles[i-1])
            angles_final.append(angles[i-1])
    lengths_final.append(normalized_lengths[i])
    lengths_final.append(1)
    angles_final.append(angles[i])
    angles_final.append(angles[i])
    return lengths_final, angles_final

def interpolate_turning_function(norm_lengths, turning_angles, num_points=100):
    interpolator = interp1d(norm_lengths, turning_angles, kind='linear', fill_value="extrapolate")
    new_lengths = np.linspace(0, 1, num_points)
    new_turning_angles = interpolator(new_lengths)
    return new_turning_angles

def turning_function_distance(interpolated_angles1, interpolated_angles2, limit=100):
    def integrand(l):
        index1 = int(l * (len(interpolated_angles1) - 1))
        index2 = int(l * (len(interpolated_angles2) - 1))
        return (interpolated_angles1[index1] - interpolated_angles2[index2]) ** 2
    distance, _ = quad(integrand, 0, 1, limit=limit)
    return math.sqrt(distance)

def calculate_similarity(vertices1, vertices2):
    norm_lengths1, turning_angles1 = compute_turning_function(vertices1)
    norm_lengths2, turning_angles2 = compute_turning_function(vertices2)
    

    interpolated_angles1 = interpolate_turning_function(norm_lengths1, turning_angles1)
    interpolated_angles2 = interpolate_turning_function(norm_lengths2, turning_angles2)
    distance = turning_function_distance(interpolated_angles1, interpolated_angles2)
    return distance

def plot_polygon_with_starting_point(vertices):
    # Ensure the vertices form a closed polygon
    vertices.append(vertices[0])
    
    # Extract the coordinates
    x_coords, y_coords = zip(*vertices)
    
    # Calculate the starting point as the midpoint between the last point and the first point
    start_x = (vertices[-2][0] + vertices[0][0]) / 2
    start_y = (vertices[-2][1] + vertices[0][1]) / 2
    starting_point = (start_x, start_y)
    
    # Reference point for angle calculation
    ref_x, ref_y = vertices[-2]
    
    # Compute the angle of the starting point with reference to the X-axis
    dx = ref_x - start_x
    dy = ref_y - start_y
    
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    
    # Plot the polygon
    plt.figure(figsize=(8, 8))
    plt.plot(x_coords, y_coords, 'b-', label='Polygon')
    plt.scatter([start_x], [start_y], color='red', zorder=5, label='Starting Point')
    
    # Plot the reference line
    plt.plot([start_x, vertices[0][0]*1.000001], [start_y, start_y], 'k--', label=None)
    
    # Plot the arc to represent the angle
    arc = np.linspace(0, angle_rad, 100)
    arc_x = start_x + 0.5 * np.cos(arc)
    arc_y = start_y + 0.5 * np.sin(arc)
    plt.plot(arc_x, arc_y, 'k-')
    
    # Add the angle annotation
    plt.text(start_x+0.35, start_y+0.35, 
             f'v', fontsize=12, va='bottom')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Polygon with Starting Point and Angle')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def plot_turning_functions(vertices1, vertices2):
    norm_lengths1, turning_angles1 = compute_turning_function(vertices1)
    norm_lengths2, turning_angles2 = compute_turning_function(vertices2)

    
    turning_angles1_deg = radians_to_degrees(turning_angles1)
    turning_angles2_deg = radians_to_degrees(turning_angles2)
    polygon_1 = np.array([[norm_lengths1[i], turning_angles1_deg[i]] for i in range(len(norm_lengths1))])
    polygon_2 = np.array([[norm_lengths2[i], turning_angles2_deg[i]] for i in range(len(norm_lengths2))])
    plt.figure(figsize=(10, 5))
    plt.plot(polygon_1[:, 0], polygon_1[:, 1], label="Polygon 1")
    plt.plot(polygon_2[:, 0], polygon_2[:, 1], label="Polygon 2")
    plt.xlabel("Normalized Accumulated Length")
    plt.ylabel("Accumulated Turning Angle (degrees)")
    plt.title("Turning Functions of Polygons")
    plt.legend()
    ax = plt.gca()
    y_ticks = ax.get_yticks()
    y_labels = [f"{'v' if tick == 0 else tick}" for tick in y_ticks]
    ax.set_yticklabels(y_labels)
    plt.grid(True)
    plt.show()

    plot_polygon_with_starting_point(vertices1)

# Example usage
vertices1 = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
vertices2 = [(0, 0), (2, 0), (2, 1), (2, 2), (0, 2), (0, 0)]

# Calculate similarity (distance) between polygons
#similarity = calculate_similarity(vertices1, vertices2)
#print(f"Similarity (distance) between polygons: {similarity}")

# Plotting the turning functions
#plot_turning_functions(vertices1, vertices2)

# Checking the angle between three points
p1 = (0, 0)
p2 = (1, 1)
p3 = (1, 2)
#print(f"Angle between points: {np.degrees(angle_between(p1, p2, p3))} degrees")
