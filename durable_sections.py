import matplotlib.pyplot as plt
import matplotlib.patches as patches

from durable_sections_data import sections



sections.pop('section_5', None)




def plot_bounding_boxes(sections):
    num_sections = len(sections)
    
    # Create subplots: one for each section
    fig, axes = plt.subplots(1, num_sections, figsize=(14 * num_sections, 10))
    
    if num_sections == 1:
        axes = [axes]  # Ensure axes is iterable when there's only one section
    
    for ax, (section, boxes) in zip(axes, sections.items()):
        ax.set_xlim(0, 1400)  # Set x-axis limit
        ax.set_ylim(0, 1024)  # Set y-axis limit
        ax.set_title(f'Section: {section}')
        ax.invert_yaxis()  # Invert y-axis to match image coordinate system
        
        # Plot each bounding box
        for box in boxes:
            x = box['x']
            y = box['y']
            width = box['width']
            height = box['height']
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        ax.set_aspect('equal')  # Ensure the aspect ratio is maintained

    
    plt.tight_layout()
    plt.show()

#plot_bounding_boxes(sections)
