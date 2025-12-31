import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon
import math
from tqdm import tqdm
import matplotlib.patches as patches
import warnings
import random

class Partitioner:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.min_x, self.min_y = None, None
        self.n_cells_x, self.n_cells_y = None, None
        self.cell_width_x, self.cell_width_y = None, None
        self.len_x, self.len_y = None, None
        self.unique_ids = set()
        self.boundary = None
    
    def get_boundary(self):
        return self.boundary
            
    def partition(self, gdf, cell_width, tolerance=0.001):
        '''
        Calculates the grid's bounds and number of cells
        O(len(gdf)) as it computes the minimum and maximum
        '''
        # Calculate the grid's bounds and number of cells
        x_list = gdf.geometry.x
        y_list = gdf.geometry.y
        self.min_x, max_x = min(x_list), max(x_list)
        self.min_y, max_y = min(y_list), max(y_list)
        self.len_x = max_x - self.min_x
        self.len_y = max_y - self.min_y
        self.n_cells_x = math.ceil(self.len_x / cell_width)
        self.n_cells_y = math.ceil(self.len_y / cell_width)
        self.cell_width_x = self.len_x / self.n_cells_x
        self.cell_width_y = self.len_y / self.n_cells_y
        self.max_x = self.min_x + self.n_cells_x*self.cell_width_x
        self.max_y = self.min_y + self.n_cells_y*self.cell_width_y
        self.boundary = Polygon([(self.min_x, self.min_y), (self.min_x, self.max_y), (self.max_x, self.max_y), (self.max_x, self.min_y)])  # Example: A square polygon
        print(f'Number of Cells: X={self.n_cells_x}, Y={self.n_cells_y}')

    def calc_cell_id(self, geometry):
        x_index = int((geometry.x - self.min_x) / self.len_x * self.n_cells_x)
        y_index = int((geometry.y - self.min_y) / self.len_y * self.n_cells_y)
        x_index = min(x_index, self.n_cells_x-1)
        y_index = min(y_index, self.n_cells_y-1)
        x_index = max(x_index, 0)
        y_index = max(y_index, 0)
        return (x_index, y_index)
    
    def add_new_cell(self, cell_id):
        # Make sure cell is within range
        if (0 <= cell_id[0] < self.n_cells_x) and (0 <= cell_id[1] < self.n_cells_y):
            self.unique_ids.add(cell_id)
        else: raise ValueError("Cell id is out of range")

    def cell_id_exists(self, cell_id):
        return cell_id in self.unique_ids

    def calculate_ids(self, gdf): # Tested Correctly
        '''
        Takes O(len(gdf)) to calculate. It's the (Point -> Cell) Mapping
        '''
        gdf['cell_id'] = None
        self.unique_ids = set()
        for i, row in tqdm(gdf.iterrows(), total=len(gdf), desc='Calculating Cell ID'):
            cell_id = self.calc_cell_id(row.geometry)
            gdf.at[i, 'cell_id'] = cell_id
            self.unique_ids.add(cell_id)
        return list(self.unique_ids)
    
    def plot_partitioning(self, gdf): # Tested Correctly
        """
        Plots the grid partitioning and points, and shows cell IDs.
        """
        # Plot points
        fig, ax = plt.subplots()
        gdf.plot(ax=ax, color='blue', markersize=5, label='Points')

        # Plot the grid cells
        for i in tqdm(range(self.n_cells_x), total=self.n_cells_x, desc='Plotting Rectangles'):
            for j in range(self.n_cells_y):
                x_left = self.min_x + i * self.cell_width_x
                y_bottom = self.min_y + j * self.cell_width_y
                cell = Polygon([
                    (x_left, y_bottom),
                    (x_left + self.cell_width_x, y_bottom),
                    (x_left + self.cell_width_x, y_bottom + self.cell_width_y),
                    (x_left, y_bottom + self.cell_width_y),
                    (x_left, y_bottom)
                ])
                gdf_cell = gpd.GeoSeries([cell])
                gdf_cell.plot(ax=ax, edgecolor='black', facecolor='none')

                # Label cell with its (i, j) index
                plt.text(x_left + self.cell_width_x / 2, y_bottom + self.cell_width_y / 2, f"({i},{j})",
                         horizontalalignment='center', verticalalignment='center', fontsize=8)

        # Label the points with their cell IDs
        for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc='Labeling Points'):
            x = row.geometry.x
            y = row.geometry.y
            # cell_id = row['cell_id']
            plt.text(x, y, f"{idx}", fontsize=9, color='red')

        plt.title("Partitioning and Points")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid(True)
        plt.legend()
        plt.show()

    def get_Nearbycells_per_cell(self, cell_id, radius, filtered=True):  # Tested Correctly
        # Complexity: O(2*raduis_in_cells_x * 2*raduis_in_cells_y) * 2 (because of filtering)
        #             Very Expensive Function as it increases with radius
        # Calculate Cell_ posiion
        cell_ids = []
        x_index, y_index = cell_id
        if radius <= self.cell_width_x/2: 
            raduis_in_cells_x = 0
            warnings.warn(f"Radius is very small: {radius}. Smaller than half the cell width in x: {self.cell_width_x/2}")
        else: raduis_in_cells_x = math.ceil(radius/self.cell_width_x - 1/2)
        if radius <= self.cell_width_y/2: 
            raduis_in_cells_y = 0
            warnings.warn(f"Radius is very small: {radius}. Smaller than half the cell width in y: {self.cell_width_y/2}")
        else: raduis_in_cells_y = math.ceil(radius/self.cell_width_y - 1/2) 
        x_upper_lim, x_lower_lim = x_index+raduis_in_cells_x, x_index-raduis_in_cells_x
        y_upper_lim, y_lower_lim = y_index+raduis_in_cells_y, y_index-raduis_in_cells_y
        cell_ids = [(x, y) for x in range(x_lower_lim, x_upper_lim+1) for y in range(y_lower_lim, y_upper_lim+1)]
        cell_ids.remove(cell_id)
        if filtered: cell_ids = self.filter_cells(cell_ids)
        return cell_ids

    def get_RightBottomNearbycells_per_cell(self, cell_id, radius, filtered=True):  # Tested Correctly
        # Complexity: O(2*raduis_in_cells_x * 2*raduis_in_cells_y) * 2 (because of filtering)
        #             Very Expensive Function as it increases with radius
        # Calculate Cell_ posiion
        cell_ids = []
        x_index, y_index = cell_id
        if radius <= self.cell_width_x/2: 
            raduis_in_cells_x = 0
            warnings.warn(f"Radius is very small: {radius}. Smaller than half the cell width in x: {self.cell_width_x/2}")
        else: raduis_in_cells_x = math.ceil(radius/self.cell_width_x - 1/2)
        if radius <= self.cell_width_y/2: 
            raduis_in_cells_y = 0
            warnings.warn(f"Radius is very small: {radius}. Smaller than half the cell width in y: {self.cell_width_y/2}")
        else: raduis_in_cells_y = math.ceil(radius/self.cell_width_y - 1/2) 
        
        x_upper_lim, x_lower_lim = x_index+raduis_in_cells_x, x_index
        y_upper_lim, y_lower_lim = y_index+raduis_in_cells_y, y_index
        cell_ids = [(x, y) for x in range(x_lower_lim, x_upper_lim+1) for y in range(y_lower_lim, y_upper_lim+1)]
        
        return cell_ids
    
    def filter_cells(self, cell_ids): # O(cell_ids) // searching in self.unique_ids is O(1) as it 's a hash table
        return [cell for cell in cell_ids if cell in self.unique_ids]
    
    def plot_nearby_cells_results(self, nearby_cells, central_cell, radius): # Tested Correctly
        if nearby_cells == []: raise ValueError("No Nearby Cells Are Found")
        # Define the central cell and the radius (assuming cell_width_x = cell_width_y)
        plotting_width_x, plotting_width_y = 1, 1
        # Step 1: Create polygons for each nearby cell
        def create_cell_polygon(x, y, width, height):
            """Creates a rectangular polygon for the cell located at (x, y) with the given width and height."""
            return Polygon([
                (x, y),  # Bottom-left corner
                (x + width, y),  # Bottom-right corner
                (x + width, y + height),  # Top-right corner
                (x, y + height),  # Top-left corner
                (x, y)  # Closing the polygon (back to bottom-left)
            ])

        # Step 2: Convert the nearby cells to GeoDataFrame with rectangular polygons
        polygons = [create_cell_polygon(x, y, plotting_width_x, plotting_width_y) for x, y in nearby_cells]
        polygons += [create_cell_polygon(central_cell[0], central_cell[1], plotting_width_x, plotting_width_y)]
        gdf = gpd.GeoDataFrame(geometry=polygons) #, crs="EPSG:4326")

        # Step 3: Plot the nearby cells
        fig, ax = plt.subplots()
        gdf[:-1].plot(ax=ax, color='lightblue', edgecolor='black')
        gdf.tail(1).plot(ax=ax, color='#FFCCCC', edgecolor='black')
        # Step 4: Plot the central cell and the radius circle
        central_x, central_y = central_cell
        adjusted_raduis = radius / 5
        circle = patches.Circle((central_x + 0.5, central_y + 0.5), radius=adjusted_raduis, edgecolor='red', facecolor='none')

        # Add the circle to the plot
        ax.add_patch(circle)

        # Optionally, you can add labels to the cells to indicate (x, y) coordinates
        for x, y in nearby_cells:
            plt.text(x + 0.5, y + 0.5, f"({x},{y})", ha='center', va='center', fontsize=8)

        plt.title("Nearby Cells and Radius Visualization")
        plt.grid(True)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def get_random_id(self):
        return random.choice(list(self.unique_ids))
    
    def get_cell_ids(self):
        return list(self.unique_ids)