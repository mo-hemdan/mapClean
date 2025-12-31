from .Partitioner import Partitioner
from tqdm import tqdm
import pickle
import geopandas as gpd

class SpatialQueryExecutor():
    def __init__(self):
        self.partitioner = Partitioner()
        self.nearby_cells_dict = None
        self.cell_to_points = None
        self.radius = None
        self.index_built = False
    
    def reset(self):
        self.partitioner.reset()
        self.nearby_cells_dict = None
        self.cell_to_points = None
        self.radius = None
        self.index_built = False
    
    def build_index(self, gdf, cell_width):
        self.partition_space(gdf, cell_width)
        self.construct_cell2point_Map(gdf)
        
    def partition_space(self, gdf, cell_width): # This function may need to be dealing with disk
        self.partitioner.partition(gdf, cell_width) 
        self.partitioner.calculate_ids(gdf) # Takes O(len(gdf))
        self.index_built = False
    
    def construct_cell2point_Map(self, gdf):
        '''
        Get the nearby points per point through two main data structures. First is the nearby_cells_dict which contains 
        the nearby cells for each cell in our dataset, then comes the cell_to_points which stores the points in each cell
        To get the nearby points for a specific point. You do the following mapping steps
        1. Get cell_id from point dataframe
        3. Get points in those nearby cells using the cell_to_points dictionary
        '''
        # self.cell_to_points = gdf.groupby('cell_id')['point_id'].apply(list).to_dict()
        self.cell_to_points = gdf.groupby('cell_id').apply(lambda x: x.index.tolist()).to_dict()
        self.index_built = True
        return self.cell_to_points
    
    def construct_Nearby_cells(self, radius):
        self.radius = radius
        self.nearby_cells_dict = {} 
        # loop over the points one by one
        for cell_id in tqdm(self.partitioner.unique_ids, total=len(self.partitioner.unique_ids), desc='Getting Nearby Cells per cell'): # Takes O(cells_keys * O(getting nearby cells))
            cell_ids = self.partitioner.get_Nearbycells_per_cell(cell_id, self.radius)
            self.nearby_cells_dict[cell_id] = cell_ids
    
    def get_nearby_cells(self, cell_id):
        '''
        Get the nearby Cells for a cell. The index has to be built first
        '''
        self.make_sure_nearbyQuery_executed()
        if not self.partitioner.cell_id_exists(cell_id): raise ValueError('Cell doesn\'t exist')
        return self.nearby_cells_dict[cell_id]
    
    def get_nearbyPoints_for_cell(self, cell_id):
        '''
        Getting Nearby Points for a cell
        '''
        nearby_cells = self.get_nearby_cells(cell_id)
        nearby_points = []
        for cell in nearby_cells:
            points = self.get_points_in_cell(cell)
            nearby_points.extend(points)
        return nearby_points

    # def get_OptimizedNearbyPoints_for_cell(self, raduis, cell_id, cell_to_points):
    #     '''
    #     Getting Nearby Points for a cell
    #     '''
    #     nearby_cells = self.partitioner.get_Nearbycells_per_cell(cell_id, raduis)
    #     nearby_points = []
    #     for cell in nearby_cells:
    #         points = cell_to_points[cell_id]
    #         nearby_points.extend(points)
    #     return nearby_points
    
    def get_points_in_cell(self, cell_id):
        return self.cell_to_points[cell_id]
    
    def adjust_query_results(self, gdf, points_idxs, supress_warning=False): #: TODO: To support adding custom data  TODO: trade-off optimization when it comes to how much shall we support until we rerun the query again
        '''
        Adjusting the query results we had previously
        Complexity: 
        Returns the number of changed points
        '''
        self.make_sure_nearbyQuery_executed()
        
        n_changes = 0
        n_new_cells = 0
        for point_idx in tqdm(points_idxs, total=len(points_idxs), desc="Updating Data structures"):
            # row = gdf[gdf.point_id == point_id]
            row = gdf.loc[point_idx]
            if isinstance(row, gpd.GeoDataFrame):
                raise Exception('There are two points with the same cell id')
            
            cell_id = self.partitioner.calc_cell_id(row.geometry)
            
            if cell_id == row.cell_id:
                continue  # No need to change anything
            
            n_changes += 1
            
            # Case (2) Point is in different Cell: 
            # First change there cell_id, 
            # TODO: This has to change so that point_id is the index itself
            gdf.at[point_idx, 'cell_id'] = cell_id
            # gdf.at[gdf[gdf['point_id'] == point_id].index[0], 'cell_id'] = cell_id

            # Second, remove it from the current cell, 
            self.cell_to_points[row.cell_id].remove(point_idx)
            # TODO: Remove the cells with no points if possible

            if not self.partitioner.cell_id_exists(cell_id):
                self.add_new_cell(cell_id)  # Add the new cell that will update the data structures we have for cells
                n_new_cells += 1

            # Add the new point
            self.cell_to_points[cell_id].append(point_idx)
        if not supress_warning: print(f'NOTE: {n_changes} changes, {n_new_cells} new cells have been added')
        return n_changes

    def add_new_cell(self, cell_id):
        '''
        Adding a new cell to the partitioner and updating the Nearby Cells Dictionary and initialize Cell_to_points to empty list. It doesn't run if there is no query run before
        '''
        self.make_sure_nearbyQuery_executed()
            
        if self.partitioner.cell_id_exists(cell_id):
            raise ValueError("Partitioner already has this cell_id, and by inference the SpatialExecutor should have it")
        
        # Add the cell to the Partitioner
        self.partitioner.add_new_cell(cell_id)

        # Update the cell to points Dictionary
        self.cell_to_points[cell_id] = []

        # Update the Nearby Cells Dictionary
        nearby_cells = self.partitioner.get_Nearbycells_per_cell(cell_id, self.radius)
        self.nearby_cells_dict[cell_id] = nearby_cells
        for id in nearby_cells: self.nearby_cells_dict[id].append(cell_id)
    
    def plot_nearby_cells(self, cell_id):
        self.make_sure_nearbyQuery_executed()

        if not self.partitioner.cell_id_exists(cell_id):
            raise ValueError('Cell Id doesn\'t exist')
        
        self.partitioner.plot_nearby_cells_results(self.nearby_cells_dict[cell_id], cell_id, self.radius)
    
    def make_sure_nearbyQuery_executed(self):
        if self.index_built and (self.nearby_cells_dict is not None) and (self.cell_to_points is not None):
            pass
        else:
            raise ValueError("No Range Query has been run. Make sure to run nearby first")
        
    def get_random_id(self):
        return self.partitioner.get_random_id()
    
    def get_cell_ids(self):
        return self.partitioner.get_cell_ids()

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)        