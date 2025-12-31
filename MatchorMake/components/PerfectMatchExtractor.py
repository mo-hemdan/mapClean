import pandas as pd
import copy
import pickle
from tqdm import tqdm

class PerfectMatchExtractor():
    def __init__(self, delta, beta, prefix):
        self.delta = delta
        self.beta = beta
        self.prefix = prefix
    
    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        
    def set_prefix(self, prefix):
        self.prefix = prefix

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)  
    
    def extract_perfect_instances(self, gdf, verbose=True):
        '''
        Extracting Perfect Instances & Uncertain ones. Keep in mind this function duplicates the 
        '''
        perfectly_matched_points = copy.deepcopy(gdf[gdf[self.prefix + 'distance_to_matched_road'] <= self.delta])
        uncertain_points = copy.deepcopy(gdf[gdf[self.prefix + 'distance_to_matched_road'] > self.delta])
        
        if verbose: print('Extracting perfect roads..') 
        # Calculating the number of points matching to each road (for both perfect points and uncertain points)
        road_to_perfect_points_count = dict(perfectly_matched_points[self.prefix+'matched_road_id'].value_counts())
        road_to_uncertain_points_count = dict(uncertain_points[self.prefix+'matched_road_id'].value_counts())
        
        candidate_perfect_roads = set(road_to_perfect_points_count.keys())
        perfect_roads = []
        for r in tqdm(candidate_perfect_roads, desc='Getting Perfect Roads', total=len(candidate_perfect_roads)):
            
            # Calculating the number of perfect points matching to this road
            NP = road_to_perfect_points_count[r] # Appreviated for Number of Perfect points (NP)
            
            # Calculating the total number of matched points to that road
            NM = NP
            if r in road_to_uncertain_points_count.keys():    NM += road_to_uncertain_points_count[r] 
            
            # Calculating the Score
            score = NP / NM
            if score >= self.beta:
                perfect_roads.append(r)
        
        # Step 4: Choose the Perfect points that match to the roads
        if verbose: print('Updating perfect points...')
        perfect_points = perfectly_matched_points[perfectly_matched_points[self.prefix+'matched_road_id'].isin(perfect_roads)]
        uncertain_points = pd.concat([uncertain_points, perfectly_matched_points[~ perfectly_matched_points[self.prefix+'matched_road_id'].isin(perfect_roads)]])
        
        if verbose:
            print("Total Number of Points: ", len(gdf))
            print("   Number of Perfectly-Matched Points: ", len(perfectly_matched_points))
            print("   Uncertain Points (inc. None) (", 100*len(uncertain_points)/len(gdf), "%): ", len(uncertain_points))
            print("   None-Matched Points (", 100*sum(gdf[self.prefix+'matched_road_id'].isna()) / len(gdf), "%): ", sum(gdf[self.prefix+'matched_road_id'].isna()))
            print("   Perfect Roads: ", len(perfect_roads))
        
        return perfect_points, uncertain_points, perfect_roads
    
    def extract_perfect_instances_optimized(self, gdf, distance_to_matched_road_col_name, verbose=True):
        '''
        Extracting Perfect Instances & Uncertain ones. Keep in mind this function duplicates the 
        '''
        
        # Extract perfect and uncertain points based on the distance threshold
        if verbose: print('Extracting perfect and uncertain points...')
        perfectly_matched_points = gdf[gdf[distance_to_matched_road_col_name] <= self.delta]
        uncertain_points = gdf[gdf[distance_to_matched_road_col_name] > self.delta]
        
        # Count the number of points per road for perfect and uncertain points
        if verbose: print('Calculating road match counts...')
        perfect_counts = perfectly_matched_points['matched_road_id'].value_counts()
        uncertain_counts = uncertain_points['matched_road_id'].value_counts()

        # Combine perfect and uncertain counts, then compute the score
        if verbose: print('Computing scores...')
        combined_counts = perfect_counts.add(uncertain_counts, fill_value=0)
        scores = (perfect_counts / combined_counts).fillna(0)
        
        # Select roads where the score exceeds beta threshold
        perfect_roads = scores[scores >= self.beta].index
        
        # Filter points that match the perfect roads
        if verbose: print('Filtering perfect points...')
        perfect_points = perfectly_matched_points[perfectly_matched_points['matched_road_id'].isin(perfect_roads)]
        
        # Move non-perfect points to uncertain points
        non_perfect_points = perfectly_matched_points[~perfectly_matched_points['matched_road_id'].isin(perfect_roads)]
        uncertain_points = pd.concat([uncertain_points, non_perfect_points])
        
        if verbose:
            print("Total Number of Points: ", len(gdf))
            print("   Number of Perfectly-Matched Points: ", len(perfectly_matched_points))
            print("   Uncertain Points (inc. None) (", 100*len(uncertain_points)/len(gdf), "%): ", len(uncertain_points))
            print("   None-Matched Points (", 100*sum(gdf[self.prefix+'matched_road_id'].isna()) / len(gdf), "%): ", sum(gdf[self.prefix+'matched_road_id'].isna()))
            print("   Perfect Roads: ", len(perfect_roads), ' out of ')
        
        return perfect_points, uncertain_points, perfect_roads
    
# def perfect_matching_extraction(gdf, delta=40, beta=0.6, prefix='', verbose=True):
#     # TODO: the Matching score is chosen to be the distance since LCSS is a distance-based map matching technique
#     """
#     Do Map Matching over the data (using LCSS Map Matching technique) and get the perfectly matching points and the uncertain points
#     Note: the GeoPandas DataFrame is updated by reference
#     Args:
#         gdf: [GeoPandas DataFrame] the dataframe that contains the points
#     """
#     # NOTE: Map Matching is removed to be a separate stage

#     print('Step 2: Extracting perfect points and uncertain points..!')
#     # Step 2: Extract Perfectly Matched Points and Uncertain Points
#     perfectly_matched_points = copy.deepcopy(gdf[gdf.distance_to_matched_road <= delta])
#     uncertain_points = copy.deepcopy(gdf[gdf.distance_to_matched_road > delta])

#     print('Step 3: Extracting perfect roads..!')
#     # Step 3: Extract Perfect Roads from the points based on the parameter beta
#     perfect_road_count = dict(perfectly_matched_points.matched_road_id.value_counts())
#     uncertain_road_count = dict(uncertain_points.matched_road_id.value_counts())
#     roads = set(perfect_road_count.keys())
#     road_score = dict()
#     perfect_roads = []
#     for r in roads:
#         good_points = 0
#         if r in perfect_road_count.keys():
#             good_points = perfect_road_count[r]
#         bad_points = 1
#         if r in uncertain_road_count.keys():
#             bad_points = uncertain_road_count[r]
#         score = good_points / (good_points + bad_points)
#         road_score[r] = score
#         if score >= beta:
#             perfect_roads.append(r)
    
#     # Step 4: Choose the Perfect points that match to the roads
#     print('Step 4: Updating perfect points...!')
#     perfect_points = perfectly_matched_points[perfectly_matched_points.matched_road_id.isin(perfect_roads)]
#     uncertain_points = pd.concat([uncertain_points, perfectly_matched_points[~ perfectly_matched_points.matched_road_id.isin(perfect_roads)]])

#     if verbose:
#         print("Total Number of Points: ", len(gdf))
#         print("   Number of Perfectly-Matched Points: ", len(perfectly_matched_points))
#         print("   Uncertain Points (inc. None) (", 100*len(uncertain_points)/len(gdf), "%): ", len(uncertain_points))
#         print("   None-Matched Points (", 100*sum(gdf[prefix+'matched_road_id'].isna()) / len(gdf), "%): ", sum(gdf[prefix+'matched_road_id'].isna()))
#         print("   Perfect Roads: ", len(perfect_roads))
#     return perfect_points, uncertain_points, perfect_roads