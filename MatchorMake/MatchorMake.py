from . import SpatialQueryExecutor, MapMatcher, PerfectMatchExtractor, ErrorInjector, FeatureExtractor, PointClassificationModels
from .DataHandler import combine_uncertain_and_perfect, split_train_test, get_train_from_split, get_test_from_split


class MatchorMake():
    def __init__(self):
        self.query_executor = None
        self.map_matcher = None
        self.perfect_match_extractor = None
        self.error_injector = None
        self.feature_extractor = None
        self.classification_model = None
        pass
    
    def initialize_components(self, query_executor, map_matcher, perfect_match_extractor, error_injector, feature_extractor, classification_model):
        self.query_executor = query_executor
        self.map_matcher = map_matcher
        self.perfect_match_extractor = perfect_match_extractor
        self.error_injector = error_injector
        self.feature_extractor = feature_extractor
        self.classification_model = classification_model
    
    def build_index(self, gdf, cell_width):
        self.query_executor.build_index(
            gdf=gdf, 
            cell_width=cell_width
            )
        return gdf
    
    def preQuery_process(self, raduis):
        self.query_executor.construct_Nearby_cells(raduis)
    
    def save_index(self, directory, prefix):
        self.query_executor.save(directory+'/'+prefix+'query_executor.pkl')
        
    def load_index(self, directory, prefix): 
        self.query_executor = SpatialQueryExecutor.load(directory+'/'+prefix+'query_executor.pkl')
        
    def train(self, gdf, road_network, apply_post_process=False):
        # Map Matching the Internal one
        gdf['distance_to_matched_road'] = gdf['ture_distance_to_matched_road']
        gdf['matched_road_geom'] = gdf['ture_matched_road_geom']
        gdf['matched_road_id'] = gdf['ture_matched_road_id']
        gdf['matched_road_metadata'] = gdf['ture_matched_road_metadata']

        # Perfect Match Extraction
        self.perfect_match_extractor.set_prefix('')
        perfect_points, uncertain_points, perfect_roads = self.perfect_match_extractor.extract_perfect_instances(gdf, verbose=True)

        # Error Injection
        self.map_matcher.set_prefix('')
        perfect_points, insideSystem_edgesRemoved_nx_map, roads_to_removed, g_removed_edges, p_make_percentage = self.error_injector.run(perfect_points, road_network, False, perfect_roads, self.query_executor.partitioner)
        perfect_points = self.map_matcher.match(perfect_points, insideSystem_edgesRemoved_nx_map)

        # Adjust the query result
        point_Ids = list(perfect_points[perfect_points.isAddedNoise].index)
        self.query_executor.adjust_query_results(perfect_points, point_Ids)

        # Combining both
        total_points = combine_uncertain_and_perfect(uncertain_points, perfect_points)

        # Run Feature Extraction
        point_Ids = list(total_points[total_points.Perfect].index)
        total_points = self.feature_extractor.extract_features(total_points, self.query_executor, point_Ids)

        # Splitting data for Model run
        feature_names = self.feature_extractor.get_feature_names()
        X_train, y_train = get_train_from_split(total_points, feature_names, label_name='toBeMatched')

        # Model run
        self.classification_model.initialize_models(X_train.shape[1])
        self.classification_model.adjust_scaler(X_train)
        self.classification_model.fit(X_train, y_train)

        return self.classification_model.evaluate(X_train, y_train, apply_post_process), insideSystem_edgesRemoved_nx_map, total_points, self.classification_model.predict(X_train, apply_post_process), X_train, y_train, roads_to_removed, g_removed_edges, p_make_percentage
    
    def inject_error(self, gdf, road_network, error_injector):  #TODO: Test this function. It seems not working for some reason.
        
        # Perfect Match Extraction
        self.perfect_match_extractor.set_prefix('ture_')
        _, _, perfect_roads = self.perfect_match_extractor.extract_perfect_instances(gdf, verbose=True)
        
        gdf, new_nx_map, roads_to_removed, g_removed_edges, p_make_percentage = error_injector.run(gdf, road_network, False, perfect_roads, self.query_executor.partitioner)
        self.map_matcher.set_prefix('ture_')
        gdf = self.map_matcher.match(gdf, new_nx_map)
        
        # Adjust the query result
        point_Ids = list(gdf[gdf.ture_isAddedNoise].index)
        self.query_executor.adjust_query_results(gdf, point_Ids)
        return gdf, new_nx_map, roads_to_removed, g_removed_edges, p_make_percentage
    
    def test(self, gdf, apply_post_process=False):
        # Perfect Match Extraction
        self.perfect_match_extractor.set_prefix('ture_')
        perfect_points, uncertain_points, _ = self.perfect_match_extractor.extract_perfect_instances(gdf, verbose=True)
        # Combining both
        total_points = combine_uncertain_and_perfect(uncertain_points, perfect_points)
        # Run Feature Extraction
        point_Ids = list(uncertain_points.index)
        
        self.feature_extractor.set_prefix('ture_')
        total_points = self.feature_extractor.extract_features(total_points, self.query_executor, point_Ids)

        # MAINLY BECAUSE OF LABEL CHANGES
        self.feature_extractor.set_prefix('')
        feature_names = self.feature_extractor.get_feature_names()
        # Editing the feature extractor way of extracting
        if 'distance_to_matched_road' not in total_points.columns:
            total_points.rename(columns={'ture_distance_to_matched_road': 'distance_to_matched_road'}, inplace=True)
            
        X_test, y_test = get_test_from_split(total_points, feature_names, label_name='ture_toBeMatched')
        
        predict_df = self.classification_model.predict(X_test, apply_post_process)

        return self.classification_model.evaluate(X_test, y_test, apply_post_process), predict_df, X_test, y_test
    
    def predict(self, X_test):
        return self.classification_model.predict(X_test)
    
    def save(self, directory, prefix=''):
        # self.query_executor.save(directory+'/'+prefix+'query_executor.pkl')
        self.map_matcher.save(directory+'/'+prefix+'map_matcher.pkl')
        self.perfect_match_extractor.save(directory+'/'+prefix+'perfect_match_extractor.pkl')
        self.error_injector.save(directory+'/'+prefix+'error_injection.pkl')
        self.feature_extractor.save(directory+'/'+prefix+'feature_extractor.pkl')
        self.classification_model.save(directory+'/'+prefix+'classification_model.pkl')
    
    def load(self, directory, prefix=''):
        # self.query_executor = SpatialQueryExecutor.load(directory+'/'+prefix+'query_executor.pkl')
        self.map_matcher = MapMatcher.load(directory+'/'+prefix+'map_matcher.pkl')
        self.perfect_match_extractor = PerfectMatchExtractor.load(directory+'/'+prefix+'perfect_match_extractor.pkl')
        self.error_injector = ErrorInjector.load(directory+'/'+prefix+'error_injection.pkl')
        self.feature_extractor = FeatureExtractor.load(directory+'/'+prefix+'feature_extractor.pkl')
        self.classification_model = PointClassificationModels.load(directory+'/'+prefix+'classification_model.pkl')