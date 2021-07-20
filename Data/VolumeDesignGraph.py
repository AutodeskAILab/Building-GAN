from torch_geometric.data import Data


class VolumeDesignGraph(Data):
    def __init__(self, data_id_str=None, program_floor_cluster=None, story_level_feature=None, program_class_feature=None,
                 program_edge=None, neg_same_story_program_edge=None, neg_diff_story_program_edge=None, program_class_cluster=None, program_target_ratio=None, FAR=None,
                 voxel_floor_cluster=None, voxel_projection_cluster=None, voxel_feature=None, voxel_bool=None, voxel_label=None,
                 voxel_edge=None, voxel_edge_mask=None, cross_edge_program_index_select=None, cross_edge_voxel_index_select=None, voxel_label_program_node_index=None):
        """
        data_id_str: (str) data ID

        program_edge: (e_program x 2) program edges (include same_story and diff_story)
        story_level_feature: (n_program_node x 1) the story level for each program nodes
        program_class_feature: (n_program_node x d) one-hot vector of program type
        program_class_cluster: (n_program_type x n_element) list of program node IDs on the same program type
        program_floor_cluster: (n_cluster x n_element) list of program node IDs on the same story
        program_target_ratio: (n_program_type x 1) the target ratio between all program types
        FAR: (1) target floor area ratio

        voxel_feature: (n_voxel x d) voxel feature
        voxel_floor_cluster: (n_cluster x n_element) list of voxel node IDs on the same story
        voxel_projection_cluster:　(n_cluster x n_element) list of voxel node IDs on the same x,y coordinates
        voxel_bool: (n_voxel x 1) if this voxel is void
        voxel_label: (n_voxel x program_type) which program type is this voxel
        voxel_edge: (e_voxel x 2) voxel edges
        voxel_edge_mask: (e_voxel x 1) Not used! if the two voxel nodes on the voxel edge is on the same story

        cross_edge_program_index_select: (e_cross x 1) the program node IDs of the cross edges
        cross_edge_voxel_index_select: (e_cross x 1) the voxel node IDS of the cross edges
        voxel_label_program_node_index: (n_voxel x 1) which program node ID is assigned to this voxel
        """

        super(VolumeDesignGraph, self).__init__()
        self.data_id_str = data_id_str
        self.program_floor_cluster = program_floor_cluster
        self.story_level_feature, self.program_class_feature = story_level_feature, program_class_feature
        self.program_edge, self.neg_same_story_program_edge, self.neg_diff_story_program_edge = program_edge, neg_same_story_program_edge, neg_diff_story_program_edge
        self.program_class_cluster = program_class_cluster
        self.program_target_ratio = program_target_ratio
        self.FAR = FAR

        self.voxel_floor_cluster = voxel_floor_cluster
        self.voxel_projection_cluster = voxel_projection_cluster
        self.voxel_feature = voxel_feature
        self.voxel_bool, self.voxel_label = voxel_bool, voxel_label
        self.voxel_edge = voxel_edge
        self.voxel_edge_mask = voxel_edge_mask

        self.cross_edge_program_index_select = cross_edge_program_index_select
        self.cross_edge_voxel_index_select = cross_edge_voxel_index_select
        self.voxel_label_program_node_index = voxel_label_program_node_index

        self._batch_program_inc_keys = ["program_edge", "cross_edge_program_index_select", "voxel_label_program_node_index", "neg_same_story_program_edge", "neg_diff_story_program_edge"]
        self._batch_voxel_inc_keys = ["voxel_edge", "cross_edge_voxel_index_select"]
        self._floor_inc_keys = ["program_floor_cluster", "voxel_floor_cluster"]
        self._program_class_inc_keys = ["program_class_cluster"]
        self._voxel_projection_inc_keys = ["voxel_projection_cluster"]
        self._cat_dim_keys = ["program_edge", "voxel_edge", "neg_same_story_program_edge", "neg_diff_story_program_edge"]

    def __inc__(self, key, value):
        if key in self._batch_program_inc_keys:
            return self.program_class_feature.size(0)           # number of edges in program graph
        if key in self._batch_voxel_inc_keys:
            return self.voxel_feature.size(0)                   # number of nodes in voxel graph
        if key in self._floor_inc_keys:
            return int(max(self.story_level_feature) + 1)       # number of stories in each volume design graph
        if key in self._program_class_inc_keys:
            return self.program_class_feature.size(1)           # number of program class
        if key in self._voxel_projection_inc_keys:
            return int(max(self.voxel_projection_cluster) + 1)  # number of voxels on XY plane
        else:
            return super(VolumeDesignGraph, self).__inc__(key, value)

    def __cat_dim__(self, key, value):
        # `*index*` and `*face*` should be concatenated in the last dimension,
        # everything else in the first dimension.
        return -1 if key in self._cat_dim_keys else 0



