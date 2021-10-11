import json
import os
from Data.VolumeDesignGraph import VolumeDesignGraph
import torch
from collections import Counter


class GraphConstructor:
    number_of_class = 6
    dimension_norm_factor = 100

    global_graph_prefix = "graph_global_"
    local_graph_prefix = "graph_local_"
    voxel_graph_prefix = "voxel_"
    data_id_length = 6

    def __init__(self, ):
        pass

    @staticmethod
    def get_program_unicode(n):
        return n["floor"], n["type"], n["type_id"]

    @staticmethod
    def one_hot_vector(_id):
        assert _id < GraphConstructor.number_of_class
        return [1 if i == _id else 0 for i in range(GraphConstructor.number_of_class)]

    @staticmethod
    def load_graph_jsons(data_id, data_dir):
        data_id_str = str(data_id).zfill(GraphConstructor.data_id_length)
        with open(os.path.join(data_dir, "global_graph_data", GraphConstructor.global_graph_prefix + data_id_str+".json")) as f:
            global_graph = json.load(f)  # Keys: FAR, site area, Global node (Room Type, target_ratio, connections)
        with open(os.path.join(data_dir, "local_graph_data", GraphConstructor.local_graph_prefix + data_id_str+".json")) as f:
            local_graph = json.load(f)
        with open(os.path.join(data_dir, "voxel_data", GraphConstructor.voxel_graph_prefix + data_id_str+".json")) as f:
            raw_voxel_graph = json.load(f)

        program_graph = GraphConstructor.construct_program_graph(local_graph, global_graph)
        voxel_graph = GraphConstructor.construct_voxel_graph(raw_voxel_graph)

        return GraphConstructor.tensorfy(program_graph, voxel_graph, data_id_str)

    @staticmethod  # not used
    def load_graph_from_UI(global_graph, local_graph, raw_voxel_graph, data_id):
        data_id_str = str(data_id).zfill(GraphConstructor.data_id_length)

        program_graph = GraphConstructor.construct_program_graph(local_graph, global_graph)
        voxel_graph = GraphConstructor.construct_voxel_graph(raw_voxel_graph)

        return GraphConstructor.tensorfy(program_graph, voxel_graph, data_id_str)

    @staticmethod
    def tensorfy(program_grpah, voxel_graph, data_id_str):
        program_floor_index = torch.tensor(program_grpah["program_floor_index"], dtype=torch.long)
        story_level_feature = torch.FloatTensor(program_grpah["story_level_feature"])                   # N x 1
        program_class_feature = torch.FloatTensor(program_grpah["program_class_feature"])
        program_edge = torch.tensor(program_grpah["program_edge"], dtype=torch.long)
        neg_same_story_program_edge = torch.tensor(program_grpah["neg_same_story_program_edge"], dtype=torch.long)
        neg_diff_story_program_edge = torch.tensor(program_grpah["neg_diff_story_program_edge"], dtype=torch.long)
        program_class_cluster = torch.tensor(program_grpah["program_class_cluster"], dtype=torch.long)
        program_target_ratio = torch.FloatTensor(program_grpah["program_target_ratio"])
        FAR = torch.FloatTensor([program_grpah["FAR"]])
        voxel_floor_index = torch.tensor(voxel_graph["voxel_floor_index"], dtype=torch.long)
        voxel_projection_cluster = torch.tensor(voxel_graph["voxel_projection_cluster"], dtype=torch.long)
        voxel_feature = torch.FloatTensor(voxel_graph["voxel_feature"])
        voxel_bool = torch.FloatTensor(voxel_graph["voxel_bool"])
        voxel_label = torch.FloatTensor(voxel_graph["voxel_label"])
        voxel_edge = torch.tensor(voxel_graph["voxel_edge"], dtype=torch.long)

        # add index_select for cross-edges
        pfc = program_grpah["program_floor_index"]
        program_node_id_in_story_group = []
        voxel_node_counter = Counter(voxel_graph["voxel_floor_index"])
        for program_node_id, story_id in enumerate(pfc):
            if program_node_id-1 >= 0 and story_id == pfc[program_node_id-1]:
                program_node_id_in_story_group[-1].append(program_node_id)
            else:
                program_node_id_in_story_group.append([program_node_id])

        cross_edge_program_index_select, cross_edge_voxel_index_select, voxel_id = [], [], 0
        for global_story_id, program_node_id_in_this_story_group in enumerate(program_node_id_in_story_group):
            number_of_voxel_in_this_story = voxel_node_counter[global_story_id]
            number_of_program_in_this_story = len(program_node_id_in_this_story_group)
            cross_edge_program_index_select += program_node_id_in_this_story_group * number_of_voxel_in_this_story
            for _ in range(number_of_voxel_in_this_story):
                cross_edge_voxel_index_select += [voxel_id] * number_of_program_in_this_story
                voxel_id += 1

        assert len(cross_edge_program_index_select) == len(cross_edge_voxel_index_select)

        cross_edge_program_index_select = torch.tensor(cross_edge_program_index_select, dtype=torch.long)
        cross_edge_voxel_index_select = torch.tensor(cross_edge_voxel_index_select, dtype=torch.long)

        # add label_program_index. This is for the discriminator to point the voxel label back to its corresponding program node
        voxel_label_program_node_index = []

        for voxel_id, story_id in enumerate(voxel_graph["voxel_floor_index"]):
            program_unicode = (story_id, ) + voxel_graph["node_id_to_program_unicode"][voxel_id]  # (floor, -1, 0) if type=-1
            program_node_id = program_grpah["unicode_to_node_id_map"].get(program_unicode, -1)
            if program_node_id != -1:
                assert(program_floor_index[program_node_id]) == story_id
            voxel_label_program_node_index.append(program_node_id)

        voxel_label_program_node_index = torch.tensor(voxel_label_program_node_index, dtype=torch.long)

        # add voxel_edge_mask to extract edges across same story in voxel graph. This is not used.
        voxel_edge_mask = []
        for p1, p2 in zip(voxel_edge[0], voxel_edge[1]):
            voxel_edge_mask.append(1.0 if voxel_graph["voxel_floor_index"][p1] == voxel_graph["voxel_floor_index"][p2] else 0.0)
        voxel_edge_mask = torch.FloatTensor(voxel_edge_mask)

        graph = VolumeDesignGraph(data_id_str, program_floor_index, story_level_feature, program_class_feature,
                                  program_edge, neg_same_story_program_edge, neg_diff_story_program_edge, program_class_cluster, program_target_ratio, FAR,
                                  voxel_floor_index, voxel_projection_cluster, voxel_feature, voxel_bool, voxel_label,
                                  voxel_edge, voxel_edge_mask, cross_edge_program_index_select, cross_edge_voxel_index_select, voxel_label_program_node_index)
        return graph

    @staticmethod
    def construct_voxel_graph(voxel_graph):
        # Extract Node Feature
        unicode_to_node_id_map, node_id_to_program_unicode, projection_map = {}, {}, {}
        voxel_floor_index, voxel_feature, voxel_bool, voxel_label = [], [], [], []
        for i, n in enumerate(voxel_graph["voxel_node"]):
            voxel_bool.append(0 if n["type"] < 0 else 1)
            voxel_label.append(GraphConstructor.one_hot_vector(n["type"]))
            voxel_feature.append([*[x / GraphConstructor.dimension_norm_factor for x in n["dimension"] + n["coordinate"]], n["weight"]])
            unicode_to_node_id_map[tuple(n["location"])] = i
            node_id_to_program_unicode[i] = (n["type"], n["type_id"])
            voxel_floor_index.append(n["location"][0])

            horizontal_location = (n["location"][2], n["location"][1])  # x, y
            if projection_map.get(horizontal_location, None):
                projection_map.get(horizontal_location).append(i)
            else:
                projection_map[horizontal_location] = [i]

        voxel_projection_cluster = [-1] * len(voxel_graph["voxel_node"])
        for i, v in enumerate(projection_map.values()):
            for n_i in v:
                voxel_projection_cluster[n_i] = i
        assert(-1 not in voxel_projection_cluster)

        # Extract Edge Feature <- Already bi-directional
        voxel_edge_src, voxel_edge_dst = [], []
        for i, n_i in enumerate(voxel_graph["voxel_node"]):
            for unicode_j in n_i["neighbors"]:
                voxel_edge_src.append(i)
                voxel_edge_dst.append(unicode_to_node_id_map[tuple(unicode_j)])
        voxel_edge = [voxel_edge_src, voxel_edge_dst]

        return {
            "voxel_floor_index": voxel_floor_index,     # dict (F --> nodes in each floor)
            "voxel_projection_cluster": voxel_projection_cluster,  # N x 1
            "voxel_feature": voxel_feature,                 # N x 7
            "voxel_bool": voxel_bool,                       # N x 1
            "voxel_label": voxel_label,                     # N x C
            "voxel_edge": voxel_edge,                       # 2 x E
            "node_id_to_program_unicode": node_id_to_program_unicode  # voxel node id -> type, type id
        }

    @staticmethod
    def construct_program_graph(local_graph, global_graph):
        # Extract Node Feature
        unicode_to_node_id_map = {}
        program_floor_index, story_level_feature, program_class_feature, program_class_cluster = [], [], [], []
        for i, n in enumerate(local_graph["node"]):
            # n is a dict. Keys: floor, type, type_id
            unicode = GraphConstructor.get_program_unicode(n)
            unicode_to_node_id_map[unicode] = i
            # n["region_far"] and n["center"] are not used

            story_level_feature.append(n["floor"])
            program_class_feature.append(GraphConstructor.one_hot_vector(n["type"]))
            program_class_cluster.append(n["type"])
            program_floor_index.append(n["floor"])

        # Extract Edge Feature <- Already bi-directional
        program_edge_src, program_edge_dst, adj_matrix = [], [], [[0 for _ in range(len(local_graph["node"]))] for _ in range(len(local_graph["node"]))]
        for i, n_i in enumerate(local_graph["node"]):
            for unicode_j in n_i["neighbors"]:
                program_edge_src.append(i)
                program_edge_dst.append(unicode_to_node_id_map[tuple(unicode_j)])
                adj_matrix[i][unicode_to_node_id_map[tuple(unicode_j)]] = 1
        program_edge = [program_edge_src, program_edge_dst]

        # Construct Negative edges on the same story
        neg_same_story_program_edge_src, neg_same_story_program_edge_dst, neg_diff_story_program_edge_src, neg_diff_story_program_edge_dst = [], [], [], []
        for i in range(len(local_graph["node"])):
            for j in range(i+1, len(local_graph["node"])):
                if adj_matrix[i][j] == 1:
                    continue
                if story_level_feature[i] == story_level_feature[j]:
                    neg_same_story_program_edge_src += [i, j]
                    neg_same_story_program_edge_dst += [j, i]
                else:
                    neg_diff_story_program_edge_src += [i, j]
                    neg_diff_story_program_edge_dst += [j, i]
        neg_same_story_program_edge = [neg_same_story_program_edge_src, neg_same_story_program_edge_dst]
        neg_diff_story_program_edge = [neg_diff_story_program_edge_src, neg_diff_story_program_edge_dst]

        # Global Feature and Edge
        program_target_ratio = [0] * GraphConstructor.number_of_class
        FAR = global_graph["far"]
        for c in global_graph["global_node"]:
            program_target_ratio[c["type"]] = c["proportion"]

        return {
            "program_floor_index": program_floor_index,     # N x 1
            "story_level_feature": story_level_feature,         # N x 1
            "program_class_feature": program_class_feature,     # N x C
            "program_edge": program_edge,                       # 2 x E
            "neg_same_story_program_edge": neg_same_story_program_edge,  # 2 x E
            "neg_diff_story_program_edge": neg_diff_story_program_edge,  # 2 x E
            "program_class_cluster": program_class_cluster,     # N x 1
            "program_target_ratio": program_target_ratio,       # C
            "FAR": FAR,                                         # 1
            "unicode_to_node_id_map": unicode_to_node_id_map    # floor, type, type_id -> program node id
        }

