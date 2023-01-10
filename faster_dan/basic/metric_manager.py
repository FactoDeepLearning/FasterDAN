from faster_dan.Datasets.dataset_formatters.rimes_formatter import SEM_TOKENS as RIMES_TOKENS
from faster_dan.Datasets.dataset_formatters.read2016_formatter import SEM_TOKENS as READ_TOKENS
import re
import networkx as nx
import editdistance
import numpy as np
from faster_dan.basic.post_pocessing_layout import PostProcessingModuleREAD, PostProcessingModuleRIMES


class MetricManager:

    def __init__(self, metric_names, dataset_name):
        self.dataset_name = dataset_name
        self.edit_and_num_edge_nodes = edit_and_num_items_for_ged_from_str

        self.tokens = READ_TOKENS | RIMES_TOKENS

        self.begin_tokens = "".join(np.unique(list(self.tokens.values())))
        self.matching_tokens = dict()
        for t in self.begin_tokens:
            self.matching_tokens[t] = t.upper()
        self.layout_tokens = "".join(list(self.matching_tokens.values()) + list(self.matching_tokens.keys()))
        if len(self.layout_tokens) == 0:
            self.layout_tokens = None

        self.page_token = self.tokens["page"] if "page" in self.tokens else None
        self.metric_names = metric_names
        self.epoch_metrics = None

        self.linked_metrics = {
            "cer": ["edit_chars", "nb_chars"],
            "wer": ["edit_words", "nb_words"],
            "loer": ["edit_graph", "nb_nodes_and_edges", "nb_pp_op_layout", "nb_gt_layout_token"],
            "map_cer_per_class": ["map_cer", ],
            "layout_precision_per_class_per_threshold": ["map_cer", ],
        }

        self.init_metrics()

    def init_metrics(self):
        """
        Initialization of the metrics specified in metrics_name
        """

        self.epoch_metrics = {
            "nb_samples": list(),
            "names": list(),
            "ids": list(),
        }

        for metric_name in self.metric_names:
            if metric_name in self.linked_metrics:
                for linked_metric_name in self.linked_metrics[metric_name]:
                    if linked_metric_name not in self.epoch_metrics.keys():
                        self.epoch_metrics[linked_metric_name] = list()
            else:
                self.epoch_metrics[metric_name] = list()

    def update_metrics(self, batch_metrics):
        """
        Add batch metrics to the metrics
        """
        for key in batch_metrics.keys():
            if key in self.epoch_metrics:
                self.epoch_metrics[key] += batch_metrics[key]

    def get_display_values(self, output=False):
        """
        format metrics values for shell display purposes
        """
        metric_names = self.metric_names.copy()
        if output:
            metric_names.extend(["nb_samples"])
        display_values = dict()
        for metric_name in metric_names:
            value = None
            if output:
                if metric_name in ["nb_samples", "weights"]:
                    value = np.sum(self.epoch_metrics[metric_name])
                elif metric_name in ["time", ]:
                    total_time = np.sum(self.epoch_metrics[metric_name])
                    sample_time = total_time / np.sum(self.epoch_metrics["nb_samples"])
                    display_values["sample_time"] = round(sample_time, 4)
                    value = total_time
                elif metric_name == "loer":
                    display_values["percent_pp_op"] = np.sum(self.epoch_metrics["nb_pp_op_layout"]) / np.sum(self.epoch_metrics["nb_gt_layout_token"])
                elif metric_name == "map_cer_per_class":
                    value = compute_global_mAP_per_class(self.epoch_metrics["map_cer"])
                    for key in value.keys():
                        display_values["map_cer_" + key] = round(value[key], 4)
                    continue
                elif metric_name == "layout_precision_per_class_per_threshold":
                    value = compute_global_precision_per_class_per_threshold(self.epoch_metrics["map_cer"])
                    for key_class in value.keys():
                        for threshold in value[key_class].keys():
                            display_values["map_cer_{}_{}".format(key_class, threshold)] = round(
                                value[key_class][threshold], 4)
                    continue
            if metric_name == "cer":
                value = np.sum(self.epoch_metrics["edit_chars"]) / np.sum(self.epoch_metrics["nb_chars"])
                if output:
                    display_values["nb_chars"] = np.sum(self.epoch_metrics["nb_chars"])
            elif metric_name == "wer":
                value = np.sum(self.epoch_metrics["edit_words"]) / np.sum(self.epoch_metrics["nb_words"])
                if output:
                    display_values["nb_words"] = np.sum(self.epoch_metrics["nb_words"])
            elif metric_name in ["loss", ]:
                value = np.average(self.epoch_metrics[metric_name], weights=np.array(self.epoch_metrics["nb_samples"]))
            elif metric_name in ["map_cer", ]:
                value = compute_global_mAP(self.epoch_metrics[metric_name])
            elif metric_name == "loer":
                value = np.sum(self.epoch_metrics["edit_graph"]) / np.sum(self.epoch_metrics["nb_nodes_and_edges"])
            elif value is None:
                continue

            display_values[metric_name] = round(value, 4)
        return display_values

    def compute_metrics(self, values, metric_names):
        metrics = {
            "nb_samples": [values["nb_samples"], ],
        }
        pp_modules = None
        if "names" in values:
            pp_modules = [PostProcessingModuleREAD() if "read" in name.lower() else PostProcessingModuleRIMES() if "rimes" in name.lower() else None for name in values["names"]]
        for v in ["weights", "time"]:
            if v in values:
                metrics[v] = [values[v]]
        for metric_name in metric_names:
            if metric_name == "cer":
                metrics["edit_chars"] = [edit_cer_from_string(u, v, self.layout_tokens) for u, v in zip(values["str_y"], values["str_x"])]
                metrics["nb_chars"] = [nb_chars_cer_from_string(gt, self.layout_tokens) for gt in values["str_y"]]
            elif metric_name == "wer":
                split_gt = [format_string_for_wer(gt, self.layout_tokens) for gt in values["str_y"]]
                split_pred = [format_string_for_wer(pred, self.layout_tokens) for pred in values["str_x"]]
                metrics["edit_words"] = [edit_wer_from_formatted_split_text(gt, pred) for (gt, pred) in zip(split_gt, split_pred)]
                metrics["nb_words"] = [len(gt) for gt in split_gt]
            elif metric_name in ["loss", ]:
                metrics[metric_name] = [values[metric_name], ]
            elif metric_name in ["map_cer", ]:
                str_x = values["str_x"] if metric_name == "map_cer" else values["str_x_correction"]
                confidence = values["confidence_score"] if metric_name == "map_cer" else values["confidence_score_correction"]
                pp_pred = list()
                pp_score = list()
                for pred, score, pp_module in zip(str_x, confidence, pp_modules):
                    if pp_module is None:
                        pp_pred.append(None)
                        pp_score.append(None)
                    else:
                        pred_score = pp_module.post_process(pred, score)
                        pp_pred.append(pred_score[0])
                        pp_score.append(pred_score[1])
                metrics[metric_name] = [compute_layout_mAP_per_class(y, x, conf, self.matching_tokens) if x is not None else dict() for x, conf, y in zip(pp_pred, pp_score, values["str_y"])]
            elif metric_name == "loer":
                pp_pred = list()
                metrics["nb_pp_op_layout"] = list()
                for pred, pp_module in zip(values["str_x"], pp_modules):
                    if pp_module is None:
                        pp_pred.append(None)
                        metrics["nb_pp_op_layout"].append(0)
                    else:
                        pp_pred.append(pp_module.post_process(pred))
                        metrics["nb_pp_op_layout"].append(pp_module.num_op)
                metrics["nb_gt_layout_token"] = [len(keep_only_tokens(str_x, self.layout_tokens)) for str_x in values["str_x"]]
                edit_and_num_items = [self.edit_and_num_edge_nodes(y, x, self.layout_tokens, self.page_token) if x is not None else (0, 0) for x, y in zip(pp_pred, values["str_y"])]
                metrics["edit_graph"], metrics["nb_nodes_and_edges"] = [ei[0] for ei in edit_and_num_items], [ei[1] for ei in edit_and_num_items]
        return metrics

    def get(self, name):
        return self.epoch_metrics[name]


def keep_only_tokens(str, tokens):
    """
    Remove all but layout tokens from string
    """
    return re.sub('([^' + tokens + '])', '', str)


def keep_all_but_tokens(str, tokens):
    """
    Remove all layout tokens from string
    """
    return re.sub('([' + tokens + '])', '', str)


def edit_cer_from_string(gt, pred, layout_tokens=None):
    """
    Format and compute edit distance between two strings at character level
    """
    gt = format_string_for_cer(gt, layout_tokens)
    pred = format_string_for_cer(pred, layout_tokens)
    return editdistance.eval(gt, pred)


def nb_chars_cer_from_string(gt, layout_tokens=None):
    """
    Compute length after formatting of ground truth string
    """
    return len(format_string_for_cer(gt, layout_tokens))


def edit_wer_from_string(gt, pred, layout_tokens=None):
    """
    Format and compute edit distance between two strings at word level
    """
    split_gt = format_string_for_wer(gt, layout_tokens)
    split_pred = format_string_for_wer(pred, layout_tokens)
    return edit_wer_from_formatted_split_text(split_gt, split_pred)


def format_string_for_wer(str, layout_tokens):
    """
    Format string for WER computation: remove layout tokens, treat punctuation as word, replace line break by space
    """
    str = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', str)  # punctuation processed as word
    if layout_tokens is not None:
        str = keep_all_but_tokens(str, layout_tokens)  # remove layout tokens from metric
    str = re.sub('([ \n])+', " ", str).strip()  # keep only one space character
    return str.split(" ")


def format_string_for_cer(str, layout_tokens):
    """
    Format string for CER computation: remove layout tokens and extra spaces
    """
    if layout_tokens is not None:
        str = keep_all_but_tokens(str, layout_tokens)  # remove layout tokens from metric
    str = re.sub('([\n])+', "\n", str)  # remove consecutive line breaks
    str = re.sub('([ ])+', " ", str).strip()  # remove consecutive spaces
    return str


def edit_wer_from_formatted_split_text(gt, pred):
    """
    Compute edit distance at word level from formatted string as list
    """
    return editdistance.eval(gt, pred)


def extract_by_tokens(input_str, begin_token, end_token, associated_score=None, order_by_score=False):
    """
    Extract list of text regions by begin and end tokens
    Order the list by confidence score
    """
    if order_by_score:
        assert associated_score is not None
    res = list()
    for match in re.finditer("{}[^{}]*{}".format(begin_token, end_token, end_token), input_str):
        begin, end = match.regs[0]
        if order_by_score:
            res.append({
                "confidence": np.mean([associated_score[begin], associated_score[end-1]]),
                "content": input_str[begin+1:end-1]
            })
        else:
            res.append(input_str[begin+1:end-1])
    if order_by_score:
        res = sorted(res, key=lambda x: x["confidence"], reverse=True)
        res = [r["content"] for r in res]
    return res


def compute_layout_precision_per_threshold(gt, pred, score, begin_token, end_token, layout_tokens, return_weight=True):
    """
    Compute average precision of a given class for CER threshold from 5% to 50% with a step of 5%
    """
    pred_list = extract_by_tokens(pred, begin_token, end_token, associated_score=score, order_by_score=True)
    gt_list = extract_by_tokens(gt, begin_token, end_token)
    pred_list = [keep_all_but_tokens(p, layout_tokens) for p in pred_list]
    gt_list = [keep_all_but_tokens(gt, layout_tokens) for gt in gt_list]
    precision_per_threshold = [compute_layout_AP_for_given_threshold(gt_list, pred_list, threshold/100) for threshold in range(5, 51, 5)]
    if return_weight:
        return precision_per_threshold, len(gt_list)
    return precision_per_threshold


def compute_layout_AP_for_given_threshold(gt_list, pred_list, threshold):
    """
    Compute average precision of a given class for a given CER threshold
    """
    remaining_gt_list = gt_list.copy()
    num_true = len(gt_list)
    correct = np.zeros((len(pred_list)), dtype=np.bool)
    for i, pred in enumerate(pred_list):
        if len(remaining_gt_list) == 0:
            break
        cer_with_gt = [edit_cer_from_string(gt, pred)/nb_chars_cer_from_string(gt) for gt in remaining_gt_list]
        cer, ind = np.min(cer_with_gt), np.argmin(cer_with_gt)
        if cer <= threshold:
            correct[i] = True
            del remaining_gt_list[ind]
    precision = np.cumsum(correct, dtype=np.int) / np.arange(1, len(pred_list)+1)
    recall = np.cumsum(correct, dtype=np.int) / num_true
    max_precision_from_recall = np.maximum.accumulate(precision[::-1])[::-1]
    recall_diff = (recall - np.concatenate([np.array([0, ]), recall[:-1]]))
    P = np.sum(recall_diff * max_precision_from_recall)
    return P


def compute_layout_mAP_per_class(gt, pred, score, tokens):
    """
    Compute the mAP_cer for each class for a given sample
    """
    layout_tokens = "".join(list(tokens.keys()))
    AP_per_class = dict()
    for token in tokens.keys():
        if token in gt:
            AP_per_class[token] = compute_layout_precision_per_threshold(gt, pred, score, token, tokens[token], layout_tokens=layout_tokens)
    return AP_per_class


def compute_global_mAP(list_AP_per_class):
    """
    Compute the global mAP_cer for several samples
    """
    weights_per_doc = list()
    mAP_per_doc = list()
    for doc_AP_per_class in list_AP_per_class:
        APs = np.array([np.mean(doc_AP_per_class[key][0]) for key in doc_AP_per_class.keys()])
        weights = np.array([doc_AP_per_class[key][1] for key in doc_AP_per_class.keys()])
        mAP_per_doc.append(np.average(APs, weights=weights) if np.sum(weights) != 0 else 0)
        weights_per_doc.append(np.sum(weights))
    return np.average(mAP_per_doc, weights=weights_per_doc) if np.sum(weights_per_doc) != 0 else 0


def compute_global_mAP_per_class(list_AP_per_class):
    """
    Compute the mAP_cer per class for several samples
    """
    mAP_per_class = dict()
    for doc_AP_per_class in list_AP_per_class:
        for key in doc_AP_per_class.keys():
            if key not in mAP_per_class:
                mAP_per_class[key] = {
                    "AP": list(),
                    "weights": list()
                }
            mAP_per_class[key]["AP"].append(np.mean(doc_AP_per_class[key][0]))
            mAP_per_class[key]["weights"].append(doc_AP_per_class[key][1])
    for key in mAP_per_class.keys():
        mAP_per_class[key] = np.average(mAP_per_class[key]["AP"], weights=mAP_per_class[key]["weights"])
    return mAP_per_class


def compute_global_precision_per_class_per_threshold(list_AP_per_class):
    """
    Compute the mAP_cer per class and per threshold for several samples
    """
    mAP_per_class = dict()
    for doc_AP_per_class in list_AP_per_class:
        for key in doc_AP_per_class.keys():
            if key not in mAP_per_class:
                mAP_per_class[key] = dict()
                for threshold in range(5, 51, 5):
                    mAP_per_class[key][threshold] = {
                        "precision": list(),
                        "weights": list()
                    }
            for i, threshold in enumerate(range(5, 51, 5)):
                mAP_per_class[key][threshold]["precision"].append(np.mean(doc_AP_per_class[key][0][i]))
                mAP_per_class[key][threshold]["weights"].append(doc_AP_per_class[key][1])
    for key_class in mAP_per_class.keys():
        for threshold in mAP_per_class[key_class]:
            mAP_per_class[key_class][threshold] = np.average(mAP_per_class[key_class][threshold]["precision"], weights=mAP_per_class[key_class][threshold]["weights"])
    return mAP_per_class


def str_to_graph(str, begin_tokens, page_token):
    end_tokens = [t.upper() for t in begin_tokens]
    all_tokens = "".join(list(begin_tokens) + end_tokens)
    layout_token_sequence = keep_only_tokens(str, all_tokens)
    g = nx.DiGraph()
    page = 0
    g.add_node("D", type="document", level=0, page=page)
    match = {
        "ⓐ": "A",
        "ⓑ": "B",
        "ⓒ": "C",
        "ⓓ": "D",
        "ⓔ": "E",
        "ⓕ": "F",
        "ⓖ": "G",
        "ⓗ": "H",
        "ⓘ": "I",
        "ⓙ": "J",
        "ⓚ": "K",
        "ⓛ": "L",
        "ⓜ": "M",
        "ⓝ": "N",
        "ⓞ": "O",
        "ⓟ": "P",
        "ⓠ": "Q",
        "ⓡ": "R",
        "ⓢ": "S",
        "ⓣ": "T",
        "ⓤ": "U",
        "ⓥ": "V",
        "ⓦ": "W",
        "ⓧ": "X",
        "ⓨ": "Y",
        "ⓩ": "Z",
    }
    num = dict()
    for t in begin_tokens:
        num[t] = 0
    previous_close = None
    previous_begin = [["", "D"], ]
    for ind, c in enumerate(layout_token_sequence):
        if c in begin_tokens:
            if c == page_token:
                page += 1
            num[c] += 1
            node_name = "{}_{}".format(match[c], num[c])
            parent_node = previous_begin[-1][1]
            g.add_node(node_name, type=c, level=len(previous_begin), page=page)
            g.add_edge(parent_node, node_name)
            previous_begin.append([c, node_name])
            if previous_close:
                g.add_edge(previous_close[1], node_name)
            previous_close = None
        elif c in end_tokens:
            assert previous_begin[-1][0] == c.lower()
            previous_close = previous_begin[-1]
            del previous_begin[-1]
    return g


def graph_edit_distance_by_page(g1, g2, page_token):
    """
    Compute graph edit distance page by page for the READ 2016 dataset
    """
    page_nodes1 = [n for n in g1.nodes().items() if n[1]["type"] == page_token]
    page_nodes2 = [n for n in g2.nodes().items() if n[1]["type"] == page_token]
    if page_token is None or len(page_nodes1) <= 1:
        return graph_edit_distance(g1, g2)
    num_pages_g1 = len(page_nodes1)
    num_pages_g2 = len(page_nodes2)
    page_graphs_1 = [g1.subgraph([n[0] for n in g1.nodes().items() if n[1]["page"] == num_page]) for num_page in range(1, num_pages_g1 + 1)]
    page_graphs_2 = [g2.subgraph([n[0] for n in g2.nodes().items() if n[1]["page"] == num_page]) for num_page in range(1, num_pages_g2 + 1)]
    edit = 0
    for i in range(max(len(page_graphs_1), len(page_graphs_2))):
        page_1 = page_graphs_1[i] if i < len(page_graphs_1) else nx.DiGraph()
        page_2 = page_graphs_2[i] if i < len(page_graphs_2) else nx.DiGraph()
        edit += graph_edit_distance(page_1, page_2)
    return edit


def graph_edit_distance(g1, g2):
    """
    Compute graph edit distance between two graphs
    """
    for v in nx.optimize_graph_edit_distance(g1, g2,
                                             node_ins_cost=lambda node: 1,
                                             node_del_cost=lambda node: 1,
                                             node_subst_cost=lambda node1, node2: 0 if node1["type"] == node2["type"] else 1,
                                             edge_ins_cost=lambda edge: 1,
                                             edge_del_cost=lambda edge: 1,
                                             edge_subst_cost=lambda edge1, edge2: 0 if edge1 == edge2 else 1
                                             ):
        new_edit = v
    return new_edit


def edit_and_num_items_for_ged_from_str(str_gt, str_pred, tokens, page_token):
    """
    Compute graph edit distance and num nodes/edges for normalized graph edit distance
    """
    begin_tokens = "".join(np.unique(list(tokens.lower())))
    g_gt = str_to_graph(str_gt, begin_tokens, page_token)
    g_pred = str_to_graph(str_pred, begin_tokens, page_token)
    return graph_edit_distance_by_page(g_gt, g_pred, page_token), g_gt.number_of_nodes() + g_gt.number_of_edges()
