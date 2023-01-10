import cv2
from faster_dan.Datasets.dataset_formatters.generic_dataset_formatter import OCRDatasetFormatter
import os
import re
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET


class MaurdorDatasetFormatter(OCRDatasetFormatter):
    def __init__(self, level,
                 set_names=("train", "valid", "test"),
                 categories=("C1", "C2", "C3", "C4", "C5"),
                 languages=("french", "english", "arabic", "other"),
                 orientation="original",
                 custom_name=None):
        super(MaurdorDatasetFormatter, self).__init__("Maurdor", level, set_names)

        if not all([c in ["C1", "C2", "C3", "C4", "C5"] for c in categories]):
            raise Exception("Wrong argument for categories")

        if not all([c in ["french", "english", "arabic", "other"] for c in languages]):
            raise Exception("Wrong argument for languages")

        if orientation not in ["original", "rectified"]:
            raise Exception("Wrong argument for orientation")

        self.categories = categories
        self.languages = languages
        output_name = self.dataset_name if custom_name is None else custom_name
        self.target_fold_path = os.path.join("../formatted", "{}_{}".format(output_name, level))
        self.polygon_as_rect = False
        self.extract_with_dirname = True
        self.orientation = orientation
        self.a4_only = True

        self.map_datasets_files.update({
            "Maurdor": {
                # (51,738 for train, 7,464 for validation and 7,776 for test)
                "page": {
                    "arx_files": ["train/XML.tar.gz", "train/TIFF.tar.gz",
                                  "valid/XML.tar.gz", "valid/TIFF.tar.gz",
                                  "test/XML.tar.gz", "test/TIFF.tar.gz"],
                    "needed_files": [],
                    "format_function": self.format_maurdor_page,
                },
                "line": {
                    "arx_files": ["train/XML.tar.gz", "train/TIFF.tar.gz",
                                  "valid/XML.tar.gz", "valid/TIFF.tar.gz",
                                  "test/XML.tar.gz", "test/TIFF.tar.gz"],
                    "needed_files": [],
                    "format_function": self.format_maurdor_line,
                },
            }
        })

    @staticmethod
    def polygon_to_rect_coord(region):

        polygon = [a.replace("(", "").replace(")", "") for a in region.attrib["polygon"].split(";")]
        polygon = [(int(a.split(",")[0]), int(a.split(",")[1])) for a in polygon]

        left = min([int(a[0]) for a in polygon])
        right = max([int(a[0]) for a in polygon])
        top = min([int(a[1]) for a in polygon])
        bottom = max([int(a[1]) for a in polygon])
        coord = {
                "left": left,
                "right": right,
                "top": top,
                "bottom": bottom,
                "orientation": 0,
                "type": "rect"
        }
        return coord

    @staticmethod
    def points_to_rect_coord(points):
        left = min([int(a[0]) for a in points])
        right = max([int(a[0]) for a in points])
        top = min([int(a[1]) for a in points])
        bottom = max([int(a[1]) for a in points])
        coord = {
                "left": left,
                "right": right,
                "top": top,
                "bottom": bottom,
                "orientation": 0,
                "type": "rect"
        }
        return coord

    @staticmethod
    def text_region_to_coord(region):
        col, row, width, height = [int(region.attrib[a]) for a in ["col", "row", "width", "height"]]
        coord = {
            "left": col,
            "right": col + width,
            "top": row,
            "bottom": row + height,
            "orientation": -float(region.attrib["orientationD"]) if "orientationD" in region.attrib else 0,
            "type": "rect"
        }
        return coord

    @staticmethod
    def polygon_to_coord(region):
        polygon = [a.replace("(", "").replace(")", "") for a in region.attrib["polygon"].split(";")]
        polygon = [np.array([int(a.split(",")[0]), int(a.split(",")[1])]) for a in polygon]
        polygon = np.stack(polygon)
        coord = {
            "points": polygon,
            "orientation": -float(region.attrib["orientationD"]) if "orientationD" in region.attrib else 0,
            "type": "polygon"
        }
        return coord

    def region_to_coord(self, region):
        if all(a in region.attrib for a in ["col", "row", "width", "height"]):
            rect_coord = MaurdorDatasetFormatter.text_region_to_coord(region)
            poly_coord = None
        else:
            rect_coord = MaurdorDatasetFormatter.polygon_to_rect_coord(region)
            poly_coord = MaurdorDatasetFormatter.polygon_to_coord(region)
        return rect_coord, poly_coord

    def rectify_oriented_coord(self, coord, orientation, page_width, page_height):
        if coord["type"] == "polygon":
            new_points = list()
            for point in coord["points"]:
                new_points.append(self.rectify_oriented_point(point, orientation, page_width, page_height))
            coord["points"] = np.array(new_points, dtype=np.int64)
            return coord
        else:
            if orientation == 180:
                coord["left"], coord["right"] = page_width - coord["right"], page_width - coord["left"]
                coord["bottom"], coord["top"] = page_height - coord["top"], page_height - coord["bottom"]
                return coord
            elif orientation == 90:
                coord["left"], coord["right"], coord["top"], coord["bottom"] = page_height - coord["bottom"], page_height - coord["top"], coord["left"], coord["right"]
                return coord
            elif orientation == 270:
                # coord["left"], coord["right"], coord["top"], coord["bottom"] = page_height - coord["bottom"], page_height - coord["top"], coord["left"], coord["right"]
                coord["left"], coord["right"], coord["top"], coord["bottom"] = coord["top"], coord["bottom"], page_width - coord["right"], page_width - coord["left"]
                return coord

    def rectify_oriented_point(self, point, orientation, width, height):
        if orientation == 180:
            return [width - point[0], height - point[1]]
        if orientation == 90:
            return [height - point[1], point[0]]
        if orientation == 270:
            return [point[1], width - point[0]]

    def preformat(self):
        dataset = {
            "train": list(),
            "valid": list(),
            "test": list()
        }

        for set_name in self.set_names:
            xml_fold_path = os.path.join(self.temp_fold, set_name, "XML", "xml")
            for xml_filename in sorted(os.listdir(xml_fold_path)):
                xml_path = os.path.join(xml_fold_path, xml_filename)
                xml_root = ET.parse(xml_path).getroot()
                doc = xml_root.find("{http://lamp.cfar.umd.edu/media/projects/GEDI/}DL_DOCUMENT")
                document = dict()
                document["num_pages"] = int(doc.attrib["NrOfPages"])
                document["pages"] = list()
                document["languages"] = set()
                document["scripts"] = set()
                for id_page, page in enumerate(doc.findall("{http://lamp.cfar.umd.edu/media/projects/GEDI/}DL_PAGE")):
                    page_info = dict()
                    page_info["width"] = int(page.attrib["width"])
                    page_info["height"] = int(page.attrib["height"])
                    page_info["languages"] = set()
                    if id_page == 0:
                        document["category"] = page.attrib["Category"]
                    page_info["category"] = document["category"]
                    for i in range(5):
                        field_name = "Language_{}".format(i)
                        if field_name in page.attrib and page.attrib[field_name] != "":
                            page_info["languages"].add(page.attrib[field_name])
                    document["languages"] = document["languages"].union(page_info["languages"])
                    if len(page_info["languages"]) == 0:
                        page_info["languages"] = document["languages"]
                        if len(page_info["languages"]) == 0:
                            print("ko")

                    if any([l not in self.languages for l in document["languages"]]) or document["category"] not in self.categories:
                        continue
                    page_info["scripts"] = set()
                    page_info['orientation'] = int(page.attrib["GEDI_orientation"]) if "GEDI_orientation" in page.attrib else 0
                    img_name = page.attrib["src"]
                    document["img_path"] = os.path.join(self.temp_fold, set_name, "TIFF", img_name)
                    zones = page.findall("{http://lamp.cfar.umd.edu/media/projects/GEDI/}DL_ZONE")
                    text_regions = [z for z in zones if z.attrib["gedi_type"] == "TextRegion"]
                    graphic_regions = [z for z in zones if z.attrib["gedi_type"] == "GraphicRegion"]
                    page_info["text_regions"] = list()
                    page_info["graphic_regions"] = list()
                    for text_region in text_regions:
                        if page_info['orientation'] != 0 and "orientationD" in text_region.attrib:
                            correct_text_region(text_region, page)
                        text_region_description = dict()
                        if "contents" not in text_region.attrib or text_region.attrib["contents"] == "":
                            print("no content in text region")
                            continue
                        if "script" in text_region.attrib:
                            page_info["scripts"].add(text_region.attrib["script"])
                        text_region_description["label"] = self.convert_label(text_region.attrib["contents"])
                        text_region_description["id"] = int(text_region.attrib["id"])
                        if "nextZoneID" in text_region.attrib:
                            text_region_description["id_next"] = int(text_region.attrib["nextZoneID"])
                        else:
                            text_region_description["id_next"] = None
                        rect_coord, poly_coord = self.region_to_coord(text_region)
                        if self.orientation == "rectified" and page_info['orientation'] != 0:
                            rect_coord = self.rectify_oriented_coord(rect_coord, page_info['orientation'], page_info["width"], page_info["height"])
                            if poly_coord:
                                poly_coord = self.rectify_oriented_coord(poly_coord, page_info['orientation'], page_info["width"], page_info["height"])
                        text_region_description["rect_coord"] = rect_coord
                        text_region_description["polygon_coord"] = poly_coord
                        text_region_description["functions"] = list()
                        for i in range(5):
                            field_name = "function_{}".format(i)
                            if field_name in text_region.attrib and text_region.attrib[field_name] != "":
                                text_region_description["functions"].append(text_region.attrib[field_name])
                        page_info["text_regions"].append(text_region_description)
                    for graphic_region in graphic_regions:
                        graphic_region_description = dict()
                        coord = self.region_to_coord(graphic_region)
                        graphic_region_description["coord"] = coord
                        graphic_region_description["id"] = int(graphic_region.attrib["id"])
                        graphic_region_description["functions"] = list()
                        page_info["graphic_regions"].append(graphic_region_description)
                        for i in range(5):
                            field_name = "function_{}".format(i)
                            if field_name in graphic_region.attrib and graphic_region.attrib[field_name] != "":
                                graphic_region_description["functions"].append(graphic_region.attrib[field_name])
                    for i in range(len(page_info["text_regions"])):
                        text_ids = [tr["id"] for tr in page_info["text_regions"]]
                        graphic_ids = [gr["id"] for gr in page_info["graphic_regions"]]
                        if page_info["text_regions"][i]["id_next"]:
                            if page_info["text_regions"][i]["id_next"] in text_ids:
                                page_info["text_regions"][i]["id_next_type"] = "text"
                            elif page_info["text_regions"][i]["id_next"] in graphic_ids:
                                page_info["text_regions"][i]["id_next_type"] = "graphic"
                            else:
                                page_info["text_regions"][i]["id_next_type"] = "unknown"

                    document["scripts"] = document["scripts"].union(page_info["scripts"])
                    if len(page_info["scripts"]) == 0:
                        page_info["scripts"] = document["scripts"]
                        if len(page_info["scripts"]) == 0:
                            print("ko")
                    document["pages"].append(page_info)
                if len(document["pages"]) != 0:
                    dataset[set_name].append(document)
        dataset = self.filter_dataset(dataset)
        return dataset

    def format_maurdor_page(self):
        dataset = self.preformat()
        for set_name in self.set_names:
            i = 0
            for doc in dataset[set_name]:
                img_path = doc["img_path"]
                for page_id, page in enumerate(doc["pages"]):
                    pil_img = Image.open(img_path)
                    pil_img.seek(page_id)
                    img = np.array(pil_img)
                    if len(img.shape) == 3 and img.shape[2] == 4:
                        img = img[:, :, :3]
                    if self.orientation == "rectified":
                        img = rotate_image(img, page["orientation"])
                    page, img = self.convert_a4_150dpi_page_img(page, img)
                    new_img_name = "{}_{}.png".format(set_name, i)
                    new_img_path = os.path.join(self.target_fold_path, set_name, new_img_name)
                    Image.fromarray(img).save(new_img_path)
                    self.gt[set_name][new_img_name] = {
                        "text": page["label"],
                        "lines": [self.text_region_to_line(tr) for tr in page["text_regions"] if self.is_line_tr(tr)],
                        "languages": page["languages"],
                        "category": page["category"],
                        "scripts": page["scripts"],
                    }
                    i += 1

    def format_maurdor_line(self):
        dataset = self.preformat()
        for set_name in self.set_names:
            i = 0
            for doc in dataset[set_name]:
                img_path = doc["img_path"]
                for page_id, page in enumerate(doc["pages"]):
                    pil_img = Image.open(img_path)
                    pil_img.seek(page_id)
                    page_img = np.array(pil_img)
                    page, page_img = self.convert_a4_150dpi_page_img(page, page_img)
                    for text_region in page["text_regions"]:
                        label = text_region["label"].strip()
                        if not self.is_line_tr(text_region):
                            continue
                        img = np.array(page_img, copy=True)
                        if len(img.shape) == 3 and img.shape[2] == 4:
                            img = img[:, :, :3]
                        if self.orientation == "rectified":
                            img = rotate_image(img, page["orientation"])
                        coord = text_region["rect_coord"]
                        if text_region["polygon_coord"] is None:
                            img = img[coord["top"]: coord["bottom"], coord["left"]:coord["right"]]
                        else:
                            mask = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
                            cv2.fillConvexPoly(mask, text_region["polygon_coord"]["points"], color=1)
                            img[mask != 1] = 255
                            img = img[coord["top"]: coord["bottom"], coord["left"]:coord["right"]]
                        if img.shape[0] >= 200 or img.shape[0] > img.shape[1]:
                            continue
                        new_img_name = "{}_{}.png".format(set_name, i)
                        new_img_path = os.path.join(self.target_fold_path, set_name, new_img_name)
                        Image.fromarray(img).save(new_img_path)
                        self.gt[set_name][new_img_name] = {
                            "text": label,
                        }
                        i += 1

    def convert_a4_150dpi_page_img(self, page, img):
        h, w = img.shape[:2]
        ratio_h, ratio_w = (1754 / h, 1240 / w) if h > w else (1240 / h, 1754 / w)
        new_h, new_w = int(ratio_h*h), int(ratio_w*w)
        img = Image.fromarray(img)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        img = np.array(img)
        page["width"] = new_w
        page["height"] = new_h
        for tr in page["text_regions"]:
            for coord in ["top", "bottom"]:
                tr["rect_coord"][coord] = int(ratio_h * tr["rect_coord"][coord])
            for coord in ["left", "right"]:
                tr["rect_coord"][coord] = int(ratio_w * tr["rect_coord"][coord])
            if tr["polygon_coord"]:
                tr["polygon_coord"]["points"] = (tr["polygon_coord"]["points"] * np.array([ratio_w, ratio_h])).astype(np.int64)
        return page, img

    def convert_label(self, label):
        label = re.sub('¤{([^¤]*)[/|]([^¤]*)}¤', r'\1', label, flags=re.DOTALL)
        label = re.sub('¤([^¤]*)¤', r'\1', label, flags=re.DOTALL)
        sub = {
            "\xa0": "",
            "…": "...",
            "—": "-",
            "№": "N°",
            "\t": "",
            "▶": "",
            "►": "",
            "●": "",
            "": ""
        }
        for key in sub:
            label = label.replace(key, sub[key])
        return label

    def is_line_tr(self, tr):
        label = tr["label"].strip()
        height = tr["rect_coord"]["bottom"] - tr["rect_coord"]["top"]
        width = tr["rect_coord"]["right"] - tr["rect_coord"]["left"]
        return not ("\n" in label or len(label) == 0 or height > width)

    def text_region_to_line(self, tr):
        line = {
            "text": tr["label"],
            "top": tr["rect_coord"]["top"],
            "bottom": tr["rect_coord"]["bottom"],
            "right": tr["rect_coord"]["right"],
            "left": tr["rect_coord"]["left"]
        }
        if "polygon_coord" in tr and tr["polygon_coord"]:
            line["polygon"] = tr["polygon_coord"]
        return line

    def filter_dataset(self, dataset):
        new_dataset = dataset.copy()
        for set_name in self.set_names:
            new_set = list()
            for i, sample in enumerate(dataset[set_name]):
                to_del = False
                for page_id, page in enumerate(sample["pages"]):
                    if self.a4_only:
                        h, w = page["height"], page["width"]
                        ratio = max([h, w]) / min([h, w])
                        if not 1.3 < ratio < 1.5:
                            to_del = True
                    if not to_del:
                        for text_region in page["text_regions"]:
                            if not text_region["polygon_coord"] and text_region["rect_coord"]["orientation"] != 0:
                                to_del = True
                                break
                if not to_del:
                    new_set.append(self.order_text_regions(dataset[set_name][i]))
                    for i in range(len(new_set[-1]["pages"])):
                        page_label = "\n".join([tr["label"] for tr in new_set[-1]["pages"][i]["text_regions"]])
                        new_set[-1]["pages"][i]["label"] = page_label
                        self.charset = self.charset.union(set(page_label))
            new_dataset[set_name] = new_set
        return new_dataset

    def order_text_regions(self, sample):
        def gather_by_id(gr, list_id, list_gr, same_level=False):
            new_list = list()
            gr_to_gather = list()
            for i, g in enumerate(list_gr):
                if i in list_id:
                    gr_to_gather.append(g)
                else:
                    new_list.append(g)
            new_list.append({
                "coord": gather_tr_coords(gr_to_gather + [gr, ]),
                "atomic_items": [gr, ],
            })
            for tr in gr_to_gather:
                new_list[-1]["atomic_items"].extend(tr["atomic_items"])
            return new_list

        def merge_groups_as_rows(list_group, same_level=False):
            def is_same_row(gr1, gr2):
                higher, lower = (gr1["coord"], gr2["coord"]) if gr1["coord"]["top"] < gr2["coord"]["top"] else (gr2["coord"], gr1["coord"])
                h_high = higher["bottom"] - higher["top"]
                h_low = lower["bottom"] - lower["top"]
                return min([h_high, h_low, higher["bottom"] - lower["top"]]) > 0.25 * min([h_high, h_low])

            new_rows = list()
            for gr in list_group:
                if len(new_rows) == 0:
                    new_rows.append({
                        "coord": gr["coord"],
                        "atomic_items": [gr, ]
                    })
                else:
                    group_with = list()
                    for id_gr, ngr in enumerate(new_rows):
                        if is_same_row(gr, ngr):
                            group_with.append(id_gr)

                    if len(group_with) == 0:
                        new_rows.append({
                            "coord": gr["coord"],
                            "atomic_items": [gr, ]
                        })

                    else:
                        new_rows = gather_by_id(gr, group_with, new_rows, same_level=same_level)
            return new_rows

        def gather_form_key_values(text_regions):
            tr_groups = list()
            used_ids = list()
            next_ids = [t["id_next"] for t in text_regions if t["id_next"]]
            for tr in text_regions:
                if tr["id"] in used_ids or tr["id"] in next_ids:
                    continue
                used_ids.append(tr["id"])
                if tr["id_next"] and tr["id_next_type"] == "text":
                    list_consecutive_items = [tr, ]
                    temp_tr = tr
                    while temp_tr["id_next"] and temp_tr["id_next_type"] == "text":
                        used_ids.append(temp_tr["id_next"])
                        temp_tr = [t for t in text_regions if t["id"] == temp_tr["id_next"]][0]

                        list_consecutive_items.append(temp_tr)

                    tr_groups.append({
                        "coord": gather_tr_coords(list_consecutive_items),
                        "regions": list_consecutive_items
                    })
                else:
                    tr_groups.append({
                        "coord": tr["rect_coord"],
                        "regions": [tr, ]
                    })
            return tr_groups

        def split_in_cols(row):
            def is_same_col(gr1, gr2):
                lefter, righter = (gr1["coord"], gr2["coord"]) if gr1["coord"]["left"] < gr2["coord"]["left"] else (gr2["coord"], gr1["coord"])
                l_width = lefter["right"] - lefter["left"]
                r_width = righter["right"] - righter["left"]
                return min([l_width, r_width, lefter["right"] - righter["left"]]) > 0.25 * min([r_width, l_width])

            list_cols = list()
            for item in row["atomic_items"]:
                if len(list_cols) == 0:
                    list_cols.append({
                        "coord": item["coord"],
                        "atomic_items": [item, ]
                    })
                else:
                    group_with = list()
                    for id_gr, ngr in enumerate(list_cols):
                        if is_same_col(item, ngr):
                            group_with.append(id_gr)
                    if len(group_with) == 0:
                        list_cols.append({
                            "coord": item["coord"],
                            "atomic_items": [item, ]
                        })
                    else:
                        list_cols = gather_by_id(item, group_with, list_cols)
            row["cols"] = list_cols
            del row["atomic_items"]
            return row

        def gather_tr_coords(list_tr):
            return {
                    "top": min([g["coord"]["top"] if "coord" in g else g["rect_coord"]["top"] for g in list_tr]),
                    "bottom": max([g["coord"]["bottom"] if "coord" in g else g["rect_coord"]["bottom"] for g in list_tr]),
                    "left": min([g["coord"]["left"] if "coord" in g else g["rect_coord"]["left"] for g in list_tr]),
                    "right": max([g["coord"]["right"] if "coord" in g else g["rect_coord"]["right"] for g in list_tr])
                }

        new_sample = sample.copy()
        new_pages = list()
        for page in sample["pages"]:
            new_page = page.copy()
            # group form key/value together and encapsulate each tr into group tr
            atomic_items = gather_form_key_values(page["text_regions"])
            rows = merge_groups_as_rows(atomic_items)
            split_rows = [split_in_cols(r) for r in rows]
            ordered_text_regions = list()
            split_rows.sort(key=lambda x: x["coord"]["top"])
            for row in split_rows:
                row["cols"].sort(key=lambda x: x["coord"]["left"])
                for col in row["cols"]:
                    col["atomic_items"].sort(key=lambda x: x["coord"]["top"])
                    for item in col["atomic_items"]:
                        for tr in item["regions"]:
                            ordered_text_regions.append(tr)
            new_page["text_regions"] = ordered_text_regions
            new_pages.append(new_page)
        new_sample["pages"] = new_pages
        return new_sample


def rotate_image(img, angle):
    if angle == 0:
        return img
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def correct_text_region(text_region, page):

    angle = int(page.attrib["GEDI_orientation"])
    if angle == 180:
        text_region.attrib["col"] = str(int(text_region.attrib["col"]) - int(text_region.attrib["width"]))
        text_region.attrib["row"] = str(int(text_region.attrib["row"]) - int(text_region.attrib["height"]))
    elif angle == 90:
        text_region.attrib["col"] = str(int(text_region.attrib["row"]) - int(text_region.attrib["width"]))# - int(text_region.attrib["width"]))
        text_region.attrib["row"] = str(int(page.attrib["width"]) - int(text_region.attrib["col"]))
    elif angle == 270:
        pass
    return text_region


if __name__ == "__main__":

    MaurdorDatasetFormatter("page",
                            categories=["C3", "C4"],
                            languages=["french", "english"],
                            orientation="rectified",
                            custom_name="MAURDOR_C3_C4").format()

    MaurdorDatasetFormatter("page",
                            categories=["C3", ],
                            languages=["french", "english"],
                            orientation="rectified",
                            custom_name="MAURDOR_C3").format()

    MaurdorDatasetFormatter("page",
                            categories=["C4", ],
                            languages=["french", "english"],
                            orientation="rectified",
                            custom_name="MAURDOR_C4").format()

