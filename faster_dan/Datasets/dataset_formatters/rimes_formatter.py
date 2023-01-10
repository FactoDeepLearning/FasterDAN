from faster_dan.Datasets.dataset_formatters.generic_dataset_formatter import OCRDatasetFormatter
from faster_dan.Datasets.dataset_formatters.utils_dataset import natural_sort
import os
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import re


SEM_TOKENS = {
    "opening": "ⓞ",
    "body": "ⓑ",
    'PS': "ⓟ",
    'sender': "ⓢ",
    'reference': "ⓢ",
    'object': "ⓨ",
    'date': "ⓦ",
    'recipient': "ⓡ",
}

# Layout string to token
SEM_MATCHING_TOKENS_STR = {
            'Ouverture': SEM_TOKENS["opening"],  # opening
            'Corps de texte': SEM_TOKENS["body"],  # body
            'PS/PJ': SEM_TOKENS["PS"],  # post scriptum
            'Coordonnées Expéditeur': SEM_TOKENS["sender"],  # sender
            'Reference': SEM_TOKENS["reference"],  # also counted as sender information
            'Objet': SEM_TOKENS["object"],  # why
            'Date, Lieu': SEM_TOKENS["date"],  # where, when
            'Coordonnées Destinataire': SEM_TOKENS["recipient"],  # recipient
        }


class RIMESDatasetFormatter(OCRDatasetFormatter):
    def __init__(self, level, set_names=["train", "valid", "test"], dpi=150, sem_token=False, order_sem=False):
        super(RIMESDatasetFormatter, self).__init__("RIMES", level, "_sem" if sem_token else "", set_names)

        self.source_fold_path = os.path.join("./Datasets/raw", "RIMES" if level!= "word" else "RIMES_word")
        self.dpi = dpi
        self.sem_token = sem_token
        self.map_datasets_files.update({
            "RIMES": {
                # (10,532 for train, 801 for validation and 778 for test)
                "line": {
                    "arx_files": ["eval_2011_gray.tar", "training_2011_gray.tar"],
                    "needed_files": ["eval_2011_annotated.xml", "training_2011.xml"],
                    "format_function": self.format_rimes_line,
                },
                # (1,050 for train, 100 for validation and 100 for test)
                "page": {
                    "arx_files": ["RIMES_page.tar.gz", ],
                    "needed_files": [],
                    "format_function": self.format_rimes_page,
                },
            }
        })

        self.matching_tokens_str = SEM_MATCHING_TOKENS_STR
        self.tokens = SEM_TOKENS
        self.ordering_function = order_text_regions_sem if order_sem else order_text_regions

    def preformat_rimes_paragraph(self):
        dataset = {
            "train": list(),
            "valid": list(),
            "test": list()
        }
        img_folder_path = os.path.join(self.temp_fold, "images_gray")
        nb_training_samples = sum([1 if "train" in name else 0 for name in os.listdir(img_folder_path)])
        nb_valid_samples = 100
        begin_valid_ind = nb_training_samples - nb_valid_samples
        for set_name, xml_path in zip(["train", "eval"], ["training_2011.xml", "eval_2011_annotated.xml"]):
            xml_root = ET.parse(os.path.join(self.source_fold_path, xml_path)).getroot()
            for page in xml_root:
                lines = list()
                full_text = ""
                name = page.attrib.get("FileName").split("/")[-1].split(".")[0]
                img_path = os.path.join(img_folder_path, name + ".png")
                lines_xml = [l for l in page[0]]
                if set_name == "train":
                    new_set_name = "train" if int(name.split("-")[-1]) < begin_valid_ind else "valid"
                    pg_labels = page[0].attrib["Value"].split("\\n")
                    if 0 in [len(line.attrib.get("Value").strip()) for line in lines_xml]:
                        labels = [l for l in pg_labels if len(l.strip()) != 0]
                    else:
                        labels = [l.attrib.get("Value") for l in lines_xml]
                else:
                    new_set_name = "test"
                    labels = [l.attrib.get("Value") for l in lines_xml]

                labels = [self.format_text_label(self.convert_label(l)) for l in labels]
                for i, (label, line) in enumerate(zip(labels, lines_xml)):
                    lines.append({
                        "text": self.convert_label(label),
                        "left": max(0, int(line.attrib.get("Left"))),
                        "bottom": int(line.attrib.get("Bottom")),
                        "right": int(line.attrib.get("Right")),
                        "top": max(0, int(line.attrib.get("Top"))),
                    })
                    full_text = "{}{}\n".format(full_text, lines[-1]["text"])
                paragraph = {
                    "text": full_text[:-1],
                    "lines": lines,
                    "img_path": img_path
                }
                dataset[new_set_name].append(paragraph)
        return dataset

    def format_rimes_line(self):
        dataset = self.preformat_rimes_paragraph()
        for set_name in self.set_names:
            fold = os.path.join(self.target_fold_path, set_name)
            for sample in dataset[set_name]:
                img = np.array(Image.open(sample["img_path"], "r"))
                for line in sample["lines"]:
                    new_name = "{}_{}.png".format(set_name, len(os.listdir(fold)))
                    new_path = os.path.join(fold, new_name)
                    line_img = img[line["top"]:line["bottom"], line["left"]:line["right"]]
                    line_img = self.resize(line_img, 300, self.dpi)
                    Image.fromarray(line_img).save(new_path)
                    line = {
                        "text": line["text"],
                    }
                    self.charset = self.charset.union(set(line["text"]))
                    self.gt[set_name][new_name] = line

    def preformat_rimes_page(self):
        """
        Extract all information from dataset and correct some annotations
        """
        dataset = {
            "train": list(),
            "valid": list(),
            "test": list()
        }
        img_folder_path = os.path.join(self.temp_fold, "RIMES page", "Images")
        xml_folder_path = os.path.join(self.temp_fold, "RIMES page", "XML")
        xml_files = natural_sort([os.path.join(xml_folder_path, name) for name in os.listdir(xml_folder_path)])
        train_xml = xml_files[:1050]
        valid_xml = xml_files[1050:1150]
        test_xml = xml_files[1150:]
        for set_name, xml_files in zip(self.set_names, [train_xml, valid_xml, test_xml]):
            for i, xml_path in enumerate(xml_files):
                text_regions = list()
                root = ET.parse(xml_path).getroot()
                img_name = root.find("source").text
                if img_name == "01160_L.png":
                    text_regions.append({
                        "label": "LETTRE RECOMMANDEE\nAVEC ACCUSE DE RECEPTION",
                        "type": "",
                        "coords": {
                            "left": 88,
                            "right": 1364,
                            "top": 1224,
                            "bottom": 1448,
                        }
                    })
                for text_region in root.findall("box"):
                    type = text_region.find("type").text
                    label = text_region.find("text").text
                    if label is None or len(label.strip()) <= 0:
                        continue
                    if label == "Ref : QVLCP¨65":
                        label = label.replace("¨", "")
                    if img_name == "01094_L.png" and type == "Corps de texte":
                        label = "Suite à la tempête du 19.11.06, un\narbre est tombé sur mon toît et l'a endommagé.\nJe d'eplore une cinquantaine de tuiles à changer,\nune poutre à réparer et une gouttière à\nremplacer. Veuillez trouver ci-joint le devis\nde réparation. Merci de m'envoyer votre\nexpert le plus rapidement possible.\nEn esperant une réponse rapide de votre\npart, veuillez accepter, madame, monsieur,\nmes salutations distinguées."
                    elif img_name == "01111_L.png" and type == "Corps de texte":
                        label = "Je vous ai envoyé un courrier le 20 octobre 2006\nvous signalant un sinistre survenu dans ma\nmaison, un dégât des eaux consécutif aux\nfortes pluis.\nVous deviez envoyer un expert pour constater\nles dégâts. Personne n'est venu à ce jour\nJe vous prie donc de faire le nécessaire\nafin que les réparations nécessaires puissent\nêtre commencés.\nDans l'attente, veuillez agréer, Monsieur,\nmes sincères salutations"

                    label = self.convert_label_accent(label)
                    label = self.convert_label(label)
                    label = self.format_text_label(label)
                    coords = {
                        "left": int(text_region.attrib["top_left_x"]),
                        "right": int(text_region.attrib["bottom_right_x"]),
                        "top": int(text_region.attrib["top_left_y"]),
                        "bottom": int(text_region.attrib["bottom_right_y"]),
                    }
                    text_regions.append({
                        "label": label,
                        "type": type,
                        "coords": coords
                    })
                text_regions = self.ordering_function(text_regions)
                dataset[set_name].append({
                    "text_regions": text_regions,
                    "img_path": os.path.join(img_folder_path, img_name),
                    "label": "\n".join([tr["label"] for tr in text_regions]),
                    "sem_label": "".join([self.sem_label(tr["label"], tr["type"]) for tr in text_regions]),
                })
        return dataset

    def convert_label_accent(self, label):
        """
        Solve encoding issues
        """
        return label.replace("\\n", "\n").replace("<euro>", "€").replace(">euro>", "€").replace(">fligne>", " ")\
            .replace("Â¤", "¤").replace("Ã»", "û").replace("�", "").replace("ï¿©", "é").replace("Ã§", "ç")\
            .replace("Ã©", "é").replace("Ã´", "ô").replace(u'\xa0', " ").replace("Ã¨", "è").replace("Â°", "°")\
            .replace("Ã", "À").replace("Ã¬", "À").replace("Ãª", "ê").replace("Ã®", "î").replace("Ã¢", "â")\
            .replace("Â²", "²").replace("Ã¹", "ù").replace("Ã", "à").replace("¬", "€")

    def format_rimes_page(self):
        """
        Format RIMES page dataset
        """
        dataset = self.preformat_rimes_page()
        for set_name in self.set_names:
            fold = os.path.join(self.target_fold_path, set_name)
            for sample in dataset[set_name]:
                new_name = "{}_{}.png".format(set_name, len(os.listdir(fold)))
                new_img_path = os.path.join(fold, new_name)
                self.load_resize_save(sample["img_path"], new_img_path, 300, self.dpi)
                for tr in sample["text_regions"]:
                    tr["coords"] = self.adjust_coord_ratio(tr["coords"], self.dpi / 300)
                page = {
                    "text": sample["label"] if not self.sem_token else sample["sem_label"],
                    "paragraphs": sample["text_regions"],
                    "nb_cols": 1,
                }
                self.charset = self.charset.union(set(page["text"]))
                self.gt[set_name][new_name] = page

    def convert_label(self, label):
        """
        Some annotations presents many options for a given text part, always keep the first one only
        """
        if "¤" in label:
            label = re.sub('¤{([^¤]*)[/|]([^¤]*)}¤', r'\1', label, flags=re.DOTALL)
            label = re.sub('¤{([^¤]*)[/|]([^¤]*)[/|]([^¤]*)>', r'\1', label, flags=re.DOTALL)
            label = re.sub('¤([^¤]*)[/|]([^¤]*)¤', r'\1', label, flags=re.DOTALL)
            label = re.sub('¤{}¤([^¤]*)[/|]([^ ]*)', r'\1', label, flags=re.DOTALL)
            label = re.sub('¤{/([^¤]*)/([^ ]*)', r'\1', label, flags=re.DOTALL)
            label = re.sub('¤{([^¤]*)[/|]([^ ]*)', r'\1', label, flags=re.DOTALL)
            label = re.sub('([^¤]*)/(.*)[¤}{]+', r'\1', label, flags=re.DOTALL)
            label = re.sub('[¤}{]+([^¤}{]*)[¤}{]+', r'\1', label, flags=re.DOTALL)
            label = re.sub('¤([^¤]*)¤', r'\1', label, flags=re.DOTALL)
        label = re.sub('[ ]+', " ", label, flags=re.DOTALL)
        label = label.strip()
        return label

    def sem_label(self, label, type):
        """
        Add layout tokens
        """
        if type == "":
            return label
        begin_token = self.matching_tokens_str[type]
        end_token = self.matching_tokens_str[type].upper()
        return begin_token + label + end_token


def order_text_regions(text_regions):
    """
    Establish reading order based on text region pixel positions
    """
    sorted_text_regions = list()
    for tr in text_regions:
        added = False
        if len(sorted_text_regions) == 0:
            sorted_text_regions.append(tr)
            added = True
        else:
            for i, sorted_tr in enumerate(sorted_text_regions):
                tr_height = tr["coords"]["bottom"] - tr["coords"]["top"]
                sorted_tr_height = sorted_tr["coords"]["bottom"] - sorted_tr["coords"]["top"]
                tr_is_totally_above = tr["coords"]["bottom"] < sorted_tr["coords"]["top"]
                tr_is_top_above = tr["coords"]["top"] < sorted_tr["coords"]["top"]
                is_same_level = sorted_tr["coords"]["top"] <= tr["coords"]["bottom"] <= sorted_tr["coords"]["bottom"] or\
                                sorted_tr["coords"]["top"] <= tr["coords"]["top"] <= sorted_tr["coords"]["bottom"] or\
                                tr["coords"]["top"] <= sorted_tr["coords"]["bottom"] <= tr["coords"]["bottom"] or\
                                tr["coords"]["top"] <= sorted_tr["coords"]["top"] <= tr["coords"]["bottom"]
                vertical_shared_space = tr["coords"]["bottom"]-sorted_tr["coords"]["top"] if tr_is_top_above else sorted_tr["coords"]["bottom"]-tr["coords"]["top"]
                reach_same_level_limit = vertical_shared_space > 0.3*min(tr_height, sorted_tr_height)
                is_more_at_left = tr["coords"]["left"] < sorted_tr["coords"]["left"]
                equivalent_height = abs(tr_height-sorted_tr_height) < 0.3*min(tr_height, sorted_tr_height)
                is_middle_above_top = np.mean([tr["coords"]["top"], tr["coords"]["bottom"]]) < sorted_tr["coords"]["top"]
                if tr_is_totally_above or\
                    (is_same_level and equivalent_height and is_more_at_left and reach_same_level_limit) or\
                    (is_same_level and equivalent_height and tr_is_top_above and not reach_same_level_limit) or\
                    (is_same_level and not equivalent_height and is_middle_above_top):
                    sorted_text_regions.insert(i, tr)
                    added = True
                    break
        if not added:
            sorted_text_regions.append(tr)

    return sorted_text_regions


def order_text_regions_sem(text_regions):
    order = ['Coordonnées Expéditeur', 'Date, Lieu', 'Coordonnées Destinataire', 'Reference', 'Objet', 'Ouverture', 'Corps de texte', 'PS/PJ']
    text_region_dict = dict()
    for tr in text_regions:
        if tr["type"] in text_region_dict:
            text_region_dict[tr["type"]].append(tr)
        else:
            text_region_dict[tr["type"]] = [tr, ]
    sorted_text_regions = list()
    for type_name in order:
        if type_name in text_region_dict:
            for tr in order_text_regions(text_region_dict[type_name]):
                sorted_text_regions.append(tr)
    return sorted_text_regions


if __name__ == "__main__":


    # RIMESDatasetFormatter("line").format()
    RIMESDatasetFormatter("page", sem_token=True, order_sem=False).format()