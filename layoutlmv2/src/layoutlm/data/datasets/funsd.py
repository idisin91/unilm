# coding=utf-8

import json
import os

import datasets
import torch

from detectron2.data.detection_utils import read_image
from detectron2.data.transforms import ResizeTransform, TransformList
import pandas as pd

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{Jaume2019FUNSDAD,
  title={FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents},
  author={Guillaume Jaume and H. K. Ekenel and J. Thiran},
  journal={2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)},
  year={2019},
  volume={2},
  pages={1-6}
}
"""

_DESCRIPTION = """\
https://guillaumejaume.github.io/FUNSD/
"""




from functools import cmp_to_key
from bbox import BBox2D, XYXY
from bbox.metrics import jaccard_index_2d
import random

def compare_bboxes(el1, el2, special_line_order = True):
    return random.random()-0.5 # if >0 -- swap two elements
    
def rearrange_data(*list_of_lists, key, reverse=True):
    arranged_list = list(zip(*list_of_lists))
    sorted_arranged_list = sorted(arranged_list, key=key, reverse=reverse)
    return list(map(list, zip(*sorted_arranged_list)))














def get_line_annotations_for_path(path):
    results = {}
    with open(path, "r") as f:
        df = pd.DataFrame(json.load(f))
    for index, row in df.iterrows():
        all_lines = []
        if 'objects' in row['Label']:
            for obj in row['Label']['objects']:
                x0,y0 = obj['line'][0].values()
                x1,y1 = obj['line'][1].values()
                if x1 < x0:
                    x0,y0,x1,y1 = x1,y1,x0,y0
                all_lines.append((x0,y0,x1,y1))


        filename = row['External ID']
        results[filename] = all_lines
    return results

class FunsdConfig(datasets.BuilderConfig):
    """BuilderConfig for FUNSD"""

    def __init__(self, **kwargs):
        """BuilderConfig for FUNSD.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(FunsdConfig, self).__init__(**kwargs)


class Funsd(datasets.GeneratorBasedBuilder):
    """Conll2003 dataset."""

    BUILDER_CONFIGS = [
        FunsdConfig(name="funsd", version=datasets.Version("1.0.0"), description="FUNSD dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                }
            ),
            supervised_keys=None,
            homepage="https://guillaumejaume.github.io/FUNSD/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # downloaded_file = dl_manager.download_and_extract("https://guillaumejaume.github.io/FUNSD/dataset.zip")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"/gdrive/MyDrive/FUNSD/training_data/"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": f"/gdrive/MyDrive/FUNSD/testing_data/"}
            ),
        ]
    def _generate_examples(self, filepath):
        ann_dir = os.path.join(filepath, "adjusted_annotations")
        img_dir = os.path.join(filepath, "images")
        # line_ann = f"{filepath}/lines.json"
        # line_annotations = get_line_annotations_for_path(line_ann)
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
                tokens = []
                bboxes = []
                ner_tags = []
                file_path = os.path.join(ann_dir, file)
                # basename = os.path.basename(file_path)
                # line_annotation_name = f"{basename[:-5]}.png" # .json -> .png
                with open(file_path, "r", encoding="utf8") as f:
                    data = json.load(f)
                image_path = os.path.join(img_dir, file)
                image_path = image_path.replace("json", "png")
                image, size = load_image(image_path)
                # for line in line_annotations[line_annotation_name]: # for every line annotation in path
                #         x0,y0,x1,y1 = map(int, line)
                #         if y0 > y1:
                #                 y0,y1 = y1,y0
                #         if y0 == y1:
                #                 y1 = y0+1
                #         tokens.append("<LINE>")
                #         ner_tags.append("LINE")
                #         bbox = [x0,y0,x1,y1]
                #         bboxes.append(normalize_bbox(bbox, size))

                for item in data["form"]:
                        words, label = item["words"], item["label"]
                        words = [w for w in words if "text" in w and w["text"].strip() != ""] # some boxes may be empty after revision
                        if len(words) == 0:
                                continue
                        if label == "other":
                                for w in words:
                                        tokens.append(w["text"])
                                        ner_tags.append("O")
                                        bboxes.append(normalize_bbox(w["box"], size))
                        else:
                                tokens.append(words[0]["text"])
                                ner_tags.append("B-" + label.upper())
                                bboxes.append(normalize_bbox(words[0]["box"], size))
                                for w in words[1:]:
                                        tokens.append(w["text"])
                                        ner_tags.append("I-" + label.upper())
                                        bboxes.append(normalize_bbox(w["box"], size))
                tokens, bboxes, ner_tags = rearrange_data(tokens, bboxes, ner_tags, key=cmp_to_key(compare_bboxes))
                
                yield guid, {"id": str(guid), "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags, "image": image}


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def load_image(image_path):
    image = read_image(image_path, format="BGR")
    h = image.shape[0]
    w = image.shape[1]
    img_trans = TransformList([ResizeTransform(h=h, w=w, new_h=224, new_w=224)])
    image = torch.tensor(img_trans.apply_image(image).copy()).permute(2, 0, 1)  # copy to make it writeable
    return image, (w, h)
