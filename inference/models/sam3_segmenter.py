from __future__ import annotations

from typing import Dict, List, Optional, Any

import torch
from PIL import Image
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.utils.misc import copy_data_to_device
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api as collate
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image as SAMImage,
    InferenceMetadata,
)
from sam3.train.transforms.basic_for_api import ComposeAPI, NormalizeAPI, RandomResizeAPI, ToTensorAPI

_GLOBAL_COUNTER = 1


def _create_empty_datapoint() -> Datapoint:
    return Datapoint(find_queries=[], images=[])


def _set_image(datapoint: Datapoint, pil_image: Image.Image) -> None:
    w, h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]


def _add_text_prompt(datapoint: Datapoint, text_query: str) -> None:
    global _GLOBAL_COUNTER
    assert len(datapoint.images) == 1, "please set the image first"
    w, h = datapoint.images[0].size

    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=_GLOBAL_COUNTER,
                original_image_id=_GLOBAL_COUNTER,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            ),
        )
    )
    _GLOBAL_COUNTER += 1


def _add_visual_prompt(
    datapoint: Datapoint,
    boxes: List[List[float]],
    labels: List[bool],
    text_prompt: str = "visual",
) -> None:
    global _GLOBAL_COUNTER
    assert len(datapoint.images) == 1, "please set the image first"
    assert len(boxes) > 0, "please provide at least one box"
    assert len(boxes) == len(labels), "Expecting one label per box"

    for b in boxes:
        assert len(b) == 4, "Boxes must have 4 coordinates"

    labels_tensor = torch.tensor(labels, dtype=torch.bool).view(-1)
    w, h = datapoint.images[0].size

    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_prompt,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            input_bbox=torch.tensor(boxes, dtype=torch.float).view(-1, 4),
            input_bbox_label=labels_tensor,
            inference_metadata=InferenceMetadata(
                coco_image_id=_GLOBAL_COUNTER,
                original_image_id=_GLOBAL_COUNTER,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            ),
        )
    )
    _GLOBAL_COUNTER += 1


class SAM3Segmenter:
    def __init__(
        self,
        bpe_path: str | None = None,
        checkpoint_path: str | None = None,
        detection_threshold: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=checkpoint_path,
        )
        self.postprocessor = PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=detection_threshold,
            to_cpu=True,
        )
        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def segment(
        self,
        image: Image.Image | List[Image.Image],
        text_prompt: str = "a PV Panel",
        geometry_prompt: List[List[float]] | List[Optional[List[List[float]]]] | None = None,
        geometry_labels: List[bool] | List[Optional[List[bool]]] | None = None,
    ) -> List[Dict]:
        if not isinstance(image, list):
            image = [image]

        def _looks_like_boxes(value: Any) -> bool:
            if not isinstance(value, list) or not value:
                return False
            first = value[0]
            if not isinstance(first, (list, tuple)) or len(first) != 4:
                return False
            return all(isinstance(v, (int, float)) for v in first)

        prompts_per_image: List[Optional[List[List[float]]]]
        labels_per_image: List[Optional[List[bool]]]

        if geometry_prompt is None:
            prompts_per_image = [None] * len(image)
        elif len(image) > 1 and _looks_like_boxes(geometry_prompt):
            prompts_per_image = [geometry_prompt] * len(image)
        elif len(image) > 1 and isinstance(geometry_prompt, list) and len(geometry_prompt) == len(image):
            prompts_per_image = list(geometry_prompt)
        else:
            prompts_per_image = [geometry_prompt] * len(image)

        if geometry_labels is None:
            labels_per_image = [None] * len(image)
        elif len(image) > 1 and isinstance(geometry_labels, list) and len(geometry_labels) == len(image):
            labels_per_image = list(geometry_labels)
        else:
            labels_per_image = [geometry_labels] * len(image)

        datapoints = []
        for idx, img in enumerate(image):
            dp = _create_empty_datapoint()
            _set_image(dp, img)
            prompts = prompts_per_image[idx]
            labels = labels_per_image[idx]
            if prompts is not None:
                if (
                    isinstance(prompts, list)
                    and len(prompts) == 1
                    and isinstance(prompts[0], list)
                    and prompts[0]
                    and isinstance(prompts[0][0], (list, tuple))
                ):
                    prompts = prompts[0]
                if labels is None:
                    labels = [True] * len(prompts)
                if (
                    isinstance(labels, list)
                    and len(labels) == 1
                    and isinstance(labels[0], list)
                ):
                    labels = labels[0]
                _add_visual_prompt(dp, prompts, labels, text_prompt=text_prompt)
            else:
                _add_text_prompt(dp, text_prompt)
            dp = self.transform(dp)
            datapoints.append(dp)

        batch = collate(datapoints, dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, self.device, non_blocking=True)

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = self.model(batch)

        results = self.postprocessor.process_results(outputs, batch.find_metadatas)

        global _GLOBAL_COUNTER
        _GLOBAL_COUNTER = 1

        return list(results.values())
