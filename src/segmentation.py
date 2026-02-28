import numpy as np
import mediapipe as mp


class PersonSegmenter:

    def __init__(self):
        self.mp_selfie = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie.SelfieSegmentation(model_selection=1)

    def segment_and_whiten(self, image_rgb):

        result = self.segmenter.process(image_rgb)
        mask = result.segmentation_mask
        mask = (mask > 0.5).astype(np.uint8)

        white_bg = np.ones_like(image_rgb) * 255
        segmented = image_rgb * mask[..., None] + white_bg * (1 - mask[..., None])

        ys, xs = np.where(mask == 1)
        if len(xs) == 0:
            return None, None

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        return segmented, (x0, y0, x1, y1)