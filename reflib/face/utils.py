import cv2
import os
import os.path as osp
import numpy as np
import torch
from PIL import Image
from torch.hub import download_url_to_file, get_dir
from urllib.parse import urlparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CFaceUtils:

    def __init__(self ):
        self.mydata = 0

    def compute_increased_bbox(self, bbox, increase_area, preserve_aspect=True):
        left, top, right, bot = bbox
        width = right - left
        height = bot - top

        if preserve_aspect:
            width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
            height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))
        else:
            width_increase = height_increase = increase_area
        left = int(left - width_increase * width)
        top = int(top - height_increase * height)
        right = int(right + width_increase * width)
        bot = int(bot + height_increase * height)
        return (left, top, right, bot)


    def get_valid_bboxes(self, bboxes, h, w):
        left = max(bboxes[0], 0)
        top = max(bboxes[1], 0)
        right = min(bboxes[2], w)
        bottom = min(bboxes[3], h)
        return (left, top, right, bottom)


    def align_crop_face_landmarks(self,
                                  img,
                                  landmarks,
                                  output_size,
                                  transform_size=None,
                                  enable_padding=True,
                                  return_inverse_affine=False,
                                  shrink_ratio=(1, 1)):

        lm_type = 'retinaface_5'  # Options: dlib_5, retinaface_5

        if isinstance(shrink_ratio, (float, int)):
            shrink_ratio = (shrink_ratio, shrink_ratio)
        if transform_size is None:
            transform_size = output_size * 4

        # Parse landmarks
        lm = np.array(landmarks)
        if lm.shape[0] == 5 and lm_type == 'retinaface_5':
            eye_left = lm[0]
            eye_right = lm[1]
            mouth_avg = (lm[3] + lm[4]) * 0.5
        elif lm.shape[0] == 5 and lm_type == 'dlib_5':
            lm_eye_left = lm[2:4]
            lm_eye_right = lm[0:2]
            eye_left = np.mean(lm_eye_left, axis=0)
            eye_right = np.mean(lm_eye_right, axis=0)
            mouth_avg = lm[4]
        elif lm.shape[0] == 68:
            lm_eye_left = lm[36:42]
            lm_eye_right = lm[42:48]
            eye_left = np.mean(lm_eye_left, axis=0)
            eye_right = np.mean(lm_eye_right, axis=0)
            mouth_avg = (lm[48] + lm[54]) * 0.5
        elif lm.shape[0] == 98:
            lm_eye_left = lm[60:68]
            lm_eye_right = lm[68:76]
            eye_left = np.mean(lm_eye_left, axis=0)
            eye_right = np.mean(lm_eye_right, axis=0)
            mouth_avg = (lm[76] + lm[82]) * 0.5

        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        eye_to_mouth = mouth_avg - eye_avg

        # Get the oriented crop rectangle
        # x: half width of the oriented crop rectangle
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        #  - np.flipud(eye_to_mouth) * [-1, 1]: rotate 90 clockwise
        # norm with the hypotenuse: get the direction
        x /= np.hypot(*x)  # get the hypotenuse of a right triangle
        rect_scale = 1  # TODO: you can edit it to get larger rect
        x *= max(np.hypot(*eye_to_eye) * 2.0 * rect_scale, np.hypot(*eye_to_mouth) * 1.8 * rect_scale)
        # y: half height of the oriented crop rectangle
        y = np.flipud(x) * [-1, 1]

        x *= shrink_ratio[1]  # width
        y *= shrink_ratio[0]  # height

        # c: center
        c = eye_avg + eye_to_mouth * 0.1
        # quad: (left_top, left_bottom, right_bottom, right_top)
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        # qsize: side length of the square
        qsize = np.hypot(*x) * 2

        quad_ori = np.copy(quad)
        # Shrink, for large face
        # TODO: do we really need shrink
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            h, w = img.shape[0:2]
            rsize = (int(np.rint(float(w) / shrink)), int(np.rint(float(h) / shrink)))
            img = cv2.resize(img, rsize, interpolation=cv2.INTER_AREA)
            quad /= shrink
            qsize /= shrink

        # Crop
        h, w = img.shape[0:2]
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, w), min(crop[3] + border, h))
        if crop[2] - crop[0] < w or crop[3] - crop[1] < h:
            img = img[crop[1]:crop[3], crop[0]:crop[2], :]
            quad -= crop[0:2]

        # Pad
        # pad: (width_left, height_top, width_right, height_bottom)
        h, w = img.shape[0:2]
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
               int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - w + border, 0), max(pad[3] - h + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(img, ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w = img.shape[0:2]
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0],
                                               np.float32(w - 1 - x) / pad[2]),
                              1.0 - np.minimum(np.float32(y) / pad[1],
                                               np.float32(h - 1 - y) / pad[3]))
            blur = int(qsize * 0.02)
            if blur % 2 == 0:
                blur += 1
            blur_img = cv2.boxFilter(img, 0, ksize=(blur, blur))

            img = img.astype('float32')
            img += (blur_img - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = np.clip(img, 0, 255)  # float32, [0, 255]
            quad += pad[:2]

        # Transform use cv2
        h_ratio = shrink_ratio[0] / shrink_ratio[1]
        dst_h, dst_w = int(transform_size * h_ratio), transform_size
        template = np.array([[0, 0], [0, dst_h], [dst_w, dst_h], [dst_w, 0]])
        # use cv2.LMEDS method for the equivalence to skimage transform
        # ref: https://blog.csdn.net/yichxi/article/details/115827338
        affine_matrix = cv2.estimateAffinePartial2D(quad, template, method=cv2.LMEDS)[0]
        cropped_face = cv2.warpAffine(
            img, affine_matrix, (dst_w, dst_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))  # gray

        if output_size < transform_size:
            cropped_face = cv2.resize(
                cropped_face, (output_size, int(output_size * h_ratio)), interpolation=cv2.INTER_LINEAR)

        if return_inverse_affine:
            dst_h, dst_w = int(output_size * h_ratio), output_size
            template = np.array([[0, 0], [0, dst_h], [dst_w, dst_h], [dst_w, 0]])
            # use cv2.LMEDS method for the equivalence to skimage transform
            # ref: https://blog.csdn.net/yichxi/article/details/115827338
            affine_matrix = cv2.estimateAffinePartial2D(
                quad_ori, np.array([[0, 0], [0, output_size], [dst_w, dst_h], [dst_w, 0]]), method=cv2.LMEDS)[0]
            inverse_affine = cv2.invertAffineTransform(affine_matrix)
        else:
            inverse_affine = None
        return cropped_face, inverse_affine


    def paste_face_back(self, img, face, inverse_affine):
        h, w = img.shape[0:2]
        face_h, face_w = face.shape[0:2]
        inv_restored = cv2.warpAffine(face, inverse_affine, (w, h))
        mask = np.ones((face_h, face_w, 3), dtype=np.float32)
        inv_mask = cv2.warpAffine(mask, inverse_affine, (w, h))
        # remove the black borders
        inv_mask_erosion = cv2.erode(inv_mask, np.ones((2, 2), np.uint8))
        inv_restored_remove_border = inv_mask_erosion * inv_restored
        total_face_area = np.sum(inv_mask_erosion) // 3
        # compute the fusion edge based on the area of face
        w_edge = int(total_face_area**0.5) // 20
        erosion_radius = w_edge * 2
        inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
        blur_size = w_edge * 2
        inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
        img = inv_soft_mask * inv_restored_remove_border + (1 - inv_soft_mask) * img
        # float32, [0, 255]
        return img

    def download_pretrained_models(self, file_ids, save_path_root):
        import gdown
        
        os.makedirs(save_path_root, exist_ok=True)

        for file_name, file_id in file_ids.items():
            file_url = 'https://drive.google.com/uc?id='+file_id
            save_path = osp.abspath(osp.join(save_path_root, file_name))
            if osp.exists(save_path):
                user_response = input(f'{file_name} already exist. Do you want to cover it? Y/N\n')
                if user_response.lower() == 'y':
                    print(f'Covering {file_name} to {save_path}')
                    gdown.download(file_url, save_path, quiet=False)
                    # download_file_from_google_drive(file_id, save_path)
                elif user_response.lower() == 'n':
                    print(f'Skipping {file_name}')
                else:
                    raise ValueError('Wrong input. Only accepts Y/N.')
            else:
                print(f'Downloading {file_name} to {save_path}')
                gdown.download(file_url, save_path, quiet=False)
                # download_file_from_google_drive(file_id, save_path)


    def imwrite(self, img, file_path, params=None, auto_mkdir=True):
        """Write image to file.

        Args:
            img (ndarray): Image array to be written.
            file_path (str): Image file path.
            params (None or list): Same as opencv's :func:`imwrite` interface.
            auto_mkdir (bool): If the parent folder of `file_path` does not exist,
                whether to create it automatically.

        Returns:
            bool: Successful or not.
        """
        if auto_mkdir:
            dir_name = os.path.abspath(os.path.dirname(file_path))
            os.makedirs(dir_name, exist_ok=True)
        return cv2.imwrite(file_path, img, params)


    def img2tensor(self, imgs, bgr2rgb=True, float32=True):
        """Numpy array to tensor.

        Args:
            imgs (list[ndarray] | ndarray): Input images.
            bgr2rgb (bool): Whether to change bgr to rgb.
            float32 (bool): Whether to change to float32.

        Returns:
            list[tensor] | tensor: Tensor images. If returned results only have
                one element, just return tensor.
        """

        def _totensor(img, bgr2rgb, float32):
            if img.shape[2] == 3 and bgr2rgb:
                if img.dtype == 'float64':
                    img = img.astype('float32')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img.transpose(2, 0, 1))
            if float32:
                img = img.float()
            return img

        if isinstance(imgs, list):
            return [_totensor(img, bgr2rgb, float32) for img in imgs]
        else:
            return _totensor(imgs, bgr2rgb, float32)


    def load_file_from_url(self, url, model_dir=None, progress=True, file_name=None):
        """Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py
        """
        if model_dir is None:
            hub_dir = get_dir()
            model_dir = os.path.join(hub_dir, 'checkpoints')

        os.makedirs(os.path.join(ROOT_DIR, model_dir), exist_ok=True)

        parts = urlparse(url)
        filename = os.path.basename(parts.path)
        if file_name is not None:
            filename = file_name
        cached_file = os.path.abspath(os.path.join(ROOT_DIR, model_dir, filename))
        if not os.path.exists(cached_file):
            print(f'Downloading: "{url}" to {cached_file}\n')
            download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
        return cached_file


    def scandir(self, dir_path, suffix=None, recursive=False, full_path=False):
        """Scan a directory to find the interested files.
        Args:
            dir_path (str): Path of the directory.
            suffix (str | tuple(str), optional): File suffix that we are
                interested in. Default: None.
            recursive (bool, optional): If set to True, recursively scan the
                directory. Default: False.
            full_path (bool, optional): If set to True, include the dir_path.
                Default: False.
        Returns:
            A generator for all the interested files with relative paths.
        """

        if (suffix is not None) and not isinstance(suffix, (str, tuple)):
            raise TypeError('"suffix" must be a string or tuple of strings')

        root = dir_path

        def _scandir(dir_path, suffix, recursive):
            for entry in os.scandir(dir_path):
                if not entry.name.startswith('.') and entry.is_file():
                    if full_path:
                        return_path = entry.path
                    else:
                        return_path = osp.relpath(entry.path, root)

                    if suffix is None:
                        yield return_path
                    elif return_path.endswith(suffix):
                        yield return_path
                else:
                    if recursive:
                        yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                    else:
                        continue

        return _scandir(dir_path, suffix=suffix, recursive=recursive)


    def is_gray(self, img, threshold=10):
        img = Image.fromarray(img)
        if len(img.getbands()) == 1:
            return True
        img1 = np.asarray(img.getchannel(channel=0), dtype=np.int16)
        img2 = np.asarray(img.getchannel(channel=1), dtype=np.int16)
        img3 = np.asarray(img.getchannel(channel=2), dtype=np.int16)
        diff1 = (img1 - img2).var()
        diff2 = (img2 - img3).var()
        diff3 = (img3 - img1).var()
        diff_sum = (diff1 + diff2 + diff3) / 3.0
        if diff_sum <= threshold:
            return True
        else:
            return False

    def rgb2gray(self, img, out_channel=3):
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        if out_channel == 3:
            gray = gray[:,:,np.newaxis].repeat(3, axis=2)
        return gray

    def bgr2gray(self, img, out_channel=3):
        b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        if out_channel == 3:
            gray = gray[:,:,np.newaxis].repeat(3, axis=2)
        return gray


    def calc_mean_std(self, feat, eps=1e-5):
        """
        Args:
            feat (numpy): 3D [w h c]s
        """
        size = feat.shape
        assert len(size) == 3, 'The input feature should be 3D tensor.'
        c = size[2]
        feat_var = feat.reshape(-1, c).var(axis=0) + eps
        feat_std = np.sqrt(feat_var).reshape(1, 1, c)
        feat_mean = feat.reshape(-1, c).mean(axis=0).reshape(1, 1, c)
        return feat_mean, feat_std


    def adain_npy(self, content_feat, style_feat):
        """Adaptive instance normalization for numpy.

        Args:
            content_feat (numpy): The input feature.
            style_feat (numpy): The reference feature.
        """
        size = content_feat.shape
        style_mean, style_std = calc_mean_std(style_feat)
        content_mean, content_std = calc_mean_std(content_feat)
        normalized_feat = (content_feat - np.broadcast_to(content_mean, size)) / np.broadcast_to(content_std, size)
        return normalized_feat * np.broadcast_to(style_std, size) + np.broadcast_to(style_mean, size)