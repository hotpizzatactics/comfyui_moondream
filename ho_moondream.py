"""
@author: AlexL
@title: ComfyUI-Hangover-Moondream
@nickname: Hangover-Moondream
@description: An implementation of the moondream visual LLM
"""
# https://huggingface.co/vikhyatk/moondream2

# by https://github.com/Hangover3832


from transformers import AutoModelForCausalLM as AutoModel, CodeGenTokenizerFast as Tokenizer
from PIL import Image
import torch
import gc
import numpy as np
import codecs
import subprocess
import os
import requests
import re

def Run_git_status(repo:str) -> list[str]:
    """resturns a list of all model tag references for this huggingface repo"""
    url = f"https://huggingface.co/{repo}"
    process = subprocess.Popen(['git', 'ls-remote', url], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    result = []
    if process.returncode == 0:
        revs = stdout.decode().splitlines()
        revs = [r.replace('main', 'latest') for r in revs if ('/tags' in r) or ('/main' in r)]
        for line in revs:
            rev = line.split('\t')
            result.append(f"{rev[-1].split('/')[-1]} -> {rev[0]}")
    return result

class Moondream:
    HUGGINGFACE_MODEL_NAME = "vikhyatk/moondream2"
    DEVICES = ["cpu", "gpu"] if torch.cuda.is_available() else  ["cpu"]
    Versions = 'versions.txt'
    Model_Revisions_URL = f"https://huggingface.co/{HUGGINGFACE_MODEL_NAME}/raw/main/{Versions}"
    current_path = os.path.abspath(os.path.dirname(__file__))
    try:
        print("[Moondream] trying to update model versions...", end='')
        response = requests.get(Model_Revisions_URL)
        if response.status_code == 200:
            with open(f"{current_path}/{Versions}", 'w') as f:
                f.write(response.text)
            print('ok')
    except Exception as e:
        if hasattr(e, 'message'):
            msg = e.message
        else:
            msg = e
        print(f'failed ({msg})')

    with open(f"{current_path}/{Versions}", 'r') as f:
        versions = f.read()
    
    MODEL_REVISIONS = [v for v in versions.splitlines() if v.strip()]
    print(f"[Moondream] found model versions: {', '.join(MODEL_REVISIONS)}")
    MODEL_REVISIONS.insert(0,'ComfyUI/models/moondream2')

    try:
        print('\033[92m\033[4m[Moondream] model revsion references:\033[0m\033[92m')
        git_status = Run_git_status(HUGGINGFACE_MODEL_NAME)
        for s in git_status:
            print(s)
        # return ("",)
    except:
        pass
    finally:
        print('\033[0m')


    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.revision = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Please provide a detailed description of this image."},),
                "separator": ("STRING", {"multiline": False, "default": r"\n"},),
                # "huggingface_model": (s.HUGGINGFACE_MODEL_NAMES, {"default": s.HUGGINGFACE_MODEL_NAMES[-1]},),
                "model_revision": (s.MODEL_REVISIONS, {"default": s.MODEL_REVISIONS[-1]},),
                "temperature": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.},),
                "device": (s.DEVICES, {"default": s.DEVICES[0]},),
                "trust_remote_code": ("BOOLEAN", {"default": False},),
            }
        }

    RETURN_TYPES = ("STRING", "BBOX")
    RETURN_NAMES = ("description", "bbox")
    FUNCTION = "interrogate"
    OUTPUT_NODE = False
    CATEGORY = "Hangover"

    def extract_floats(self, text):
        pattern = r"\[\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\]"
        match = re.search(pattern, text)
        if match:
            return [float(num) for num in match.groups()]
        return None

    def extract_bbox(self, text, image_width, image_height):
        floats = self.extract_floats(text)
        if floats is not None:
            x1, y1, x2, y2 = floats
            # Scale normalized coordinates to image dimensions
            x1 = int(x1 * image_width)
            y1 = int(y1 * image_height)
            x2 = int(x2 * image_width)
            y2 = int(y2 * image_height)
            width = x2 - x1
            height = y2 - y1
            return (x1, y1, width, height)
        return None

    def interrogate(self, image:torch.Tensor, prompt:str, separator:str, model_revision:str, temperature:float, device:str, trust_remote_code:bool):
        if not trust_remote_code:
            raise ValueError("You have to trust remote code to use this node!")

        dev = "cuda" if device.lower() == "gpu" else "cpu"
        if temperature < 0.01:
            temperature = None
            do_sample = None
        else:
            do_sample = True

        if (self.model == None) or (self.tokenizer == None) or (device != self.device) or (model_revision != self.revision):
            del self.model
            del self.tokenizer
            gc.collect()
            if (device == "cpu") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None
            self.revision = model_revision

            print(f"[Moondream] loading model moondream2 revision '{model_revision}', please stand by....")
            if model_revision == Moondream.MODEL_REVISIONS[0]:
                model_name = model_revision
                model_revision = None
            else:
                model_name = Moondream.HUGGINGFACE_MODEL_NAME

            try:
                self.model = AutoModel.from_pretrained(
                    model_name, 
                    trust_remote_code=trust_remote_code,
                    revision=model_revision
                ).to(dev)
                self.tokenizer = Tokenizer.from_pretrained(model_name)
            except RuntimeError:
                raise ValueError(f"[Moondream] Please check if the tramsformer package fulfills the requirements. "
                                  "Also note that older models might not work anymore with newer packages.")

            self.device = device

        descriptions = ""
        bboxes = []
        prompts = list(filter(lambda x: x!="", [s.lstrip() for s in prompt.splitlines()])) # make a prompt list and remove unnecessary whitechars and empty lines
        if len(prompts) == 0:
            prompts = [""]

        try:
            for im in image:
                i = 255. * im.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                width, height = img.size
                enc_image = self.model.encode_image(img)
                descr = ""
                sep = codecs.decode(separator, 'unicode_escape')
                for p in prompts:
                    answer = self.model.answer_question(enc_image, p, self.tokenizer, temperature=temperature, do_sample=do_sample)
                    descr += f"{answer}{sep}"
                    bbox = self.extract_bbox(answer, width, height)
                    if bbox:
                        # Ensure bbox coordinates are within image boundaries
                        x_min, y_min, box_width, box_height = bbox
                        x_min = max(0, min(x_min, width - 1))
                        y_min = max(0, min(y_min, height - 1))
                        box_width = min(box_width, width - x_min)
                        box_height = min(box_height, height - y_min)
                        bbox = (x_min, y_min, box_width, box_height)
                    bboxes.append(bbox)
                descriptions += f"{descr[0:-len(sep)]}\n"
        except RuntimeError:
            raise ValueError(f"[Moondream] Please check if the tramsformer package fulfills the requirements. "
                                  "Also note that older models might not work anymore with newer packages.")
        
        return (descriptions[0:-1], bboxes)

class BboxToMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "bbox": ("BBOX",),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "bbox_to_mask"
    CATEGORY = "Hangover"

    def bbox_to_mask(self, image, bbox):
        # Handle case where bbox is None or an empty list
        if bbox is None or (isinstance(bbox, list) and len(bbox) == 0):
            # Return an empty mask
            c, h, w = image.shape
            return (torch.zeros((1, h, w), dtype=torch.float32),)

        # If bbox is a list of bboxes, take the first one
        if isinstance(bbox, list):
            bbox = bbox[0]

        if not isinstance(bbox, tuple) or len(bbox) != 4:
            raise ValueError("Invalid bbox format. Expected (x, y, width, height)")

        x, y, width, height = bbox
        
        # Get image dimensions
        c, h, w = image.shape
        
        # Create an empty mask
        mask = torch.zeros((h, w), dtype=torch.float32)
        
        # Ensure coordinates are within image boundaries
        x = max(0, min(int(x), w-1))
        y = max(0, min(int(y), h-1))
        width = max(0, min(int(width), w-x))
        height = max(0, min(int(height), h-y))
        
        # Set the bbox region to 1
        mask[y:y+height, x:x+width] = 1.0
        
        return (mask.unsqueeze(0),)  # Return as a 1,H,W tensor