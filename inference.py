# Taken and modified from: https://colab.research.google.com/drive/1NtcTgRbSBKN7pYD3Vdx1j9m8pt3fhFDB

"""
To accomodate the GPU poors, the default hyper-parameters in this tutorial are chosen so that the fine-tuning takes less than 32 GB of GPU memory. For instance, an V100 in Google Colab should be sufficient.

If you happen to have more ressources, you are encouraged to revisit some of these constraints, in particular:
- Using 4 bit quantization
- Lora fine-tuning
- Freezing the vision encoder
- Small batch size compensated with higher gradient accumulation degree
- Deactivate image splitting
- Using flash-attention
"""
import warnings
from PIL import Image
warnings.filterwarnings("ignore")

import torch
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration

DEVICE = "cuda:0"
USE_LORA = False
USE_QLORA = True

iteration = 180
peft_model_id = f"./iter_{iteration}"

processor = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    do_image_splitting=False
)

import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Set parameters for s and part.")
parser.add_argument('--s', type=int, default=0, help='Value for s')
parser.add_argument('--part', type=int, default=16399, help='Value for part')

# Parse arguments
args = parser.parse_args()

# Assign values from command line arguments
s = args.s
part = args.part

# Three options for training, from the lowest precision training to the highest precision training:
# - QLora
# - Standard Lora
# - Full fine-tuning

def process(x):
    return x.replace('numeric_value entity_unit.', 'numeric_value entity_unit. If the numeric_value is a range, return the first value. Return empty string ONLY IF you cannot find the entity_unit in the image.')


if USE_QLORA or USE_LORA:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules='.*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
        use_dora=False if USE_QLORA else True,
        init_lora_weights="gaussian"
    )
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    model = Idefics2ForConditionalGeneration.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        torch_dtype=torch.float16,
        quantization_config=bnb_config if USE_QLORA else None,
    )
    model.load_adapter(peft_model_id)
    model.enable_adapters()
else:
    model = Idefics2ForConditionalGeneration.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2", # Only available on A100 or H100
    ).to(DEVICE)

from datasets import load_dataset

# test_dataset = load_dataset("hwaseem04/docedit-new", split="test")
test_dataset = load_dataset("hwaseem04/sample-amz", split="test")
print(test_dataset)
print(test_dataset[:1])

from transformers import StoppingCriteria, StoppingCriteriaList
class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence = [32002]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids

# Inference
import csv
from tqdm import tqdm 

# Batch size

# def add_special_tokens(s):
#     value = s.split(' ')[0]
#     unit = ' '.join(s.split(' ')[1:])
#     return f'<SOE>{value}<SEP>{unit}<EOT>'

# Open the CSV and text files
with open(f"inference_csv/output_{iteration}_s{s}.csv", "w", newline="") as csvfile:
    # Define the column headers
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["index", "prediction"])

    for i in tqdm(range(s, part+1)):
        example = test_dataset[i]

        model.eval()

        image_path = example["image_link"]
        image = Image.open(image_path)
        if image.size[0] > 980 or image.size[1] > 980:
            image = image.resize((980, 980))
        query = process(example["prompt"])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract appropriate data."},
                    {"type": "image"},
                    {"type": "text", "text": query}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text.strip()], images=[image], return_tensors="pt", padding=True)
        
        generated_ids = model.generate(**inputs, max_new_tokens=80, stopping_criteria=[EosListStoppingCriteria()])
        

        generated_texts = processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):])

        # Append the data to the CSV file
        csvwriter.writerow([i, generated_texts[0].replace('<end_of_utterance>', '')])