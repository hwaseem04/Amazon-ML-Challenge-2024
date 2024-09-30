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
warnings.filterwarnings("ignore")

from PIL import Image
import torch
from peft import LoraConfig
from datasets import load_dataset, Dataset

from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration

from torchvision import transforms
from PIL import Image
import pandas as pd
import re

# Custom script to remove noisy samples from training. Around 300. So remoing them wont hurt.
def preprocess(dataset):
    # Function to check if the value matches the pattern
    # Set of allowable units
    df1=pd.DataFrame(dtype='object')
    allowable_units = {
        'millimetre', 'kilovolt', 'cubic foot', 'inch', 'pound', 'kilowatt', 'foot',
        'imperial gallon', 'yard', 'volt', 'gallon', 'watt', 'decilitre', 'pint',
        'cubic inch', 'microlitre', 'litre', 'fluid ounce', 'quart', 'metre',
        'millivolt', 'millilitre', 'microgram', 'cup', 'ounce', 'centilitre',
        'centimetre', 'ton', 'milligram', 'gram', 'kilogram'}

    # Function to split entity_value into numeric and units
    def check_valid(value):
        if pd.isna(value) or not isinstance(value, str):
            return value
        # r'^\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]\s*({})$'.format("|".join(allowable_units)),  # [float1, float2] unit
        patterns = [
            r'^(\d+(?:\.\d+)?)\s*({})\s*to\s*(\d+(?:\.\d+)?)\s*({})$'.format("|".join(allowable_units), "|".join(allowable_units)),  # float1 unit to float2 unit
            r'^(\d+(?:\.\d+)?)\s*({})$'.format("|".join(allowable_units))  # float unit
        ]
        
        # Apply the patterns
        for pattern in patterns:
            match = re.match(pattern, value)
            #print(type(value),value)
            if match:
                if len(match.groups()) == 3:  # Case 1: [float1, float2] unit
                    #print(f"{match.group(1)} {match.group(3)}")
                    return f"{match.group(1)} {match.group(3)}"
                elif len(match.groups()) == 4:  # Case 2: float1 unit to float2 unit
                    return f"{match.group(1)} {match.group(2)}"
                elif len(match.groups()) == 2:  # Case 3: float unit
                    return value
        # Values that are removed
        print(value)
        return None

    df = pd.DataFrame(dataset)

    # Check for valid datas
    df['entity_value'] = df['entity_value'].apply(lambda x: pd.Series(check_valid(x)))
    print(df['entity_value'].isna().value_counts())
    df.dropna(inplace=True)
    # print(df)
    print(df['entity_value'].isna().value_counts())
    dataset = Dataset.from_pandas(df)
    return dataset

# Finetuning checkpoint. This number represents iteration and not epoch
iteration = 1845
peft_model_id = f"./old_iter_{iteration}"

# Rotational augmentation introduced to deal with vertical and oriented text
image_transform = transforms.Compose([
    transforms.RandomApply([transforms.RandomRotation(degrees=30)], p=0.2),  # Random rotation between -30 and 30 degrees
])

DEVICE = "cuda:0"
USE_LORA = False
USE_QLORA = True

train_dataset = load_dataset("hwaseem04/sample-amz", split="train")

#TODO 1: process the above data to get clean data
# train_dataset=preprocess(train_dataset)


processor = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    do_image_splitting=False
)

# Three options for training, from the lowest precision training to the highest precision training:
# - QLora
# - Standard Lora
# - Full fine-tuning
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

import random

# def add_special_tokens(s):
#     value = s.split(' ')[0]
#     unit = ' '.join(s.split(' ')[1:])
#     return f'<SOE>{value}<SEP>{unit}<EOT>'

# print(processor.tokenizer.all_special_tokens)
# print(processor.tokenizer.all_special_ids)
# print(processor.tokenizer.eos_token)
# print(processor.tokenizer.bos_token)



def process(x):
    return x.replace('numeric_value entity_unit.', 'numeric_value entity_unit. If the numeric_value is a range, return the first value. Return empty string ONLY IF you cannot find the entity_unit in the image.')

class MyDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            image_path = example["image_link"]
            image = Image.open(image_path)

            #TODO: 3
            if image.size[0] > 980 or image.size[1] > 980:
                image = image.resize((980, 980))

            image = image_transform(image)

            question = process(example["prompt"])

            answer = example["entity_value"]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract appropriate data."},
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                }
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = self.image_token_id
        batch["labels"] = labels

        return batch

data_collator = MyDataCollator(processor)

"""We will use HuggingFace Trainer."""

from transformers import TrainingArguments, Trainer

import os
os.environ["WANDB_PROJECT"]="ml_hack"

training_args = TrainingArguments(
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=1,
    output_dir="./weights_exp4_final",
    save_strategy="steps",
    save_steps=15,
    save_total_limit=10,
    # evaluation_strategy="epoch",
    fp16=True,
    push_to_hub_model_id="ml_hack",
    remove_unused_columns=False,
    report_to="wandb",
    run_name='exp4_experiment_extended',
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset, # You can also evaluate (loss) on the eval set, note that it will incur some additional GPU memory
)

trainer.train()
trainer.push_to_hub()

