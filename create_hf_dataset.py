import os
import pandas as pd
from datasets import Dataset
from PIL import Image
from constants import entity_unit_map

DATASET_FOLDER = '../dataset/'
train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'), index_col=False)

from utils import download_images
download_images(train['image_link'], '../images/train_images')
download_images(test['image_link'], '../images/test_images')


# Few links are invalid, Hence those images are stored as blank canvas. Those are found
from PIL import Image
import os
def find_blank_images(folder_path):
    blank_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image file extensions
            image_path = os.path.join(folder_path, filename)
            with Image.open(image_path) as img:
                if img.size == (100, 100) and img.getpixel((0, 0)) == (0, 0, 0):  # Check for black placeholder
                    blank_images.append(image_path)
    return blank_images

blank_train_images = find_blank_images('../images/train_images')

# these 18 images are removed
# ['../images/train_images/9BIu8SYSAek.jpg',
#  '../images/train_images/VCEdbX8DT28.jpg',
#  '../images/train_images/lwd2cSmT2ux.jpg',
#  '../images/train_images/J2DXsUjR8ay.jpg',
#  '../images/train_images/1yw53vfQtS.jpg',
#  '../images/train_images/T8hQGdjTcGp.jpg',
#  '../images/train_images/VjCkaPeR1o.jpg',
#  '../images/train_images/DzP2RMRQO0.jpg',
#  '../images/train_images/H8fMd0pRI6n.jpg',
#  '../images/train_images/caDEyEaRMCm.jpg',
#  '../images/train_images/mWyQ79S76i.jpg',
#  '../images/train_images/3sSrJnc5R58.jpg',
#  '../images/train_images/PBWKX4CRl2o.jpg',
#  '../images/train_images/fUyC7fnSnys.jpg',
#  '../images/train_images/VRs4UBsSHaM.jpg',
#  '../images/train_images/RBE3EPzT4OZ.jpg',
#  '../images/train_images/BEJwJEFSTSp.jpg',
#  '../images/train_images/l8BsJVaKRCe.jpg']

blank_test_images = find_blank_images('../images/test_images') # There are no such blank images

# Blank images are removed
def remove_empty_img_data(df, blank_images):
    for file in blank_images:
        file_name = file.split('/')[-1]  # Extract the file name        
        df = df[~df['image_link'].str.contains(file_name)]  # Use ~ to negate the condition
    return df

clean_train = remove_empty_img_data(train.copy(), blank_train_images)


# Creating additional column with local paths
def replace_image_links_with_paths(df, image_folder):
    df['amz_link'] = df['image_link']
    for index, row in df.iterrows():
        file_name = row['image_link'].split('/')[-1]  # Extract the file name from the image_link
        image_path = os.path.join(image_folder, file_name)  # Construct the full image path
        
        # Replace the image_link with the image path
        # print(image_path)
        if os.path.exists(image_path):
            df.at[index, 'image_link'] = image_path  # Update the DataFrame with the new path
    return df

new_train = replace_image_links_with_paths(clean_train.copy(), '../images/resize_train_images/')
new_test = replace_image_links_with_paths(test.copy(), '../images/resize_test_images/')
new_test['entity_value'] = '' # empty placeholder


# Prompt template for VLM
def create_prompt(x):
    s = f'Extract {x} from the image. The output should be of format: numeric_value entity_unit. Entity_unit should be strictly one among {entity_unit_map[x]}'
    return s

new_train.reset_index(drop=False, inplace=True)

# Create prompts
new_train['prompt'] = new_train['entity_name'].apply(create_prompt)
new_test['prompt'] = new_test['entity_name'].apply(create_prompt)

# Dataset uploaded to HF repository. This is just for ease of access
train_hf = Dataset.from_pandas(new_train)
test_hf = Dataset.from_pandas(new_test)
train_hf.push_to_hub("hwaseem04/sample-amz", split='train')
test_hf.push_to_hub("hwaseem04/sample-amz", split='test')

# new_train.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 263841 entries, 0 to 263840
# Data columns (total 8 columns):
#  #   Column        Non-Null Count   Dtype 
# ---  ------        --------------   ----- 
#  1   index         263841 non-null  int64 
#  2   image_link    263841 non-null  object
#  3   group_id      263841 non-null  int64 
#  4   entity_name   263841 non-null  object
#  5   entity_value  263841 non-null  object
#  6   amz_link      263841 non-null  object
#  7   prompt        263841 non-null  object
# dtypes: int64(3), object(5)
# memory usage: 16.1+ MB

# new_test.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 131187 entries, 0 to 131186
# Data columns (total 7 columns):
#  #   Column        Non-Null Count   Dtype 
# ---  ------        --------------   ----- 
#  0   index         131187 non-null  int64 
#  1   image_link    131187 non-null  object
#  2   group_id      131187 non-null  int64 
#  3   entity_name   131187 non-null  object
#  4   amz_link      131187 non-null  object
#  5   prompt        131187 non-null  object
#  6   entity_value  131187 non-null  object
# dtypes: int64(2), object(5)
# memory usage: 7.0+ MB