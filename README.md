# Amazon ML Challenge 2024

*Muhmmad Abrar¹, Mhanjhusriee Baskar¹, and Muhammad Waseem²*

¹Vellore Institute of Technology Chennai
²Shiv Nadar University Chennai

## Problem Statement

This challenge aims to build a Machine learning model that extracts key product details, like weight and dimensions, from images. This is essential in industries like e-commerce, where accurate product data is needed for better listings and inventory management. By automating entity extraction from images, we can reduce the reliance on manual inputs and improve efficiency in digital marketplaces.

The task is to develop a Machine learning model that extracts specific entity values from product images. For each image, the model must predict the value of a queried entity such as width, height, depth, item weight, maximum weight recommendation, voltage, wattage, or item volume. The model should interpret the context from the image based on the queried entity and return the value in the correct unit (e.g., grams, volts, litres).

## Dataset

The train.csv had a total of 263,859 entries and the test.csv had a total of 131,187 entries. Total images for the train set is 255,906 and for test set is 90,666.

## Dataset Pre-Processing

1. **Noisy Data Removal**: Eliminated blank URLs and extreme values to ensure relevant data entries.
2. **Unit Standardization**: Filtered out unrecognized units and Non-Allowable units, retaining only valid ones for consistency.
3. **Resolution of Measurement Ambiguities**: Standardized ranges by selecting the upper bound for uniformity.
4. **Consistent Numeric Formatting**: Reformatted all numbers to standard floating point format.

## Model Used

We used Idefics-2, 8B visual language model from hugging face for fine-tuning. The model uses interleaved text and image tokens to extract the required information from the input image. We therefore didn't use an external OCR model.

## ML approach

Idefics can only handle input dimensions of up to 980 pixels in height and 960 pixels in width. However, the average dimensions of the training images are 1430 pixels in height and 1434 pixels in width, while the test images average 1262 pixels in height and 1256 pixels in width. During fine-tuning, we resized all images with dimensions exceeding 980 pixels in height or width to 980 pixels in both dimensions. Additionally, we applied rotational augmentation to improve the model's ability to recognize text in various orientations.

## Training

We fine-tuned the Idefics2 model using processed images and custom prompt text as input to adapt it to our specific requirements.

## Inference

Given the large volume of test data, performing inference one image at a time would take several hours. To expedite the process, we divided the dataset into several subsets and ran multiple tmux sessions in parallel. This approach reduced the overall inference time from over 14 hours to just 4 hours.

## Post-Processing

After obtaining model predictions, several edge cases required specific handling to ensure accurate outputs:

1. **Range Values**: For predictions with a range (e.g., "10-15 kg"), we selected the highest value to maintain consistency.
2. **Invalid Units**: Predicted invalid units (e.g., "centigram") were replaced with an empty string using the allowed units list.
3. **Invalid Numeric Formats**: Predictions with improper numeric formats (e.g., "0.0.0") were replaced with an empty string to ensure validity.

## Experiments

Due to the extensive volume of training data, our time for experimentation was limited, allowing us to conduct only a few key tests. Here's a summary of the approaches we explored:

1. **Zero-Shot Inference**: We conducted inference with the idefics2 model without any preliminary fine-tuning. This approach allowed us to gauge the model's initial performance on our data without any additional adjustments.
2. **Basic Fine-Tuning**: After meticulously preprocessing the data, we performed a fine-tuning procedure using a straightforward text prompt. This method aimed to adapt the model to our specific dataset with minimal intervention.
3. **Custom Fine-Tuning**: We employed a tailored prompt template in conjunction with rotation augmentation techniques. This custom approach was designed to enhance the model's performance by introducing additional complexity and variability into the fine-tuning process.

These experiments were intended to evaluate different levels of model adaptation and assess their impact on performance within the constraints of our timeframe.

### Experiment Results

| Experiment | F1 Score |
|------------|----------|
| Without Finetuning | 0.44 |
| Vanilla Finetuning | 0.56 |
| Custom Prompted finetuning | 0.616 |

## Conclusion

In this challenge, we successfully developed a machine learning model capable of extracting key product attributes from images, automating a crucial task for industries like e-commerce. By leveraging the powerful Idefics-2 model and applying tailored pre-processing techniques, such as unit standardization and numeric formatting, we improved the accuracy and consistency of the predictions. Fine-tuning the model with custom prompts and rotational augmentation significantly enhanced its performance, leading to an F1 score improvement from 0.44 to 0.616. Through efficient inference parallelization and post-processing strategies, we also optimized the model's response time and handled complex edge cases. Future work could involve further experimentation and ablation with different model architectures and optimization techniques to push the boundaries of performance even further.
