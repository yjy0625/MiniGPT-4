# MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models

This is a fork of the original [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) repo. Please follow installation instructions in the "Installation" section to setup the repo.

## Installation

**1. Prepare the code and the environment**

In the existing conda environment, run the following command to install necessary packages:

```bash
conda env update -f environment.yml
```

**2. Prepare the pretrained Vicuna weights**

We will utilize a version of MiniGPT-4 built on the v0 version of Vicuna-7B.
Please refer to our instruction [here](PrepareVicuna.md) to prepare the Vicuna weights.
The final weights would be in a single folder in a structure similar to the following:

```
vicuna_weights
├── config.json
├── generation_config.json
├── pytorch_model.bin.index.json
├── pytorch_model-00001-of-00003.bin
...   
```

Then, set the path to the vicuna weight in the model config file 
[here](minigpt4/configs/models/minigpt4.yaml#L16) at Line 16.

**3. Prepare the pretrained MiniGPT-4 checkpoint**

Download the pretrained MiniGPT-4 checkpoint using the following [link](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing).

Then, set the path to the pretrained checkpoint in the evaluation config file 
in [eval_configs/minigpt4_eval.yaml](eval_configs/minigpt4_eval.yaml#L10) at Line 13. 

## Fine-tuning MiniGPT 4 on a Dataset

In the RoboFuME work, we use a small high quality image-reward pair dataset created from the prior and target demo data
and convert it to a format compatible with MiniGPT-4.

### Generating Fine-tuning Data for MiniGPT-4

The code used for generating MiniGPT-4 data is located at `robofume/scripts/data_processing/generate_minigpt_data.py`.

Since the demo data only includes task description data but MiniGPT-4 prompting requires questions instead of task descriptions, we need to use the OpenAI API to convert task descriptions to task descriptions. In order to use the OpenAI API, enter the `generate_minigpt_data.py` file and insert the OpenAI organization name and API key in lines 13 and 14.

After setting up OpenAI API, we can generate fine-tuning data in the sim weight task by entering the directory containing the `generate_minigpt_data.py` file and running the following command:
```
python generate_minigpt_data.py --env_name sim_weight --num_demos_per_task 20 --out_dir [insert directory to output data here] --use_prior_data --use_backward_data --num_omitted_neg_frames 10 --num_labels 3 --num_prompt_aug 1 --perc_pos_samples 50
```

This command will generate fine-tuning data in the specified output directory.

### Running MiniGPT-4 Fine-tuning

To launch the second stage alignment, 
first specify the path to the checkpoint file trained in stage 1 in line 10 of
[train_configs/minigpt4_stage1_pretrain.yaml](train_configs/minigpt4_stage2_finetune.yaml).
You can also specify the output path in line 39.
Then, run the following command. In our experiments, we find that we need a GPU that has at least 24GB memory.

```bash
python train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```

After the second stage alignment, the fine-tuned VLM model will be able to act as a reward model.

### Running MiniGPT-4 Server for RL Fine-tuning

Please refer to the readme of the RoboFuME repo for instructions related to running RL fine-tuning.

## License
This repository is under [BSD 3-Clause License](LICENSE.md).
Many codes are based on [Lavis](https://github.com/salesforce/LAVIS) with 
BSD 3-Clause License [here](LICENSE_Lavis.md).
