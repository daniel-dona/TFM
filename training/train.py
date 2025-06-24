import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
os.environ["WANDB_API_KEY"] =  "#####"
os.environ["WANDB_PROJECT"] = "tfm"
os.environ["WANDB_LOG_MODEL"] = "false"

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json

import re
import nltk

import wandb

import gc
import torch

import time
import traceback

import tqdm

from unsloth import FastLanguageModel

import datasets

from trl import SFTTrainer, SFTConfig
from transformers.integrations import WandbCallback
from transformers import GenerationConfig

import pyparseit
import rdflib
import SPARQLWrapper

import random

import pandas as pd
import numpy as np

seed = 42
size = 128

models = [
    #"unsloth/Qwen3-0.6B-bnb-4bit",
    #"unsloth/Qwen3-1.7B-bnb-4bit",
    "unsloth/Qwen3-4B-bnb-4bit",
    #"unsloth/Qwen3-8B-bnb-4bit",
    #"unsloth/Qwen3-14B-bnb-4bit"
]

endpoint = "http://192.168.10.174:8890/sparql"

def is_sparql_syntax_valid_rdflib(query_string):

    try:
        g = rdflib.Graph()
        g.query(query_string)

        return True

    except Exception as e:
        #print(e)
        
        return False
    
def is_sparql_syntax_valid_virtuoso(endpoint, query):

    agent = SPARQLWrapper.SPARQLWrapper(endpoint=endpoint)

    try:
        
        agent.setQuery(query)
        agent.addExtraURITag("timeout","30000")
        agent.setTimeout(35)
        agent.setReturnFormat(SPARQLWrapper.JSON)
        result = agent.queryAndConvert()
        
        return True, len(result["results"]["bindings"])
    
    except TimeoutError:
        #traceback.print_exc()
        return True, -1
    
    except Exception:
        #print(e)
        return False, 0
    
def sparql_json_to_df(sparql_json, convert_types=True):

    if 'results' not in sparql_json or 'bindings' not in sparql_json['results']:
        return pd.DataFrame()

    cols = []

    for i, var in enumerate(sparql_json['head']['vars']):
        cols.append(f"row_{i}")

    bindings = sparql_json['results']['bindings']

    if not bindings:
        return pd.DataFrame(columns=cols)

    data_rows = [
        [row.get(col, {}).get('value') for col in sparql_json['head']['vars']]
        for row in bindings
    ]

    df = pd.DataFrame(data_rows, columns=cols)

    df.fillna(value=np.nan, inplace=True)
    
    if convert_types:
        df = df.convert_dtypes()

    return df

def compare_results(r1, r2):

    r1_df = sparql_json_to_df(r1)
    r2_df = sparql_json_to_df(r2)


    if r1_df.shape != r2_df.shape:

        #print("Diff shape")

        return False
    
    # Sort r1 by first col, sort r2 by each col, take first element, sort cols

    r1_sorted = r1_df.sort_values(by=r1_df.columns[0])

    for col in r2_df.columns:

        r2_sorted = r2_df.sort_values(by=col)

        is_eq = []

        for i in range(10):

            rand_index = random.randint(0, r1_sorted.shape[0]-1)

            is_eq.append(sorted(list(r1_sorted.iloc[rand_index])) == sorted(list(r2_sorted.iloc[rand_index])))

        if all(is_eq):

            return True
        
    return False
        

    #print(r1_sorted[0])
    #print(r2_sorted[0])

def compare_queries(q1, q2, endpoint, timeout=35):

    regex = r"LIMIT\s\d*"
    subst = "LIMIT 10"


    agent = SPARQLWrapper.SPARQLWrapper(endpoint=endpoint)
            
    try:
    
        agent.setTimeout(timeout)
        agent.addExtraURITag("timeout","30000")
        agent.setQuery(re.sub(regex, subst, q1, 0, re.MULTILINE))
        agent.setReturnFormat(SPARQLWrapper.JSON)
        results1 = agent.queryAndConvert()

        agent.setTimeout(timeout)
        agent.addExtraURITag("timeout","30000")
        agent.setQuery(re.sub(regex, subst, q2, 0, re.MULTILINE))
        agent.setReturnFormat(SPARQLWrapper.JSON)
        results2 = agent.queryAndConvert()

        try:
            
            return compare_results(results1, results2)

        except Exception as e:
            print(e)
            return False

    except Exception as e:
        print(e)
        return False
    

class CustomEvalCallbackSparql(WandbCallback):
    def __init__(self, trainer, test_dataset, num_samples=32, batch_size=32, max_new_tokens=256):

        super().__init__()
        
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.sample_dataset = test_dataset.filter(lambda row: row["task"] == "sparql").take(num_samples)
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        
        self.gen_config = GenerationConfig.from_pretrained(
            trainer.model.name_or_path,
            max_new_tokens=max_new_tokens,
            cache_implementation="offloaded"
        )
        
        
    def generate(self, prompt):
        
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
        
        with torch.inference_mode():
            output = self.model.generate(tokenized_prompt, generation_config=self.gen_config)
            
        return self.tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True), len(output[0][len(tokenized_prompt[0]):])

    def batch_generate(self, prompts):

        print("Generating", len(prompts))

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self.tokenizer.padding_side = "left"

        tokenized = self.tokenizer(prompts, return_tensors='pt', padding=True)

        input_ids = tokenized['input_ids'].cuda()
        attn_mask = tokenized['attention_mask'].cuda()
        
        with torch.inference_mode():
            output = self.model.generate(input_ids, attention_mask=attn_mask, generation_config=self.gen_config)
            
        output_cut = []
        output_len = []

        for i, output_part in enumerate(output):
            output_cut.append(output_part[len(input_ids[i]):])
            output_len.append(len(output_part[len(input_ids[i]):]))
            
        return self.tokenizer.batch_decode(output_cut, skip_special_tokens=True), output_len

    def sparql_samples_table(self, examples):

        records_table = wandb.Table(columns=["nlq", "prompt", "completion", "sparql_ref", "sparql_gen", "bleu", "valid_gen", "correct_gen"])

        valid = 0
        correct = 0
        bleu = []
        gen_time = 0
        test_time = 0

        for batch in tqdm.tqdm(range(0, self.num_samples, self.batch_size)):

            examples_batch = examples.select(range(batch, batch+self.batch_size))

            prompts = examples_batch["gen_prompt"]

            t0 = time.time()
            
            generations, _ = self.batch_generate(prompts)

            t1 = time.time()

            gen_time += t1-t0


            t0 = time.time()

            for example, generation, prompt in zip(examples_batch, generations, prompts):
                
                try:
                
                    blocks = pyparseit.parse_markdown_string(generation)

                    #print(generation)
                                        
                    if len(blocks) >= 1:
                        sparql_gen = blocks[-1].content
                        valid_rdflib = is_sparql_syntax_valid_rdflib(sparql_gen)
                        valid_virtuoso, _ = is_sparql_syntax_valid_virtuoso(endpoint, sparql_gen)
                        example_bleu = nltk.translate.bleu_score.sentence_bleu([example["sparql"]], sparql_gen)
                        bleu.append(example_bleu)
                        valid_example = False
                        correct_example = False

                        if valid_rdflib and valid_virtuoso:
                            valid += 1
                            valid_example = True

                            if compare_queries(example["sparql"], sparql_gen, endpoint):
                                correct += 1
                                correct_example = True

                        records_table.add_data(example["nlq"], prompt, generation.strip(), example["sparql"], sparql_gen, example_bleu, valid_example, correct_example)


                    else:
                        bleu.append(0)
                        records_table.add_data(example["nlq"], prompt, generation.strip(), example["sparql"], None, 0, False, False)


                except Exception as e:
                    traceback.print_exc()

            t1 = time.time()

            test_time += t1-t0

            mean_bleu = np.mean(bleu)


        return records_table, valid/self.num_samples, correct/self.num_samples, mean_bleu, gen_time, test_time
    
        
    def on_evaluate(self, args, state, control,  **kwargs):
                
        super().on_evaluate(args, state, control, **kwargs)
        
        records_table, valid, correct, bleu, gen_time, test_time = self.sparql_samples_table(self.sample_dataset)

        self._wandb.log(
            {
                "eval/predictions_sparql": records_table,
                "eval/predictions_sparql_valid": valid,
                "eval/predictions_sparql_bleu": bleu,
                "eval/predictions_sparql_correct": correct,
                "eval/predictions_sparql_time_gen": gen_time,
                "eval/predictions_sparql_time_test": test_time
            }
        )

class CustomEvalCallbackRejections(WandbCallback):
    def __init__(self, trainer, test_dataset, num_samples=32, batch_size=32, max_new_tokens=256, log_model="checkpoint"):

        super().__init__()
        
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.sample_dataset = test_dataset.filter(lambda row: row["task"] == "rejections").take(num_samples)
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        
        self.gen_config = GenerationConfig.from_pretrained(
            trainer.model.name_or_path,
            max_new_tokens=max_new_tokens,
            cache_implementation="offloaded"
        )
        
        
    def generate(self, prompt):
        
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
        
        with torch.inference_mode():
            output = self.model.generate(tokenized_prompt, generation_config=self.gen_config)
            
        return self.tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True), len(output[0][len(tokenized_prompt[0]):])

    def batch_generate(self, prompts):

        print("Generating", len(prompts))

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self.tokenizer.padding_side = "left"

        tokenized = self.tokenizer(prompts, return_tensors='pt', padding=True)

        input_ids = tokenized['input_ids'].cuda()
        attn_mask = tokenized['attention_mask'].cuda()
        
        with torch.inference_mode():
            output = self.model.generate(input_ids, attention_mask=attn_mask, generation_config=self.gen_config)
            
        output_cut = []
        output_len = []

        for i, output_part in enumerate(output):
            output_cut.append(output_part[len(input_ids[i]):])
            output_len.append(len(output_part[len(input_ids[i]):]))
            
        return self.tokenizer.batch_decode(output_cut, skip_special_tokens=True), output_len

    def sparql_samples_table(self, examples):

        records_table = wandb.Table(columns=["nlq", "prompt", "completion", "rejection_ref", "rejection_gen", "valid_gen", "correct_gen"])

        valid = 0
        correct = 0
        gen_time = 0
        test_time = 0

        for batch in tqdm.tqdm(range(0, self.num_samples, self.batch_size)):

            examples_batch = examples.select(range(batch, batch+self.batch_size))

            prompts = examples_batch["gen_prompt"]

            t0 = time.time()
            
            generations, _ = self.batch_generate(prompts)

            t1 = time.time()

            gen_time += t1-t0


            t0 = time.time()

            for example, generation, prompt in zip(examples_batch, generations, prompts):
                
                try:
                
                    blocks = pyparseit.parse_markdown_string(generation)

                    #print(generation)
                                        
                    if len(blocks) >= 1:

                        valid_example = False
                        correct_example = False

                        try:
                            json_gen = json.loads(blocks[-1].content)

                            if "valid" in json_gen:

                                valid += 1
                                valid_example = True

                                if json_gen["valid"] == example["result"]:

                                    correct += 1
                                    correct_example = True

                                records_table.add_data(example["nlq"], prompt, generation.strip(), example["result"], json_gen["valid"], valid_example, correct_example)

                            else:
                                records_table.add_data(example["nlq"], prompt, generation.strip(), example["result"], None, valid_example, correct_example)


                        except:
                            records_table.add_data(example["nlq"], prompt, generation.strip(), example["result"], None, False, False)

                        
                    else:
                        
                        records_table.add_data(example["nlq"], prompt, generation.strip(), example["result"], None, False, False)


                except Exception as e:
                    traceback.print_exc()

            t1 = time.time()

            test_time += t1-t0



        return records_table, valid/self.num_samples, correct/self.num_samples, gen_time, test_time
    
        
    def on_evaluate(self, args, state, control,  **kwargs):
                
        super().on_evaluate(args, state, control, **kwargs)
        
        records_table, valid, correct, gen_time, test_time = self.sparql_samples_table(self.sample_dataset)

        self._wandb.log(
            {
                "eval/predictions_rejections": records_table,
                "eval/predictions_rejections_valid": valid,
                "eval/predictions_rejections_correct": correct,
                "eval/predictions_rejections_time_gen": gen_time,
                "eval/predictions_rejections_time_test": test_time
            }
        )


def train(model, lr):
    
    run_name = f"{model}_lora_rank_{size}_ts_{int(time.time())}"
    
    wandb.init(name=run_name, reinit="finish_previous")
    
    print(f"Training LoRA size: {size}")

    ## LOAD MODEL

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model,
        max_seq_length = 4096,
        load_in_4bit = True
    )

    ## DATASETS CONFIG
    
    ### Task: rejections
    
    dataset = datasets.load_dataset("daniel-dona/sparql-dataset-era-rejections")["train"]
    
    dataset = dataset.train_test_split(test_size=0.05, seed=seed, shuffle=True)

    train_dataset_rejections = dataset["train"]
    test_dataset_rejections = dataset["test"] 
    
    ### Task: tools
    
    #TODO

    ### Task: sparql

    #dataset_syn1 = datasets.load_dataset("daniel-dona/sparql-dataset-reasoning-test4")["train"]
    #dataset_syn2 = datasets.load_dataset("daniel-dona/sparql-dataset-era-64k")["train"]
    dataset_syn3 = datasets.load_dataset("daniel-dona/sparql-dataset-era-nocot-64k")["train"] #Single class
    dataset_syn4 = datasets.load_dataset("daniel-dona/sparql-dataset-era-nocot2-64k")["train"] #Multiclass
    dataset_ds = datasets.load_dataset("daniel-dona/sparql-dataset-era-ds-2")["train"]
    dataset_cq = datasets.load_dataset("daniel-dona/sparql-dataset-era-cq-2")["train"]

    dataset = datasets.concatenate_datasets(
        [
            #dataset_cq, 
            #dataset_ds, 
            dataset_syn3.shuffle(seed=seed), 
            dataset_syn4.shuffle(seed=seed)
        ])


    dataset.shuffle(seed=seed)
        
    dataset = dataset.train_test_split(test_size=0.05, seed=seed, shuffle=True)
    
    train_dataset_sparql = dataset["train"] #.filter(lambda x: x["lang"] == "spa")
    test_dataset_sparql = dataset["test"] #.filter(lambda x: x["lang"] == "spa")

    #prompt_template_think = open("templates/prompt_template_think.txt").read()
    prompt_template = open("templates/prompt_template_nothink.txt").read()

    #response_template_think = open("templates/response_template_think.txt").read()
    response_template = open("templates/response_template_nothink.txt").read()

    def chat_template_sparql(row):

        row["task"] = "sparql"
        
        message_system = {
            "role": "system",
            "content": "You are GenSPARQL, a finetuned model by Daniel Doña to generate SPARQL queries based on user questions."
        }
        
        message_user = {
            "role": "user",
            "content": prompt_template.replace("%nlq", row["nlq"])
        }
        
        message_model = {
            "role": "assistant",
            "content": response_template.replace("%sparql", row["sparql"]) #.replace("%reasoning", row["cot"])
        }
        
        row["gen_prompt"] = tokenizer.apply_chat_template(
            [message_system, message_user],
            tokenize=False,
            enable_think=False,
            add_generation_prompt=True
        )
        
        row["text"] = tokenizer.apply_chat_template( #defaul col
            [message_system, message_user, message_model],
            enable_think=False,
            tokenize=False,
        )
        
        return row

    train_dataset_sparql = train_dataset_sparql.map(chat_template_sparql).shuffle(seed=seed)
    test_dataset_sparql = test_dataset_sparql.map(chat_template_sparql).shuffle(seed=seed)
    
    
    #prompt_template_think = open("templates/prompt_template_think.txt").read()
    prompt_template = open("templates/prompt_template_valid.txt").read()

    #response_template_think = open("templates/response_template_think.txt").read()
    response_template = open("templates/response_template_valid.txt").read()

    def chat_template_rejections(row):

        row["task"] = "rejections"
        
        message_system = {
            "role": "system",
            "content": "You are GenSPARQL, a finetuned model by Daniel Doña to generate SPARQL queries based on user questions."
        }
        
        message_user = {
            "role": "user",
            "content": prompt_template.replace("%nlq", row["instruction"])
        }
        
        message_model = {
            "role": "assistant",
            "content": response_template.replace("%json", json.dumps({"valid": row["result"]})) #.replace("%reasoning", row["cot"])
        }
        
        row["gen_prompt"] = tokenizer.apply_chat_template(
            [message_system, message_user],
            tokenize=False,
            enable_think=False,
            add_generation_prompt=True
        )
        
        row["text"] = tokenizer.apply_chat_template( #defaul col
            [message_system, message_user, message_model],
            enable_think=False,
            tokenize=False,
        )
        
        return row

    train_dataset_rejections = train_dataset_rejections.map(chat_template_rejections).shuffle(seed=seed)
    test_dataset_rejections = test_dataset_rejections.map(chat_template_rejections).shuffle(seed=seed)
    

    train_dataset = datasets.concatenate_datasets([
        train_dataset_sparql,
        train_dataset_rejections
    ])
    
    train_dataset.shuffle(seed=seed)
    
    test_dataset = datasets.concatenate_datasets([
        test_dataset_sparql,
        test_dataset_rejections
    ])
    
    test_dataset.shuffle(seed=seed)

    ## TRAIN CONFIG
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = size,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = size,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = seed,
        #use_dora = True
    )
    
    total_bs = 1024
    device_bs = 8

    trainer = SFTTrainer(
        run_name=run_name,
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset, #.take(32*5), #.take(16*1024),
        eval_dataset = test_dataset.take(1024),
        args = SFTConfig(
            output_dir=f"runs/{run_name}",
            dataset_text_field = "text",
            per_device_train_batch_size = int(device_bs), #OK for 24GiB VRAM
            gradient_accumulation_steps = int(total_bs/device_bs),
            warmup_steps = 5,
            num_train_epochs = 1,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            #max_steps = 200,
            learning_rate = lr,
            logging_steps = 5,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = seed,
            report_to = "wandb",
            eval_on_start=True,
            #resume_from_checkpoint=""
        ),
    )

   

    custom_callback = CustomEvalCallbackSparql(trainer, test_dataset, num_samples=1024, batch_size=32, max_new_tokens=256)
    trainer.add_callback(custom_callback)

    custom_callback = CustomEvalCallbackRejections(trainer, test_dataset, num_samples=1024, batch_size=128, max_new_tokens=16)
    trainer.add_callback(custom_callback)

    print("Train started!")

    trainer.train()

    print("Train finished!")

    model.save_pretrained_merged(f"runs/{run_name}/merged", tokenizer, save_method = "merged_16bit",)
    model.save_pretrained_merged(f"runs/{run_name}/merged-bnb-4bit", tokenizer, save_method = "merged_4bit_forced",)

    model.save_pretrained(f"runs/{run_name}")
    tokenizer.save_pretrained(f"runs/{run_name}")

    #model_repo = f"daniel-dona/sparql-model-era-lora-128"

    #model.push_to_hub_merged(model_repo, tokenizer, save_method = "merged_16bit")
    #model.push_to_hub_gguf(model_repo, tokenizer, quantization_method = "f16")

    #os.system(f"python3 /home/dani/llama.cpp/convert_lora_to_gguf.py --outtype q8_0 'runs/{run_name}' --outfile 'runs/{run_name}/lora.gguf'")

    del trainer
    del model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(30)


train(model="unsloth/Qwen3-4B-bnb-4bit", lr=1e-4)
