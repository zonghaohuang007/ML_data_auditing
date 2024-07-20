
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import AdamW, get_scheduler

from peft import LoraConfig

from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

from string import punctuation
import random
import math
import argparse

import warnings
warnings.filterwarnings('ignore')

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


def formatting_func(example):
    text = f"### Input: {example['question']} ### Output: {example['best_answer']}"
    return text

def tokenize_function(examples):
    return tokenizer(examples['prompt'], padding='max_length', max_length=256, truncation=True)


paraphrase_tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").cuda()


def paraphrase(
    question,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=2,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    input_ids = paraphrase_tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids

    input_ids = input_ids.cuda()
    
    outputs = paraphrase_model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = paraphrase_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    outputs = []
    for i in res:
        output = ''
        i = i.lower()
        for j in i:
            if j not in punctuation and (j.isalpha() or j == ' '):
                output += j

        outputs.append(output)

    return outputs


from cswor import BBHG_confseq

def tight_chernoff_bound(tau, N):
    return (math.exp(tau*2/N-1) / ((tau*2/N)**(tau*2/N)))**(N/2)

def find_tau(p, N):
    tau_a = N // 2
    tau_b = N

    while tau_b - tau_a > 1:
        if tight_chernoff_bound((tau_a+tau_b)//2, N) > p:
            tau_a = (tau_a+tau_b)//2
        elif tight_chernoff_bound((tau_a+tau_b)//2, N) < p:
            tau_b = (tau_a+tau_b)//2
        else:
            tau_b = (tau_a+tau_b)//2
            break
    assert tight_chernoff_bound(tau_b, N) <= p
    return tau_b
    
def detection(model, published_data, unpublished_data, args):

    published_dataloader = DataLoader(published_data, batch_size=1)
    unpublished_dataloader = DataLoader(unpublished_data, batch_size=1)

    model.eval()
    published_logits = []
    for batch in published_dataloader:
        batch = batch['input_ids'].to(device)
        with torch.no_grad():

            outputs = model(batch, labels=batch)

        published_logits.append(outputs.loss)

    unpublished_logits = []
    for batch in unpublished_dataloader:
        batch = batch['input_ids'].to(device)
        with torch.no_grad():

            outputs = model(batch, labels=batch)
        unpublished_logits.append(outputs.loss)

    assert len(published_logits) == len(unpublished_logits)

    acc_full_me = 0
    detected_full_me = False
    alpha1 = args.p / 2
    alpha2 = args.p / 2
    tau =  find_tau(alpha2, len(published_logits))
    sequences = []
    cost_full_me = len(published_logits)

    for j in range(len(published_logits)):

        score1_full_me = -published_logits[j]
        score2_full_me = -unpublished_logits[j]

        if not detected_full_me:
            if score1_full_me > score2_full_me:
                acc_full_me += 1
                success = 1
            elif score1_full_me == score2_full_me:  # if equal, toss a coin
                if random.sample([True, False], k=1)[0]:
                    acc_full_me += 1
                    success = 1
                else:
                    success = 0
            else:
                success = 0
            sequences.append(success)
            y1, y2 = BBHG_confseq(sequences, len(published_logits), BB_alpha=1, BB_beta=1, alpha=alpha1)
            assert len(y1) == len(sequences)
            if y1[-1] >= tau:
                cost_full_me = len(sequences)
                detected_full_me = True

        if detected_full_me:
            break

    print('==>full me | cost: {} | membership acc: {}'.format(cost_full_me, acc_full_me / cost_full_me))
    if detected_full_me:
        detected_full_me = 1
    else:
        detected_full_me = 0
    
    return cost_full_me, detected_full_me


def evaluate_accuracy(model, tokenizer, test_dataset):
    model.eval()
    acc = 0
    for data in test_dataset:
        prompt = data['prompt']
        with torch.no_grad():
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            outputs = model.generate(input_ids)[0]
            response = tokenizer.decode(outputs)[len(prompt)+4:]
            if data['label'] in response:
                acc += 1
    print('acc: {}'.format(acc / len(test_dataset)))
    return acc/len(test_dataset)
  

def get_parser():

    parser = argparse.ArgumentParser(description='lol')

    ###########################################################################
    # Central:
    parser.add_argument("--p", type=float, default=0.05, help='p: upper bound on false-detection rate')
    parser.add_argument("--num_experiments", type=int, default=20, help='number of experiments to run')

    return parser


if __name__ == '__main__':

    args = get_parser().parse_args()

    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
    

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    loaded_data = load_dataset("sst2")

    results = {}
    for i in range(4):
        results[str(i)] = {'cost':0, 'detected':0, 'Q/M': 0, 'acc': 0}

    for exp_index in range(args.num_experiments):
        
        print('=================================================================================')
        print('Running {}-th experiment'.format(exp_index))
        train_dataset = loaded_data["train"]
        test_dataset = loaded_data["validation"]

        sample_idx = random.sample(list(range(len(train_dataset))), 10000)
        train_dataset = train_dataset.select(sample_idx)

        sample_idx = random.sample(list(range(len(train_dataset))), int(len(train_dataset)*0.1))  # 10% is assumed to be contributed from a data owner

        published_data = {'prompt':[]}
        unpublished_data = {'prompt':[]}
        train_data = {'prompt':[]}

        classes = ['Negative', 'Positive']

        for i in range(len(train_dataset)):
            if i in sample_idx:
                [twins1, twins2] = paraphrase(train_dataset[i]['sentence'])
                if random.sample([True, False], 1)[0]:
                    published_data['prompt'].append(f"### Given text: '{twins1}'\n### Question: Please classify the above given text into one of these two classes: [Negative, Positive].\n### Answer: {classes[train_dataset[i]['label']]}")
                    unpublished_data['prompt'].append(f"### Given text: '{twins2}'\n### Question: Please classify the above given text into one of these two classes: [Negative, Positive].\n### Answer: {classes[train_dataset[i]['label']]}")
                    train_data['prompt'].append(f"### Given text: '{twins1}'\n### Question: Please classify the above given text into one of these two classes: [Negative, Positive].\n### Answer: {classes[train_dataset[i]['label']]}")
                else:
                    published_data['prompt'].append(f"### Given text: '{twins2}'\n### Question: Please classify the above given text into one of these two classes: [Negative, Positive].\n### Answer: {classes[train_dataset[i]['label']]}")
                    unpublished_data['prompt'].append(f"### Given text: '{twins1}'\n### Question: Please classify the above given text into one of these two classes: [Negative, Positive].\n### Answer: {classes[train_dataset[i]['label']]}")
                    train_data['prompt'].append(f"### Given text: '{twins2}'\n### Question: Please classify the above given text into one of these two classes: [Negative, Positive].\n### Answer: {classes[train_dataset[i]['label']]}")
            else:
                train_data['prompt'].append(f"### Given text: '{train_dataset[i]['sentence']}'\n### Question: Please classify the above given text into one of these two classes: [Negative, Positive].\n### Answer: {classes[train_dataset[i]['label']]}")
        print('finished twins data generation.')

        test_data = {'prompt':[], 'label': []}
        for i in range(len(test_dataset)):
            test_data['prompt'].append(f"### Given text: {test_dataset[i]['sentence']}\n### Question: Please classify the above given text into one of these two classes: [Negative, Positive].\n### Answer:")
            test_data['label'].append(classes[test_dataset[i]['label']])

        train_data = Dataset.from_dict(train_data)
        test_data = Dataset.from_dict(test_data)

        published_data = Dataset.from_dict(published_data)
        unpublished_data = Dataset.from_dict(unpublished_data)

        train_data = train_data.map(tokenize_function, batched=True)
        train_data.set_format("torch")
            
        published_data = published_data.map(tokenize_function, batched=True)
        published_data.set_format("torch")

        unpublished_data = unpublished_data.map(tokenize_function, batched=True)
        unpublished_data.set_format("torch")

        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=2)

        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", quantization_config=quantization_config)

        EPOCHS = 3
        LEARNING_RATE = 2e-4
        num_training_steps = EPOCHS * len(train_dataloader)

        test_acc = evaluate_accuracy(model, tokenizer, test_data)
        results['0']['test_acc'] += test_acc / args.num_experiments
        cost, detected = detection(model, published_data, unpublished_data, args)
        results['0']['cost'] += cost / args.num_experiments
        results['0']['Q/M'] += cost / 10000 / args.num_experiments
        results['0']['detected'] += detected / args.num_experiments

        lora_config = LoraConfig(
                    r=8,
                    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                    bias="none",
                    task_type="CAUSAL_LM",
                )

        model.add_adapter(lora_config)
        lora_layers = filter(lambda p: p.requires_grad, model.parameters())

        optimizer = AdamW(lora_layers, lr=LEARNING_RATE, weight_decay=0.001)
        scheduler = get_scheduler(name="constant", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        for epoch in range(EPOCHS):
                    
            model.train()
            print(f"EPOCH {epoch} started" + '=' * 30)

            sum_loss = 0.0
            
            for batch in train_dataloader:
                batch = batch['input_ids'].to(device)

                outputs = model(batch, labels=batch)
                loss = outputs.loss                      
                loss.backward()
                sum_loss = sum_loss + loss.detach().data
                                    
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                    
            print('epoch: {} | train loss: {}'.format(epoch, sum_loss / len(train_dataloader)))

            # detection
            test_acc = evaluate_accuracy(model, tokenizer, test_data)
            results[str(epoch+1)]['test_acc'] += test_acc / args.num_experiments
            cost, detected = detection(model, published_data, unpublished_data, args)
            results[str(epoch+1)]['cost'] += cost / args.num_experiments
            results[str(epoch+1)]['Q/M'] += cost / 10000 / args.num_experiments
            results[str(epoch+1)]['detected'] += detected / args.num_experiments

    print('print out results averaged over {} experiments...'.format(args.num_experiments))
    print(results)

