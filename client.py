import os
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI, AzureOpenAI
from peft import PeftModel, prepare_model_for_kbit_training

class LlamaClient():
    def __init__(self, args, base_model="", resume_from_checkpoint=None, device="cpu", logger=None):
        # GPT Client
        self.args = args
        self.device_map = device
        self.logger = logger
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map=self.device_map,
        )

        logger.info(f"Load model '{base_model}'")

        if args.do_judge:
            self.judge = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map=self.device_map,
            )
            self.judge = PeftModel.from_pretrained(self.judge, args.judge_model)

            logger.info(f"Load Judge Model from '{args.judge_model}'")

        if resume_from_checkpoint is not None:
            self.model = PeftModel.from_pretrained(self.model, resume_from_checkpoint)
            logger.info(f"Load Defense Model from '{args.defense_model}'")


        self.model.eval()
            
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        self.label = base_model


    # ...existing code...
    def employ(self, SystemPrompts, UserPrompts, max_length=512, temperature=0.8, top_k=10, top_p=0.9, num_sequences=3, name="default"):
        batch_size = len(UserPrompts)
        messages_batch = [
            [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p}
            ]
            for sys_p, usr_p in zip(SystemPrompts, UserPrompts)
        ]
        prompts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            for messages in messages_batch
        ]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.device_map)
        input_ids = inputs["input_ids"]
        input_length = input_ids.shape[-1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=num_sequences
        )

        # outputs shape: [batch_size * num_return_sequences, seq_len]
        responses = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(num_sequences):
                idx = i * num_sequences + j
                generated_ids = outputs[idx][input_length:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                responses[i].append(generated_text)

        # 选取每个样本分数最高的 response
        best_responses = []
        for i in range(batch_size):
            prompt_list = [prompts[i]] * num_sequences
            scores = self.get_reward_score_batch(prompt_list, responses[i])
            best_idx = scores.index(max(scores))
            best_responses.append(responses[i][best_idx])

        return best_responses
    # ...existing code...

    def get_reward_score(self, prompt, response):
        # 拼接 prompt + response
        full_text = prompt + response
        inputs = self.tokenizer(full_text, return_tensors="pt", padding=True, truncation=True,).to(self.device_map)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,).input_ids.to(self.device_map)

        with torch.no_grad():
            outputs = self.judge(**inputs, labels=inputs.input_ids)
            # 取 loss 相关的 per-token logit
            log_probs = -outputs.loss * inputs.input_ids.size(1)  # 总 log prob

        # 只取 response 部分的平均 log-prob
        # 方法：先重新计算 response 部分的 token log prob
        with torch.no_grad():
            outputs = self.judge(**inputs)
            logits = outputs.logits[:, :-1, :]  # 移除最后一个token对应的logits
            labels = inputs.input_ids[:, 1:]    # 对齐
            log_probs_per_token = torch.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs_per_token.gather(2, labels.unsqueeze(2)).squeeze(2)

            # response token 部分
            resp_start = prompt_ids.shape[1] - 1  # response第一个token位置（减1是因为label对齐）
            resp_log_probs = selected_log_probs[:, resp_start:]
            avg_resp_log_prob = resp_log_probs.mean(dim=1).item()

        return avg_resp_log_prob

    def get_reward_score_batch(self, prompts, responses):
        """
        Batch compute reward scores.
        :param prompts: list[str], each sample's prompt
        :param responses: list[str], each sample's response
        :return: list[float], the average log-probability score for each sample
        """
        assert len(prompts) == len(responses), "the length of prompts and responses must be same"

        # Concatenate the full_text of the batch
        full_texts = [p + r for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(self.device_map)

        # Calculate the length (number of tokens) of each prompt to help locate the response part.
        prompt_ids = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device_map)
        prompt_lengths = (prompt_ids != self.tokenizer.pad_token_id).sum(dim=1)  # shape: [batch_size]

        with torch.no_grad():
            outputs = self.judge(**inputs)
            logits = outputs.logits[:, :-1, :]  # [B, L-1, V]
            labels = inputs.input_ids[:, 1:]    # [B, L-1]
            log_probs_per_token = torch.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs_per_token.gather(2, labels.unsqueeze(2)).squeeze(2)  # [B, L-1]

            # extract the log-prob of each sample's response part, and take the average
            scores = []
            for i, p_len in enumerate(prompt_lengths):
                resp_log_probs = selected_log_probs[i, p_len-1:]
                avg_resp_log_prob = resp_log_probs.mean().item()
                scores.append(avg_resp_log_prob)

        return scores


class OpenAIModel():
    # gpt-4o-mini-2024-07-18, gpt-4o-2024-08-06
    def __init__(self, model="gpt-4o-mini-2024-07-18", api_key=None):
        # GPT Client
        if api_key is not None:
            self.client = OpenAI(api_key=api_key)
        else:
            try:
                self.client = OpenAI(
                    api_key = os.environ.get("OPENAI_API_KEY")
                    )
            except Exception as e:
                raise ValueError("No OpenAI API key provided. Please set the OPENAI_API_KEY environment variable or pass it as an argument.")


        print(f"Using Evaluation model {model}.")
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0
        self.label = "GPT"

        self.tokens_dict = {"total_completion_tokens": 0, "total_prompt_tokens": 0, "total_total_tokens": 0}

    def employ(self, SystemPrompt=None, UserPrompt=None, response_format=None, name="default"):
        """
        Employ the LLM to response the prompt.
        """
        messages = []
        if SystemPrompt:
            messages.append({"role": "system", "content": SystemPrompt})
        if UserPrompt:  
            messages.append({"role": "user", "content": UserPrompt})

        try:
            if response_format != None:
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    response_format=response_format,
                    messages=messages,
                    max_tokens=2000
                )
                response_content = response.choices[0].message.parsed

            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=2000
                ) 
                response_content = response.choices[0].message.content

            print(f"completion_tokens: {response.usage.completion_tokens}. prompt_tokens: {response.usage.prompt_tokens}. sum_tokens: {response.usage.total_tokens}.\n")

            self.completion_tokens += response.usage.completion_tokens
            self.prompt_tokens += response.usage.prompt_tokens
            self.total_tokens += response.usage.total_tokens

            self.tokens_dict["total_completion_tokens"] = self.completion_tokens
            self.tokens_dict["total_prompt_tokens"] = self.prompt_tokens        
            self.tokens_dict["total_total_tokens"] = self.total_tokens

            print(f"total_completion_tokens: {self.completion_tokens}. total_prompt_tokens: {self.prompt_tokens}. total_sum_tokens: {self.total_tokens}.\n")

            if name not in self.tokens_dict:
                self.tokens_dict[name] = {"completion_tokens": response.usage.completion_tokens, "prompt_tokens": response.usage.prompt_tokens, "total_tokens": response.usage.total_tokens}

            else:
                self.tokens_dict[name]["completion_tokens"] += response.usage.completion_tokens
                self.tokens_dict[name]["prompt_tokens"] += response.usage.prompt_tokens
                self.tokens_dict[name]["total_tokens"] += response.usage.total_tokens 

        except:
            response_content = "FAILED GENERATION."      

        return response_content

