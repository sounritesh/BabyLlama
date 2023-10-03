from transformers import AutoTokenizer, LlamaForCausalLM, GPT2LMHeadModel, PreTrainedTokenizerFast
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--tokenizer_path", type=str, default=None)
parser.add_argument("--model_type", type=str, default="gpt2")
parser.add_argument("--prompt", type=str,
                    default="Hey, are you conscious? Can you talk to me?")

args = parser.parse_args()

if args.model_type == "gpt2":
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
else:
    model = LlamaForCausalLM.from_pretrained(args.model_path)
tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(args.tokenizer_path))

prompt = args.prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=20)
decoded = tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0]


print(decoded)
