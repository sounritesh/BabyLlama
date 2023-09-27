from transformers import AutoTokenizer, LlamaForCausalLM
import ArgumentParser from argparse

parser = ArgumentParser()

parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--tokenizer_path", type=str, default=None)
parser.add_argument("--prompt", type=str,
                    default="Hey, are you conscious? Can you talk to me?")
args = parser.parse_args()

model = LlamaForCausalLM.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

prompt = args.prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
decoded = tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0]


print(decoded)
