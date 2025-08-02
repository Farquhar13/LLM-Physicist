"""
    Fine-tune LLama-3.2-1B-Instruct with LoRA on a small set of questions
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# --- Load Llama 3.2 1B Instruct model (full precision) ---
model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu"
)

# --- Add LoRA adapter (lightweight fine-tune) ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# --- Tiny physics dataset ---
train_data = [
    {"instruction": "Explain black hole entropy",
     "output": "Black hole entropy is proportional to the event horizon area via S = kA/(4ℓ_p²)."},
    {"instruction": "State Schrödinger’s equation",
     "output": "iħ ∂ψ/∂t = Ĥψ describes quantum system evolution."},
    {"instruction": "What is Noether’s theorem?",
     "output": "Noether's theorem links continuous symmetries to conserved quantities like momentum or energy."}
]

def format_examples(batch):
    return {
        "text": [
            f"Question: {instr}\nAnswer: {out}"
            for instr, out in zip(batch["instruction"], batch["output"])
        ]
    }

dataset = Dataset.from_list(train_data).map(format_examples, batched=True)

def tokenize(batch):
    encodings = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

tokenized_dataset = dataset.map(tokenize, batched=True)

# --- Training arguments ---
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=50, 
    learning_rate=2e-4,
    logging_steps=10,
    output_dir="./llama1b-phys-lora",
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# --- Train ---
trainer.train()

# --- Test ---
prompt = "Explain black hole entropy in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

"""
Example output: 

## Step 1: Define entropy
Entropy is a measure of disorder or randomness in a system. In thermodynamics, entropy is used to quantify the amount of thermal energy unavailable to do work in a system.

## Step 2: Introduce black hole entropy
Black hole entropy is a concept introduced by Stephen Hawking in the 1970s. It is a measure of the disorder or randomness in a black hole, which is a region of spacetime with such intense gravitational pull that not
"""