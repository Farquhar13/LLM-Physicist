"""
Fine-tune Llama-3.2-1B-Instruct on physics Q&A (veggiebird/physics-scienceqa)
Demonstrates pre- and post-fine-tuning outputs using LoRA (CPU-friendly).
"""

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# -------------------------------
# 1. Load base model + tokenizer
# -------------------------------

model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Fix: Llama models lack pad token
tokenizer.pad_token_id = tokenizer.eos_token_id


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu"  # CPU-only; works for small LoRA demo
)

# -------------------------------
# 2. Load and format dataset
# -------------------------------

# Load physics-scienceqa dataset
raw_ds = load_dataset("veggiebird/physics-scienceqa", split="train")

# Format into Q/A style
def format_batch(batch):
    return {
        "text": [
            f"Question: {q}\nAnswer: {a}"
            for q, a in zip(batch["input"], batch["output"])
        ]
    }

formatted_ds = raw_ds.map(format_batch, batched=True)

# Tokenize and add labels
def tokenize(batch):
    enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)
    enc["labels"] = enc["input_ids"].copy()
    return enc

tokenized_ds = formatted_ds.map(tokenize, batched=True)

# -------------------------------
# 3. Pre-fine-tune evaluation
# -------------------------------

test_questions = [
    "What is the second law of thermodynamics?",
    "Explain Newton's third law of motion.",
    "What happens to time near the speed of light?"
]

print("=== Base model (no fine-tuning) outputs ===")
for q in test_questions:
    inputs = tokenizer(q, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)
    print(f"\nQ: {q}\nA: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

# -------------------------------
# 4. Apply LoRA adapter
# -------------------------------

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# -------------------------------
# 5. Fine-tuning
# -------------------------------

training_args = TrainingArguments(
    output_dir="./llama1b-phys-scienceqa",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=200,       # Increase for better adaptation
    learning_rate=2e-4,
    logging_steps=20,
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds
)

print("\nStarting fine-tuning...")
trainer.train()

# -------------------------------
# 6. Post-fine-tune evaluation
# -------------------------------

print("\n=== Fine-tuned model outputs ===")
for q in test_questions:
    inputs = tokenizer(q, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)
    print(f"\nQ: {q}\nA: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

# -------------------------------
# 7. (Optional) Save adapter
# -------------------------------
# model.save_pretrained("./llama1b-phys-scienceqa-adapter")


"""
1. new_max_tokens=80 

=== Base model (no fine-tuning) outputs ===
Q: What is the second law of thermodynamics?
A: What is the second law of thermodynamics? The second law of thermodynamics is a fundamental principle that describes the direction of spontaneous processes and the direction of heat transfer. The second law states that the total entropy of an isolated system will always increase over time. This means that as energy is transferred from a hotter object to a cooler object, the entropy of the cooler object will increase.

The second law of thermodynamics is often stated as:

"Entropy

Q: Explain Newton's third law of motion.
A: Explain Newton's third law of motion. Newton's third law states that every action has an equal and opposite reaction. This means that when one object exerts a force on another object, the second object will exert an equal force on the first object. The reaction force is equal in magnitude and opposite in direction to the applied force.

## Step 1: Define Newton's third law of motion
Newton's third law of motion states that every

Q: What happens to time near the speed of light?
A: What happens to time near the speed of light? Time dilation is a fundamental concept in physics that describes how time appears to pass differently for observers in different states of motion. According to the theory of special relativity, time dilation occurs when an object approaches the speed of light.
As an object approaches the speed of light, time appears to pass more slowly for that object relative to an observer. This effect is most pronounced when the object is moving at high
"""

"""
=== Fine-tuned model outputs ===

Q: What is the second law of thermodynamics?
A: What is the second law of thermodynamics? A thermodynamic system is in equilibrium if its temperature is the same throughout. The second law of thermodynamics states that it is impossible to build a machine that will keep a system at a lower temperature than the temperature of the surrounding environment.
Consider a thermodynamic system in equilibrium. The system is in equilibrium if its temperature is the same throughout. The second law of thermodynamics states that it is impossible to

Q: Explain Newton's third law of motion.
A: Explain Newton's third law of motion. A 10-kilogram block is pushed against a wall. The block exerts a force on the wall. Describe the motion of the wall.
Choose from: The wall exerts a force on the block., The block exerts a force on the wall..
Answer: The wall exerts a force on the block. According to Newton's third law, the block exerts a force on the

Q: What happens to time near the speed of light?
A: What happens to time near the speed of light? According to Einstein's theory of special relativity, time appears to pass at the same rate for all observers in uniform motion relative to one another. However, the motion is so fast that the distance between the observers does not change. As a result, time dilation occurs.
The observer who is moving at a speed close to the speed of light will experience time passing at a slower rate compared to the other
"""

"""
2. new_max_tokens=200

=== Base model (no fine-tuning) outputs ===

Q: What is the second law of thermodynamics?
A: What is the second law of thermodynamics? The second law of thermodynamics states that the total entropy of an isolated system will always increase over time. In other words, entropy always increases as energy is transferred from one place to another, and as the universe expands.
The second law of thermodynamics is often expressed in the following way: "It is impossible to build a machine that can convert all the heat energy put into it into useful work." This is known as the "second law of thermodynamics" or the "zeroth law of thermodynamics" (although it is not really a law at all, but a statement about the direction of spontaneous processes).
The second law of thermodynamics is a fundamental principle of physics and has far-reaching implications for our understanding of the universe. It is a fundamental concept that is widely accepted by scientists and engineers alike.

Q: Explain Newton's third law of motion.
A: Explain Newton's third law of motion. The law states that every action has an equal and opposite reaction. This law applies to all interactions between objects.

## Step 1: Define Newton's third law of motion
Newton's third law of motion states that for every action, there is an equal and opposite reaction.

## Step 2: Explain the concept of action and reaction
In physics, an action is any force that is applied to an object, such as a push or a pull. A reaction is the force that is applied to the object as a result of the action, and it is equal in magnitude and opposite in direction to the action.

## Step 3: Provide an example to illustrate the law
For example, when you throw a ball, you exert a force on the ball, causing it to move away from you. At the same time, the ball exerts an equal and opposite force on you, causing you to move towards the ball.

## Step 4: Summarize the key points of Newton's

Q: What happens to time near the speed of light?
A: What happens to time near the speed of light? Time dilation, a phenomenon predicted by Albert Einstein's theory of special relativity, suggests that time appears to slow down for an observer in motion relative to a stationary observer. This effect is most pronounced at high speeds and becomes significant at speeds approaching the speed of light.

### Speed of Light as a Limit

The speed of light (approximately 299,792 kilometers per second) is the maximum speed at which any object or information can travel in a vacuum. According to Einstein's theory of special relativity, nothing can reach or exceed the speed of light. Therefore, time dilation is a fundamental aspect of the theory, predicting that time will appear to pass more slowly for an observer in motion relative to a stationary observer.

### Effects of Time Dilation

The effects of time dilation are observed in various ways, including:

1. **Clocks and Time Scales:** Time dilation causes time to appear to pass more slowly for an observer in motion relative to a stationary observer. This means that time appears to
"""

"""
=== Fine-tuned model outputs ===

Q: What is the second law of thermodynamics?
A: What is the second law of thermodynamics? A thermodynamic system is in equilibrium with a reservoir at a temperature lower than the temperature of the system. The second law of thermodynamics explains why it is impossible to build a machine that can convert all the heat energy of a hot gas into useful work. The second law also explains why it is impossible to build a machine that can convert all the heat energy of a cold gas into useful work.

Q: Explain Newton's third law of motion.
A: Explain Newton's third law of motion. A 10-kilogram block is pushed with a force of 100 Newtons. What is the mass of the block? Choose from: The mass of the block, the force of the push, the speed of the block.
Look for answers in in: Newton's third law of motion
Consider: a push on a block of water
A push on a block of water results in a force of 100 Newtons. Look at the table below to see how the mass of the block is related to the force.
| Force (N) | Mass (kg) |
| --- | --- |
| 100 | 10 |
Notice that the force is 10 Newtons. Look at the table to see if the mass is related to the force. The mass of the block is 10 kilograms. Look at the table to see how the mass is related to the force.
| Force (N) | Mass (kg) |
| --- | --- |
| 100 | 

Q: What happens to time near the speed of light?
A: What happens to time near the speed of light? According to Einstein's theory of relativity, time near the speed of light is affected.
A: The time passes more slowly. Choose from: the time passes more slowly, the time passes as usual.
B: The time passes more slowly. Choose from: the time passes as usual, the time passes more slowly.
Answer: Choose the statement that describes what happens to time near the speed of light.
"""