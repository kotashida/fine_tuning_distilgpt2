from transformers import AutoTokenizer, AutoModelForCausalLM

fine_tuned_model = AutoModelForCausalLM.from_pretrained('./fine-tuned-model')
tokenizer = AutoTokenizer.from_pretrained('./fine-tuned-model')

# Generate answers to questions
def generate_answer(question):
    prompt = f"[INST] {question} [/INST]"
    inputs = tokenizer.encode(prompt, return_tensors='pt')

    # Generate output (limiting the length to prevent long outputs)
    outputs = fine_tuned_model.generate(
        inputs, 
        max_length=100, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the answer after the prompt
    answer = answer.replace(prompt, "").strip()
    print("Question:", question)
    print("Answer:", answer)