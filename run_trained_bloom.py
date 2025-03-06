from flask import Flask, render_template, request

from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("./dp_finetuned/checkpoint-33/")
model = AutoModelForCausalLM.from_pretrained("./dp_finetuned/checkpoint-33/")


# prompt = """Tell me something about"""
result_length = 150


app = Flask(__name__)

@app.route('/')
def initial():
  return render_template('index.html')


@app.route('/submit-prompt', methods=['POST'])
def generate():
  #get the prompt input
  prompt = request.form['prompt-input']
  print(f"Generating response from bloom for: {prompt}")

  inputs = tokenizer(prompt, return_tensors="pt")

  #generate text
  generated_text = ""
  
  # Sampling Top-k + Top-p
  bloom_1b7 = tokenizer.decode(model.generate(inputs["input_ids"],
                                                max_length = result_length,
                                                do_sample= True,
                                                top_k = 50,
                                                top_p = 0.9
                                                )[0])
  chars_per_line = 75
  print('Q: ', prompt)
  print('')
  print('A: ')

  for i in range(0, len(bloom_1b7), chars_per_line):
    print(bloom_1b7[i:i+chars_per_line])
    generated_text = f"{generated_text}{bloom_1b7[i:i+chars_per_line]}"
    
  return render_template('index.html', generated_text=generated_text, prompt=prompt)

if __name__ == '__main__':
    app.run()
