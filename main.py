from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
previous_responses = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({'response': ''})

    response = generate_response(user_message)
    return jsonify({'response': response})

def generate_response(message, max_length=100, num_return_sequences=5, temperature=0.7):
    input_ids = tokenizer.encode(message, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        do_sample=True,
        temperature=temperature,
    )
    responses = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]

    if message not in previous_responses:
        previous_responses[message] = set()

    unique_responses = [response for response in responses if response not in previous_responses[message]]

    if not unique_responses:
        return "I don't have any more unique answers for that question."

    chosen_response = unique_responses[0]
    previous_responses[message].add(chosen_response)

    return chosen_response

if __name__ == '__main__':
    app.run(debug=True)
