from flask import Flask, request, jsonify, stream_with_context, Response
from transformers import LlamaConfig,LlamaForCausalLM,LlamaTokenizer, TextIteratorStreamer
from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model
import torch
from threading import Thread
import flask_cors
from flask_sse import sse
import time
import json
import copy
import asyncio


app = Flask(__name__)
flask_cors.CORS(app)

model_path = '../Llama-2-7b-hf'
memory_bound = {0: '1GiB', 1: '1GiB', 2: '10GiB', 3: '10GiB'}

tokenizer = LlamaTokenizer.from_pretrained(model_path)
config = LlamaConfig.from_pretrained(model_path)

with init_empty_weights():
    model = LlamaForCausalLM._from_config(config, torch_dtype=torch.float16)

device_map = infer_auto_device_map(model, max_memory=memory_bound, no_split_module_classes=LlamaForCausalLM._no_split_modules)
load_checkpoint_in_model(model, model_path, device_map=device_map)
model = dispatch_model(model,device_map=device_map)

batch_size = 4


@app.route('/llmserver', methods=['POST'])
def infer():
    data = request.json
    input_text = data.get('input_text', 'hello i am fsassdsa')        
    return Response(generate(input_text), mimetype='text/event-stream')

def generate(input_text, max_new_tokens=20, temperature=1.0, top_k=0, top_p=0.9):
    streamers = []
    all_done = []
    wating_queue = []
    batch_ind = []

    for i in range(len(input_text)):
        input_ids = tokenizer(input_text[i], return_tensors="pt").to(model.device)
        if len(streamers) < batch_size:
            batch_ind.append(i)
            streamer = TextIteratorStreamer(tokenizer)
            streamers.append(streamer)
            all_done.append(False)
            generation_kwargs = dict(**input_ids, streamer=streamer, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
        else:
            wating_queue.append((input_ids, i))
    
    
    while not all(all_done):
        for i, streamer in enumerate(streamers):
            if all_done[i]:
                continue
            all_done[i] = True
            for new_text in streamer:
                if new_text:
                    all_done[i] = False
                    res = {"text": new_text, "id": batch_ind[i]}
                    yield json.dumps(res) + '\n'
                    break
            if all_done[i]:
                if len(wating_queue) > 0:
                    batch_ind[i] = wating_queue[0][1]
                    streamer = TextIteratorStreamer(tokenizer)
                    streamers[i] = streamer
                    all_done[i] = False
                    generation_kwargs = dict(**wating_queue[0][0], streamer=streamer, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
                    wating_queue = wating_queue[1:]
                    thread = Thread(target=model.generate, kwargs=generation_kwargs)
                    thread.start()
            
    yield '[DONE]\n'
        


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)
