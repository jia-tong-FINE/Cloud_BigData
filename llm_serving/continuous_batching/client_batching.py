import requests
import sys
import os
import json

def run_client(input_text):
    url = 'http://localhost:8001/llmserver'
    Headers = {'Accept': 'text/event-stream'}
    data = {'input_text': input_text}
    response = requests.post(url, json=data, stream=True, headers=Headers)
    generate_results = [] 
    for i in range(len(input_text)):
        generate_results.append([f'<Generate {i}>'])
    for res in response.iter_lines():
        res = res.decode()
        if res == '[DONE]':
            break

        res = json.loads(res)
        os.system('clear')
        text, id = res['text'], res['id']
        text = text.replace('\n', '\\n')
        generate_results[id].append(text)
        for result in generate_results:
            print(' '.join(result))
        # print(res.decode(), end=' ')
        # sys.stdout.flush()      

    # if response.status_code == 200:  # 请求成功
    #     result = response.iter_content()
    #     print(result)
    # else:
    #     print('Error:', response.status_code)

if __name__ == '__main__':
    input_text = []
    # for _ in range(10):
    #     input_text.append("The best way to deploy a server on cloud is")
    # input_text = "The best way to deploy a server on cloud is"
    input_text =[
        "Describe the impact of",
        "Explain the concept of",
        "Discuss the advantages and disadvantages of",
        "Analyze the relationship between",
        "Illustrate the process of",
        "Compare and contrast the similarities and differences between",
        "Explore the role of",
        "Evaluate the significance of",
        "Propose a solution for",
        "Predict the potential consequences of",
        "Examine the implications of",
        "Provide insights into",
        "Elaborate on the significance of",
        "Investigate the relationship between",
        "Illustrate the key components of",
        "Outline the main features of",
        "Explore the challenges associated with",
        "Analyze the factors influencing",
        "Discuss the role of",
        "Offer a critical assessment of"
    ]
    run_client(input_text)
