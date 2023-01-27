import re
import random
import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.util import ngrams
import MeCab
wakati = MeCab.Tagger("-Owakati")
chencherry = SmoothingFunction()

def add_prompt(question: str, lang: str, N: int=3) -> str:
    '''Given a question in English or Japanese, create a prompt with N similar questions and their answers, drawn from a text file.

    Note:
        ・The model's answer is the content of the model's output 「「」」(Japanese) or [ ] (English)
        ・Thus, we end the prompt with "「「" (Japanese) or "[" (English)'''

    def subprompt(example, lang: str) -> str:
        #input can be either a Q&A pair from the source file, or the current question to answer
        
        if type(example) == str:
            question = example
            answer = False
        elif type(example) == dict: #case of a Q&A pair
            question = example['question']
            answer = example['answers']

        if lang == "en":
            #inclusion of an "extra" question mark empirically improved performance
            q_string = f'''Question: {question}?'''
            a_string = f''' Answer: [{answer[0]}]''' if answer else ' Answer: ['
            subprompt = q_string + a_string
            
        elif lang == "ja":
            q_string = f'''質問： {question}?'''
            a_string = f'''　答え：「「{answer[0]}」」''' if answer else '　答え：「「'
            subprompt = q_string + a_string
            
        else:
            assert 0, lang
        return subprompt
    
    def get_similar(question: str, lang: str, N: int) -> list:
        #obtains the N questions from the source file which most resemble the current question
        similar_questions = []
        with open(source_file, 'r') as source:
            source_examples = [json.loads(line.rstrip()) for line in source]
            #obtain random sentences from the source file, assuming that checking all takes too long
            sample = random.sample(source_examples, 5) #can be much larger, was set to 2000 in the actual task

            #check how similar the sample candidates are to the input question, using sentence bleu
            if lang == 'en':
                bleu_scores = [sentence_bleu([question.split()], example['question'].split(), 
                                             smoothing_function=chencherry.method1) for example in sample]
            elif lang == 'ja':
                bleu_scores = [sentence_bleu([wakati.parse(question).split()], 
                                             wakati.parse(example['question']).split(), 
                                             smoothing_function=chencherry.method1) for example in sample]

            #find the best matches and return them to be added to the prompt
            similar_indices = np.argpartition(bleu_scores, -N)[-N:]
            for index in similar_indices:
                similar_questions.append(sample[index])
            return similar_questions
    
    #main part of this function
    #designate source file and start the prompt:
    if lang == "en":
        source_file = "data/qa_examples_en.jsonl"
        prompt = 'Answer these questions:\n'
    
    elif lang == "ja":
        source_file = "data/qa_examples_ja.jsonl"
        prompt = '質問を答えてください：\n'
        
    else:
        assert 0, lang
    
    #call helper functions to get examples and add to the prompt:
    with open(source_file, 'r') as source:
        #get example "question" and "answers" put them into Q&A format:
        similar_questions = get_similar(question, lang, N)
        similar_prompts = [subprompt(example, lang) for example in similar_questions]
        #construct the prompt
        prompt += '\n'.join(similar_prompts)
        #add the actual test question to the end
        prompt = '\n'.join([prompt, subprompt(question, lang)])
        
    return prompt

def extract_answer(output: str, lang: str, N:int=3) -> str:
    '''Extract the answer from the model's generation using regex.
    
    This is called elsewhere in the program and is passed the same N variable as the get_similar function.
    N is necessary for Japanese since the Japanese model was very prone to over-generation.'''
    if lang == "en":
        return re.findall("\[(.*?)\]", output)[-1] # capture []
    elif lang == "ja":
        return re.findall("「「(.*?)」」", output)[N] # capture 「「」」
    else:
        assert 0, lang