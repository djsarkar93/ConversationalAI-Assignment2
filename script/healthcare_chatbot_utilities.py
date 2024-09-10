########################################################################################################################
# Imports
########################################################################################################################
import os
import faiss
import pickle
import torch
import requests
import re

import numpy as np

from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel



########################################################################################################################
# Constants
########################################################################################################################
ASSIGNMENT_DIR = os.path.join(os.getcwd(), 'ConversationalAI-Assignment2-Group6-ProblemStatementA')

BIOBERT_MODEL_NAME = 'dmis-lab/biobert-base-cased-v1.1'
INDEX_DIR = f'{ASSIGNMENT_DIR}/index'
INDEX_FILE_PATH = f'{ASSIGNMENT_DIR}/index/healthcare_chatbot_faiss.index'
CONTEXT_FILE_PATH = f'{ASSIGNMENT_DIR}/index/healthcare_chatbot_context.pkl'

FLAN_T5_MODEL_NAME = 'google/flan-t5-base'
MODEL_DIR = f'{ASSIGNMENT_DIR}/model'

UMLS_BASE_URL = 'https://uts-ws.nlm.nih.gov'
UMLS_API_KEY = '1d42fa9c-6884-41f0-a17b-093967453139'



########################################################################################################################
# Global Variables
########################################################################################################################
# Check if CUDA (GPU) is available; if not, use the CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the tokenizer for the BioBERT model
biobert_tokenizer = AutoTokenizer.from_pretrained(
    BIOBERT_MODEL_NAME,
    clean_up_tokenization_spaces = True
)

# Load the pre-trained BioBERT model
biobert_model = AutoModel.from_pretrained(BIOBERT_MODEL_NAME)

# Move the BioBERT model to the selected device (GPU or CPU) for computation
biobert_model.to(device)

# Load the FAISS index
medical_info_index = faiss.read_index(INDEX_FILE_PATH)

# Load the context file
with open(CONTEXT_FILE_PATH, 'rb') as f:
    medical_info_context_chunks = pickle.load(f)

# Initialize the tokenizer for the FLAN T5 model
flan_t5_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Load the base FLAN-T5 model
flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained(FLAN_T5_MODEL_NAME)

# Load the LoRA adapter for the model
#flan_t5_model = PeftModel.from_pretrained(flan_t5_model, MODEL_DIR)

# Move the pre-trained FLAN T5 model to the selected device (GPU or CPU) for computation
flan_t5_model.to(device)

# Create a transformers pipeline for text-to-text generation
t2tg_pipeline = pipeline(
    'text2text-generation',
    model       = flan_t5_model,
    tokenizer   = flan_t5_tokenizer,
    device      = device,
    max_length  = 128,
    temperature = 0.01,
    do_sample   = True
)



########################################################################################################################
# Functions
########################################################################################################################
# Function to generate embeddings for a user query
def generate_user_query_embedding(query):
    # Tokenize the query using the BioBERT tokenizer
    inputs = biobert_tokenizer(
        query,
        return_tensors='pt',
        max_length=512,
        truncation=True,
        padding=True
    )

    # Move inputs to the selected device (GPU or CPU)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate embeddings without gradient computation for efficiency
    with torch.no_grad():
        outputs = biobert_model(**inputs)
    
    # Compute the mean of the last hidden layer's output (average pooling)
    query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return query_embedding


# Function to retrieve the most relevant context from the local medical information database
def retrieve_context_from_local_db(query_embedding, top_k=3, threshold=0.5):
    # Normalize the query embedding to unit length for cosine similarity
    query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)
    
    # Convert the query embedding to the correct format for FAISS (a batch of 1 query)
    query_embedding_normalized = np.expand_dims(
        query_embedding_normalized.astype('float32'),
        axis=0
    )

    # Perform the search on the FAISS index to find the top-k closest embeddings
    similarity_scores, top_k_indices = medical_info_index.search(
        query_embedding_normalized,
        top_k
    )

    # Filter the contexts based on the similarity threshold
    relevant_contexts = [
        {
            'context': medical_info_context_chunks[idx],
            'score': score
        }
        for idx, score in zip(top_k_indices[0], similarity_scores[0])
        if score >= threshold
    ]
    return relevant_contexts


# Function to search UMLS BioInformatics web Ontology for definitions
def search_umls(search_terms, top_k, api_key):
  try:
    # Define the UMLS search API URL and parameters
    search_url = f'{UMLS_BASE_URL}/rest/search/current'
    search_params = {
        'apiKey': api_key,
        'string': ' '.join(search_terms),
        'pageSize': top_k
    }

    # Request UMLS search API; Raise an error if unsuccessful
    search_response = requests.get(search_url, params=search_params)
    search_response.raise_for_status()

    # If no results are found, return an empty string
    search_results = search_response.json()['result']['results']
    if not search_results:
      return ''
    
    # Extract the content URIs from the search result
    search_result_content_uris = [r['uri'] for r in search_results]
    
    # Initialize a variable to store the combined search content
    search_content = ''
    
    # Iterate over each content URI
    for uri in search_result_content_uris:
      # Define the UMLS definition URI
      search_content_uri = f'{uri}/definitions'
      
      # Request UMLS definition API; Raise an error if unsuccessful
      search_content_response = requests.get(search_content_uri, params={'apiKey': api_key})
      search_content_response.raise_for_status()
      
      # Extract the definitions from the response; Concatenate each definition into search content
      search_content_result = search_content_response.json()['result']
      for r in search_content_result:
        search_content += r['value']
    
    # Remove any URLs and HTML tags from the search content
    search_content = re.sub(r'http\S+', '', search_content)
    search_content = re.sub(r'<.*?>', '', search_content)
    return search_content

  except Exception as e:
    return ''


# Function to generate an answer for a user query using a language model
def generate_answer(user_query, top_k, threshold):
    # Given a user prompt, generate an answer directly using the language model
    prompt_template = (
        f"You are a friendly AI. "
        f"The following is a conversation between a human and you. "
        f"you are talkative and you provide lots of specific details from your context. "
        f"If you do not know the answer to a question, you must says \"Sorry, I don't know the answer.\".\n\n"
        f"human: {user_query}\n\n"
        f"you:"
    )
    direct_answer = t2tg_pipeline(prompt_template)
    direct_answer = (direct_answer[0]).get('generated_text', "Sorry, I don't know the answer.")
    #print(f'SLM Direct Answer: {direct_answer}')
    
    # Given a user prompt, attempt to fetch (local/external) context for the prompt
    # and then generate an answer using the language model
    # Get user query embeddings
    user_query_embedding = generate_user_query_embedding(user_query)
    
    # Get the top k user query contexts from the local database
    local_contexts = retrieve_context_from_local_db(
        user_query_embedding,
        top_k     = top_k,
        threshold = threshold
    )
    
    # Concat all context strings to a single context string
    local_context = '\n\n'.join([_['context'] for _ in local_contexts])
    #print(f'Local Context: {local_context}')
    
    # If some local context is found, use that. Otherwise, attempt to get external context
    if local_contexts:
        user_query_context = local_contexts
    
    else:
        # Identify the most important terms in the user query using using the language model
        prompt_template = (
            f"Extract the most important term from the following text: "
            f"{user_query}"
        )
        imp_term = t2tg_pipeline(prompt_template)
        imp_term = (imp_term[0]).get('generated_text', '')
        #print(f'SLM Important Term: {imp_term}')
        
        # If an important term is found, search UMLs BioInformatics Web Ontology for the term.
        if imp_term:
            external_context = search_umls(
                search_terms = [imp_term.lower()],
                top_k        = top_k,
                api_key      = UMLS_API_KEY
            )
            #print(f'External Context: {external_context}')
            user_query_context = external_context if external_context else ''
        
        else:
            user_query_context = ''
        
        # Using the context generate answer using the language model
        default_answer = "Sorry, I don't know the answer."
        if user_query_context:
            # Trim the context to the appropriate number of tokens
            user_query_context_trimmed = ' '.join(user_query_context.split()[:151])
            user_query_context_trimmed = imp_term + ' is ' + user_query_context_trimmed
            #print(f'Trimmed Context: {user_query_context_trimmed}')
            
            prompt_template = (
                f"Using the information provided in the context, write a complete sentence that answers the following question. "
                f"Ensure that the answer is paraphrased in your own words. "
                f"If the context does not contain enough information to answer, write 'Sorry, I don't know the answer.'\n\n"
                f"Context: '{user_query_context_trimmed.lower()}'.\n\n"
                f"Question: '{user_query.lower()}'."
            )
            #print(prompt_template)
            
            context_answer = t2tg_pipeline(prompt_template)[0]
            context_answer =  context_answer.get('generated_text', default_answer)
            #print(f'SLM Context Answer: {context_answer}')
        
        else:
            context_answer = ''
    
    # Using the language model, determine the best answer amongst the direct & context answers
    comparison_prompt = (
        f"Given the question: '{user_query}', "
        f"select the answer that best fits a friendly and conversational context. "
        f"Which answer is more appropriate: "
        f"'Answer A: {direct_answer}' "
        f"or "
        f"'Answer B: {context_answer}'? "
        f"Choose either 'Answer A' or 'Answer B'."
    )
    #print(comparison_prompt)
    
    best_answer = t2tg_pipeline(comparison_prompt)[0].get('generated_text')
    #print(f'Best Answer: {best_answer}')
    
    best_answer = direct_answer if 'answer a' in best_answer.lower() else context_answer
    return best_answer



########################################################################################################################
# Main
########################################################################################################################
if __name__ == '__main__':
  user_queries = [
      'Hi',
      'How are you?',
      'What is Covid?',
      'What are the symptoms of Covid?',
      'Bye'
  ]

  for query in user_queries:
      print(f'You : {query}')
      answer = generate_answer(user_query=query, top_k=2, threshold=0.9)
      print(f'AI  : {answer}')
      print('-'*153)
