##==**Fundamentals of Large Language Models**==
###Basics of LLMs
   - `Language Model` = probablilstic model of text. Large LMs just refers to the number of parameters (large)
   - `Decoding` = Term for generating text from an LLM
   - `Prompting` = Affects the distribution of the LLMs vocabulary but it does not change the models parameters
   - `Training` = Affects the distribution and changes the models parameters. 

###LLM Architectures
   1. ==**Encoders**== they are used for `embedding`
   2. ==**Decoders**== they are used for ` text generation`
   - Capabilities can be embedding or text generation 
   - All models are built using the `Transformer` Architecture. 
   - `Embedding` = convert a sequence of word into a vector or sequence of vectors. Basically, embedding converts the text into a numerical representation of the text with meaning.
   - `Generation` = Generate text based on a sequence of input words. 

- Encoders and Decoders can come in all sizes. Sizes is defined by the number of trainable parameters it has. `Decoder models are larger than encoders` but you can make encoders big but its not needed. 

- `Encoders`: Semantic search is the primary use where you can store an input snippet into an index and then use a group of documents to find the one that is most similair 

- `Decoders`: take a sequence of tokens and generate the next word in the sequence. They only produce a single token at a time. 

- `Encoder-Decoder`: encodes a sequence of words and uses the encoded to output the next word. Used for machine translation. 

##Prompting & Prompt Engineering
- LLMs typically only involve decoder only models

==**Prompting**==

- Altering the content or structure of the input that you pass to the model: text provided to an LLM as input, sometimes containing instructions and/or examples.

- Decoders have pre-training which is where they are trained on a large set of input

==**Prompt Engineering**==
- The process of iteratively refining a prompt for the purpose of eliciting a particular style of response. 

1. *In-context learning*: prompting an LLM with instructions and or demonstrations of the task its meant to complete. 

2. *k-shot prompting*: explicitly providing k examples of the intended task in the prompt.
   - 0-shot prompting means not providing any examples

3. *Chain-of-Thought prompting*: prompt the LLM to emit intermediate reasoning steps 

4. *Least-to-Most prompting*: prompt the LLM to decompose the problem and solve, easy first then hardest. 

5. *Step-Back prompting*: prompt the LLM to identify high-level concepts pertinent to a specific task.

`Issues with Prompting`:
1. *Prompt Injection*: deliberately provide an LLM with input that attempts to cause it to ignore instructions, cause harm, or behave contrary to deployment expectations. 

2. *Memorization*: after answering, repeat the original prompt (leaked prompt) or data from a previous prompt answer

==**Training**==:
- prompting alone may be inappropriate when: training data exists or domain adaption is required because small changes in the prompt can lead to huge changes in the probablity of the next output. 

- *Domain-Adaption*: adapting a model through training to enhance its performance outside of the domain/subject area it was trained on. 

- Training is the process of giving a model an input then having the model guess the output and then based on the answer alter the parameters of the model so next time it generates something closer to the answer. 

Training Styles:
1. *Fine-Tuning*: take a pre-training model and train the model on custom data but its very expensive 
2. *Param Efficient Fine Tuning*: Isolate a small set of the models parameters to train - Cheaper (LORA)
3. *Soft Prompting*: Adding parameters to the prompt or specialized words in the prompt - learning and only uses a few parameters
4. *Pre-Training*: changes all parameters but it uses unlabeled data 

- Pre-training is more expensive then fine tuning

###Decoding

- The process of generating text with an LLM

- It happens 1 word at a time and it is iterative 

- Word is appended to the input and the new revised input is fed into the model to be decoded for the next word. 

Types of Decoding:

1. *Greedy Decoding*: pick the highest probability word at each step. 

2. *Non-Deterministic Decoding*: Pick randomly among high probability candidates at each step. **Tempature** = controls the sharpness or smoothness of this probability distribution. A low temperature value results in a sharper distribution, meaning that the model is more confident in its predictions and tends to select the most likely word with higher probability. Conversely, a higher temperature value smooths out the distribution, making it more likely for lower probability words to be chosen, leading to more diverse and varied output.
   - When temperature is decreased, the distribution is more peaked around the most likely word. = Greedy Decoding
   - When temperature is increased, the distribution is flattened over all words. 
   - But the probability in terms of the highest and lowest rated ones will always state that way. == Ordering of words is unaffected by temperature.

3. *Nucleus Sampling Decoding*: Precise what portion of the distrubtion words you can sample from

4. *Beam Search*: Multiple similar sequences simultaneously and it is not greedy but outputs sequences with higher probability than greedy decoding. 

###Hallucination
- generated text that is non-factual and or ungrounded. 

Reducing Hallucinations (no way to eliminate):
- *Grounded*: generated text is grounded in a document if the document supports the text. Attribution/Grounding 

###LLM Applications
1. ==**Retrieval Augmented Generation**== (RAG): a system where input is turned into a query that has access to support documents which will generate a correct answer and this can reduce hallucination. 
   - Non-parametric way of improving the model because you just add more documents but not the model itself

2. ==**Code Models**==: Are LLMs training on code and comments instead of written language 

3. ==**Mult-Modal Models**==: Are trained on multiple modalities like languages, images, etc. 

4. ==**Language Agents**==: Models that are intended for sequential decision making scenarios like playing chess and take actions iteratively in response to their enviornment. 
   - ReAct: Prompt the model for thoughts, then acts, and observes the results 
   - Toolformer: strings are replaced with calls to tools
   - Bootstrapped reasoningL emit rationalization of intermediate steps

---

##==**OCI Generative Services**==

###Chat Models
- *Tokens*: Language models understand tokens instead of characters and tokens can be a part of a word, an entire word or punctuation. 
- For example a sentence with 10 words can have 15 tokens 

Pretrained Chat Models

1. *command r-plus*: used for q&a, info retrieval and sentiment analysis

2. *command r-16k*: This is the smaller and more fast version of r and it used when speed and cost is important. 

3. *llama-3.1-405b/70b instruct*: largest publically available LLM

###Chat Model Parameters

1. *Maximum Output Tokens*: The max number of tokens model generates per response. 

2. *Premble Override*: Initial guideline message that can change the models overall chat behaviour and conversation style. 

3. *Temperature*: Controls the randomness of the output. To generate the same output for a prompt you use 0 (highest probability answer). Lower values are more correct and used for Q&A and higher values are more random and used for creative. 

4. *Top k*: Top K tells the model to pick the next token from the top 'k' tokens in the list sorted by probability

5. *Top p*: Pick top tokens based on the sum of their probabilities (finds the combination of p tokens that yields the highes probabililty)

6. *Frequency Penalty*: Used to get rid of repetition in your outputs. Frequency Penalty penalizes tokens that have already appeared in the preeceding text. 

7. *Presence Penalty*: Also used to get rid of repetition by applying the penalty regardless of frequency so if the token has appeared once it will be penalized. 

###Generative AI Inference API
- Within the Oracle cloud if you go to the Generative AI module you can actually copy the code that is developed from the playground and then take that python code and run it in a Jupyter notebook.

- inference API is the basically the endpoint you use within the Python script 

- To setup the config file within the Oracle cloud you go to My Profile and within your details you will see `API keys`. When you create an API key make sure you download the private key file as well as adding the config file

###==**Embedding Models**==

- Translation is a sequence to sequence task

1. *Word Embeddings*: Capture properties of the word. For example, the word is an animal so some properties could be size and age. But actual embeddings represent more properties than just two 

2. *Semantic Similarity*: Cosine and Dot product similairity can be used to compute ==**numerical similarility**==
   - embeddings that are numberically similair are also semantically similair

3. *Sentence Embeddings*: Associates every sentence with a vector of numbers. 

Embeddings use case:

- *Retrieval Augmented Generation (RAG)*: take a large document and generate the embeddings of each paragraph and put it into a vector database to allow you to get semantic search. 

Embedding Models in GenAI

1. *cohere.embed-english*
2. *cohere.ember-english-light*
3. *cohere.emberd.multilingual*
   -use cases: semantic search, text classification, and text clustering 
   - 1024-dimensional vector for eaxch embedding and max 512 tokens. The light version only uses a 384 dimensionsal vector.

- As you compress the embeddings to lower dimensions the information retained is less 

###Prompt Engineering

- LLMs are next word predictors they attempt to produce the next series of words that are most likely to follow from the previous text. 

- *Reinforcement Learning from Human Feedback (RLHF)* is used to fine-tune LLMs to follow a broad class of written instructions.

Prompt Formats:

- large language models are trained on a specific prompt format. 
- *Llama2 Prompt formatting*: They use a beginning and end [INST] tag. Instruction tags

- *Zero Shot Chain-of-Thought*: apply chain of thought but you don't provide example so you just ask it a phrase like lets think step by step as opposed to chain of thought were you provide it the examples for reasoning. 

###Customizing LLMs with Data

Training LLMs from scratch Cons:

1. *Cost*: Very expensive to train 
2. *Data*: A lot of data is needed and you need to annotated data (labelled)
3. *Expertise*: Pretraining is hard and you need to understand what model performance means

3 Options to Customize LLMs:

1. *In-context Learning/Few Shot Prompting*
   - *Chain of Thought Prompting*: Breaking a model down into smaller chunks and give reasoning.
   - Limitation of in context learning: ==**Model Context length**== (which is the number of tokens it can process)

2. *Fine-tuning a pretrained model*: optimizing a model on a smaller domain-specific dataset
   - Benefits: a) Improve the model performance on specific tasks
   - b) Improve the model efficiency - reduce the number of tokens needed for your model to perform well on your tasks. 

3. *Retrieval Augmented Generation (RAG)*: language model is able to query enterprise knowledge bases and its grounded. These do not require Fine Tuning.

- Few shot prompting its simple and no training cost but the con is it adds latency to each model. Fine Tuning is used when the LLM does not perform well on a particular task and its more efficient and better performance but the con is it requires a labelled dataset (expensive and time consuming.) RAG is used when the data changes rapidly and it accesses the latest data and grounds results but its complez to setup. 

- Look at `LLM Optimization` vs `Context Optimization`:
   - context is what the model needs to know and optimization is how the model needs to act. 
   - Always start with prompt engineering and then if its a context issue you do RAG but if its an optimization issue then you do fine tuning. 

###Fine Tuning and Inference in OCI Gen AI
- Inference is using a trained ML to make predictions or decisions based on new input data.

- *Custom Model* is on that you create by using a *pretrained model* as a base and using your own data set to fine-tune the model. 

Fine-Tuning Workflow: 
   1. Create a Dedicated AI Cluster for Fine Tuning
   2. Gather training data
   3. Kickstart fine tuning
   4. Fine-tuned (custom) Model is created

- *Model Endpoint*: is a designated point on an AI Cluster where a LLM can accept user request and send back responses 

Inference Worflow:
   1. Create Dedicated AI Cluster for Hosting
   2. Create Endpoint
   3. Serve the Model

==**T-Few Fine Tuning**==
- regular fine-tuning involves updating the weights of all layers in the model which takes longer training time and has more cost.
- T-Few only updates a fraction of the models weights. (Few-Shot Parameter Efficient Fine Tuning = PEFT) and this reduce the training time and the cost. ~0.01% of the baselines model size

Reducing Inference Costs:
- usually inference is expensive 
- Each hosting cluster can have 1 base model endpoint and N Fine-tuned custom models. So they share the same GPU resources. 
- GPU memory is limited so if you switch between models it can cause alot of overhead since you have to reload the full GPU memory. 
- `Parameter Sharing` reduces the total amount of memory and has minimal overheard. 

###Dedicated AI Clusters
- A single-tenant deployment where the GPU model is used to only host your custom models and since the endpoint is not shared the throughput is consistent

- 2 types:
   1. *Fine-Tuning*
   2. *Hositng*

Dedicated AI Clusters Sizing & Pricing:

Different Cluster Unit Types:
1. *Large Cohere Dedicated*: You can do both fine-tuning and hosting but its limited to the cohere R command family.

2. *Small Cohere Dedicated*: Same as above but a smaller count

3. *Embed Cohere Dedicated*: Used for embedding but no fine-tuning but you can still host. 

4. *Large Meta Dedicated*: Used for both finetuning and hosting but uses Meta Llama models. 

Unit Sizing:
   1. Cohere Command R+ - Does not support Fine tuning and needs 2 units of large cohere dedicated units for hosting.

   2. Cohere Command R - Supports fine tuning and needs 8 units of small cohere dedicated and for hosting it needs 1 unit of small cohere dedicated.

   3. Meta Llmama - It neeeds 4 units of large meta dedicated for fine tuning and 1 unit large meta for hosting. 

   4. Cohere English Embed - doesnt support fine tuning and needs 1 unit of embed dedicated for hosting. 

Pricing Example:
   - Cohere Command R 08-2024
   - Min hosting commitment = 744unit-hours/cluster
   - Fine tuning commitment = 1 unit-hour/fine tuning
   - So for hosting you have to pay for the month but fine tuning is on a per hour basis

###Fine-Tuning Configuration

2 Training Methods:
  - *T-Few*
  - *LoRA*: Low rank adaptation 
  - Both of them are PEFT (parameter efficient fine tuning) ie. they fine tune only a subset parameters.

Hyperparameters:
   1. *Total Training Epochs*: the number of times the model is trained using the entire dataset. 

   2. *Total Batch Size*: number of samples processed before updating the model parameters. Large batches speed up learning

   3. *Learning Rate*: How fast the model adjusts it settings. 
   
   4. *Early Stopping Threshold*: When the machine should stop training if its not improving fast enough 

   5. *Early Stopping Patience*: How long the machine waits before its not learning.

   6. *Log model metrics interval in steps*: Determines how frequently to log model metrics. 

Evaluating Fine-Tuning:

1. *Accuracy*: Measures whether the generated tokens match the annotated tokens (labelled output)

2. *Loss*: Tells you how many predictions the model got wrong. Loss decreases as the model icreases. A loss of 0 means all output was perfect. If the context is simialir then loss is low.
 -**Loss is the preferred metric because Gen AI doesn't always know what is right**

- Model training set needs to keys: the prompt and the completion.

###OCI AI Generative Security
- GPUs allocated for a customers gen AI task are isolated from other GPUs

- dedicated GPU cluster only handles your base and fine-tuned models within your set of GPUs so there is data isolation.

- Customer data is restricted within the customers tenancy so one customers data cant be seen by another customer. 

- Also uses `OCI IAM` for authentication and authorization. 

- `OCI Key Management` is used for secrets.

- `OCI Object Storage buckets` for customer fine tuned models.

Question about periods as a stop sequence: The model stops generating text once it reaches the end of the first sentence, even if the token limit is much higher.
Explanation: Stop sequences, in the context of text generation, are special tokens or symbols used to signal the end of the generated text. These sequences serve as markers for the model to halt its generation process. Common stop sequences include punctuation marks such as periods (.), question marks (?), and exclamation marks (!), as they typically denote the end of sentences in natural language.

- The main advantage of using few-shot model prompting to customize a Large Language Model (LLM) is its ability to adapt the model quickly and effectively to new tasks or domains with only a small amount of training data. Instead of retraining the entire model from scratch, which can be time-consuming and resource-intensive, few-shot prompting leverages the model's pre-existing knowledge.
---

##==**RAG using GenAI Service & Oracle 23 ai Vector Search**==

###OCI GenAI Integrations

- LangChain provides a wrapper class for using OCI GenAI with LangChain
- *LangChain*: is a framework for developing apps powered by language models. 

To create a chatbot you need:
1. LLM
2. Prompts
3. Memory
4. Chains
5. Vector Stores
6. Document Loaders 

2 Main Tyes of LangChain Models:

1. `LLMs`: Pure text completion models
2. `Chatbots`: tuned specific for having conversations. 
- the core element of a language model is the model

LangChain Prompt Templates:
- LangChain prompts can be created using 2 types of langchain prompt templates

1. *String Prompt Template*: created from a formatted python string and can have any number of variables and the output is a string. 
- Used for Generation Models

2. *Chat Prompt Template*: this type of prompt template supports a list of messages and is used for Chat Models. 

LangChain Chains:
- provides frameworks for creating chains of components including LLMs and other types. 

2 Ways to create Chains:
1. *LCEL*: this creates chains declaratively (LangChain Expression Language). This is preferred. 

2. *Legacy*: creates chains using python classes like LLM Chain

- Using Chains is how you link between getting user input to generating the response 

LangChain Memory
- ability to store information about past interactions = `memory`
- Chain interacts with the memory to pass the information along with the question. 
-Oracle 23ai can be used a vector store and LangChain can use these vector stores. 

- *Oracle AI Vector Search* offers vector utilities to automatically generate vector embeddings from unstructure data
- *Oracle Select AI* can generate sql from natural language. 

###RAG
- Mitigate bias in training data because its getting information from an external data source. 
- Can overcome model limitations by feeding in only the top results from the document instead of the full document
- handle queries without new training.

RAG Pipeline:
1. *Ingestion*: Where you have documents that get broken into chunks and then they are turned into embeddings and then indexed into a database. 
   - `Document Loaders` are reponsible for loading documents from a variety of sources
   - `Chunking` - Text Splitters take a document and split it into chunks:

   a. *Chunk Size* - Many LLMs have max input size constraints so splitting allows us to process documents that would otherwise exceed the limits. But if chunks are too small they wont be semantically useful AND if its too big then it wont be semantically specific. 

   - Embeddings capture semantic relationship
   - Embeddings of similair words are close in the multi-dimensional space
   - Vector embeddings can be generated outside Oracle 23ai db by using 3rd party embedding models
   - Vector embeddings generated inside oracle 23ai use ==**ONNX**== format

   b. *Chunk Overlap* - is the number of overlapping characters between adjacent chunks. This helps preserve context between chunks. 

   c. *Splitting Method* - split block of text based on seperators like new line. It tries to retain paragraphs and sentences. 

2. *Retrieval*: There is a query which then gets searched within the index database and then the system selects the top K most relevant results. 
   - A users natural language question is encoded as a vector and sent to **AI Vector Search** 
   - Vector search uses dot product and cosine product (only cares about angle). 
   - Less angle means more similarity
   - **Vector Indexes** Are used for larger sets of data to speed up vector simialirty searches. Uses clustering, partioning, and neighbour graphs to group simialir vectors together. 
   - AI Vector search supports HNSW and IVF

3. *Generation*: Using the top K results a response to the user is generated. 

###Conversational RAG
- RAGs and the Chatbot both need to be conversational.
- Dependence on memory which holds prior chat history to be inserted to the prompt as additional context. 

- In the LangChain framework, memory serves as a dynamic repository for retaining and managing information throughout the system's operation. It allows the framework to maintain state and context, enabling chains to access, reference, and utilize past interactions and information in their decision-making processes.

-  Large Language Models (LLMs) without Retrieval Augmented Generation (RAG) primarily rely on internal knowledge learned during pretraining on a large text corpus. These models are trained on vast amounts of text data, which enables them to learn complex patterns, structures, and relationships within language.

---

##==**Chatbot using Generative AI Agent Service**==

##Oracle Generative AI Agent Service
- Agents Fully managed service that combines LLM with an intelligent retrieval answers from a knowledge base

Architecture:
1. *Interface*: This is the starting point and this is where the user interacts with the AI agent like a web app or a chatbox

2. *Inputs fed into LLM*: Can be `Short/Long Term Memory`, `Tools`, and then `prompts` and there is a knowledge database

3. *Response Generation*

Agent Concepts:

- *Generative AI Model*: This is the LLM trained on large data 

- *Agent*: Autonomous system based off the LLM that understands and generates human like text with high answerability and groundness 

- *Knowledge Base*: Agent connexts to a knowledge base which is vector based

- *Data Source*: Data source provide connection to the data store

- *Data Ingestion*: extract data from data source and convert it to a structure format and then store in a knowledege base. 

- *Session*: A series of exchanges where the user sends queries or prompts and the agent responds with relevant information 

- *Agent Endpoint*: Specific points of access in a network that agents use to interact with other systems or services 

- *Trace*: Tracks and display the conversation history both the original and generated response 

- *Citation*: Source of information that the agent uses to respond (ie. document id, page number, etc)

- *Content Moderation*: Feature to help detect or filter out certain toxic, violent or abusive phrases from the generated responses and from prompts. 

Object Storage Guideline:
- Data sources: data for gen AI agents must be uploaded as files to an object storage bucket 
- Only one bucket can be allowed per data source
- Only pdf and txt files are supported no larger than 100mb
- PDF files can have images and charts 
- No need to format charts since they are already 2D
- You can also use reference tables
- Hyperlinks are also shown as clickable links in the chat reponses

Oracle Database Guidelines: 
- Gen AI Agents dont manage databases so they have to be created ahead of time and within the database you need the following:
1. DOCID
2. Body - is the actual content that you want the agent to search 
3. Vector

Optional:
4. CHUNKID
5. URL
6. Title
7. Page_Numbers 

- The embedding model used in the query has to be the same as the embedding model that was used for the database. 

###Chatbot using Object Store
Creating Agents process:
1. Create the knowledge Base
   - Data Storage type can be:
      a. *Object Storage*
      b. *OCI OpenSearch*
      c. *Oracle AI Vector Search*

   - lexical search is key word
   - semantic search is based on context. 
   - When restarting the job it will ignore previously read in storage and will only focus on the new content.
   - You can only delete a knowledge base if it is not being used by an agent. 
   - Data sources can be deleted at any time.

2. Create the Agent
   - Within the endpoints option is where you create the endpoint to the knowledge base. 

3. Test the agent

###Chatbot using Oracle 23ai
- Database Tools is used to create a connection to a database and then you can use this tool with the agent that you set up. 

- When you restart a previously run ingestion job, the pipeline detects files that were successfully ingested earlier and skips them. It only ingests files that failed previously and have since been updated.

- 3 is the number of endpoints you can create for each agent.

- If your data isn't available yet, create an empty folder for the data source and populate it later. This way, you can ingest data into the source after the folder is populated.

---
##Additional Notes
- OCI Generative AI Service = ~40% of the exam
- Increasing temperature flattens the distribution allowing for more varied words. 

---

##Sample Question Study Guide:

1. The OCI Generative AI Agents service retains customer-provided queries and retrieved context used for chatting with Large Language Models (LLMs) during the user's session. However, this data isn't stored beyond the session. Also, the service doesn't use customer chat data or knowledge base data for model training. = Permanently Deleted 

2. When a model is deprecated, it remains available for use temporarily. The company should plan to migrate to another model before the deprecation period ends and the model is retired.

3. The OnDemandServingMode is used to configure the Generative AI model to handle requests on-demand, which is suitable for use cases where requests are sporadic or less frequent.

4. Cosine distance (or cosine similarity) measures the angle between two vectors in a high-dimensional space. A cosine distance of 0 (which corresponds to a cosine similarity of 1) means that the vectors are identical in direction, indicating strong semantic similarity between the two embeddings. This is crucial in vector search and retrieval systems, where similar meanings are identified based on direction rather than magnitude.

5. Multi-modal parsing is used to parse and include information from charts and graphs in the documents.

6. In Retrieval-Augmented Generation (RAG), Groundedness ensures that the model s response is factually correct and traceable to retrieved sources, minimizing hallucinations. Answer Relevance, on the other hand, evaluates how well the response aligns with the user s query, ensuring that the retrieved and generated content is contextually appropriate rather than just factually correct. Both are essential for high-quality AI-generated responses.

7. The endpoint variable stores the URL where requests to Oracle's Generative AI inference service are sent. This endpoint acts as the gateway to communicate with the AI model hosted in the specified region.

8. You can only delete knowledge bases that aren't used by agents. Before you delete a knowledge base, you must delete the data sources in that knowledge base and delete the agents using that data source. The delete action permanently deletes the knowledge base. This action can't be undone.

9. If a seed value is provided, the model generates deterministic responses (same input leads to the same output). However, if no seed is specified, the model behaves non-deterministically, producing diverse responses each time it processes the same input.

10. In Retrieval-Augmented Generation (RAG), the ranker evaluates and prioritizes the retrieved information to ensure that the most relevant and high-quality data is passed to the generator. It refines the initial retrieval results by scoring and reordering them based on relevance, improving the accuracy and contextual appropriateness of the generated response. This step is crucial for minimizing irrelevant or misleading outputs.

11. Fine-tuning helps improve model efficiency by adapting a pre-trained model to a specific task or domain, allowing it to generate more relevant responses with fewer input tokens. This reduces computational costs and inference time while maintaining or improving accuracy. By refining the model's knowledge, fine-tuning enhances performance without requiring excessive amounts of new training data.

12. The seed parameter ensures that the model generates deterministic outputs. By setting a fixed seed value (e.g., 123), the model will consistently produce the same response for the same input prompt and parameters. Leaving it as None allows the model to generate varied responses each time. temperature, frequency_penalty, and top_p control aspects of the text generation process, but they do not enforce consistency across multiple runs.

13. Oracle Database typically uses port 1521 for SQL*Net (also known as Oracle Net) connections, which facilitate communication between clients and the database server. Some configurations may also use port 1522 for additional services or failover. When setting up ingress rules in an OCI subnet security list, allowing traffic over ports 1521-1522 ensures that Oracle Database can be accessed properly within the Generative AI Agents environment.

14. Retrieval-Augmented Generation (RAG) is a non-parametric approach because it retrieves relevant information from external data sources at inference time rather than relying solely on pre-trained parameters. This allows RAG to dynamically answer questions based on any corpus without requiring a separate fine-tuned model for each dataset.

15. The cohere.command-r-08-2024 model supports the T-Few and LoRA fine-tuning methods. NOT Vanilla.

16. For on-demand inferencing, the total number of billable characters is the sum of the prompt and response length.

17. In the LangChain framework, a chain typically interacts with memory at two specific points during a run. The first interaction occurs after user input but before chain execution begins. At this stage, the chain may access and retrieve relevant information or context stored in memory to inform its processing or decision-making process. The second interaction with memory occurs after the core logic of the chain has been executed but before generating the final output. At this stage, the chain may update or modify the memory based on the results of its processing, storing any relevant information or intermediate results for future reference or use.

18. Soft prompting involves learning a small set of continuous embeddings that guide the model's behavior without modifying its original parameters. Unlike traditional fine-tuning, soft prompts require no task-specific training of the full model and are efficient in adapting LLMs to different tasks with minimal computational overhead. This makes it ideal for scenarios where full fine-tuning is impractical but some level of customization is needed.

19. The totalTrainingSteps parameter is calculated as: totalTrainingSteps = (totalTrainingEpochs * size(trainingDataset)) / trainingBatchSize This formula determines the total number of training steps based on the number of epochs, the size of the training dataset, and the batch size.

20. A hosting dedicated AI cluster can have up to 50 endpoints.

21. A notification indicates that the endpoint resource is moved to the new compartment successfully. You might notice that the endpoint status changes to Updating. After the move is successful, the endpoint status changes back to Active.

---

```
# **NEW CODE: Calculate build duration**
if start_date and finish_date:
    try:
        start_dt = datetime.strptime(start_date[:15], '%Y%m%dT%H%M%S')
        finish_dt = datetime.strptime(finish_date[:15], '%Y%m%dT%H%M%S')
        duration = finish_dt - start_dt
        build_duration_minutes = duration.total_seconds() / 60
    except ValueError:
        pass
```
```
# **NEW CODE: Determine which filter matched this build**
matched_filter = self._get_matched_filter(build)
```
---
```
'build_duration_minutes': build_duration_minutes,  # **NEW**
'matched_filter': matched_filter  # **NEW**
```
---
```
def _get_matched_filter(self, build: Dict[str, Any]) -> str:
    """
    **NEW METHOD: Determine which specific filter matched this build**
    """
    build_type = build.get('buildType', {})
    project_name = build_type.get('projectName', '').lower()
    project_id = build_type.get('projectId', '')
    full_project_path = self._get_project_full_path(project_id, project_name).lower()
    
    # Check which project filter matched
    for filter_term in self.project_filters:
        filter_lower = filter_term.lower()
        
        # Check all the same matching logic as in _matches_filters
        if (filter_lower == project_name or
            filter_lower in full_project_path or
            (' > ' in filter_lower and (filter_lower == full_project_path or full_project_path.endswith(filter_lower))) or
            ('/' in filter_lower and (filter_lower.replace('/', ' > ') == full_project_path or full_project_path.endswith(filter_lower.replace('/', ' > ')))) or
            filter_lower == project_id.lower()):
            return filter_term  # Return original filter (not lowercased)
    
    return "Unknown Filter"
```
---
```
# Create sheets
self._create_executive_summary_sheet(wb, df, summary)  # **NEW: Make this first**
self._create_overview_sheet(wb, df, summary)
```
---
```
def _create_executive_summary_sheet(self, wb: Workbook, df: pd.DataFrame, summary: pd.DataFrame):
    """
    **NEW METHOD: Create executive summary sheet by project filter**
    """
    ws = wb.create_sheet("🎯 Executive Summary", 0)  # Make it the first sheet (index 0)
    
    # Title
    ws['A1'] = "Executive Summary - Build Analysis by Project Filter"
    ws['A1'].font = Font(size=18, bold=True, color="2E4057")
    ws['A2'] = f"Analysis Period: Last 30 Days ({datetime.now().strftime('%Y-%m-%d')})"
    ws['A2'].font = Font(size=12, italic=True)
    
    # Check if we have the required columns
    if 'matched_filter' not in df.columns:
        ws['A4'] = "Error: Build analysis data incomplete. Please regenerate the report."
        ws['A4'].font = Font(size=12, color="FF0000", bold=True)
        return
    
    # Group by matched filter to get summary statistics
    filter_summary = df.groupby('matched_filter').agg({
        'build_id': 'count',  # Total builds
        'is_successful': 'sum',  # Successful builds
        'is_failed': 'sum',  # Failed builds
        'build_duration_minutes': ['mean', 'median']  # Average build time
    }).round(2)
    
    # Flatten column names
    filter_summary.columns = ['total_builds', 'successful_builds', 'failed_builds', 'avg_build_time_minutes', 'median_build_time_minutes']
    
    # Calculate rates
    filter_summary['success_rate'] = (filter_summary['successful_builds'] / filter_summary['total_builds'] * 100).round(1)
    filter_summary['failure_rate'] = (filter_summary['failed_builds'] / filter_summary['total_builds'] * 100).round(1)
    
    # Reset index to make matched_filter a regular column
    filter_summary = filter_summary.reset_index()
    
    # Sort by total builds (descending)
    filter_summary = filter_summary.sort_values('total_builds', ascending=False)
    
    # Create the table
    start_row = 5
    headers = [
        "Project Filter",
        "Total Builds", 
        "Successful",
        "Failed",
        "Success Rate (%)",
        "Failure Rate (%)",
        "Avg Build Time (min)",
        "Median Build Time (min)"
    ]
    
    # Write headers
    for j, header in enumerate(headers):
        cell = ws.cell(row=start_row, column=j + 1, value=header)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color="2E4057", end_color="2E4057", fill_type="solid")
        cell.alignment = Alignment(horizontal='center', vertical='center')
        cell.border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
    
    # Write data rows
    for i, (_, row) in enumerate(filter_summary.iterrows()):
        row_data = [
            row['matched_filter'],
            row['total_builds'],
            row['successful_builds'],
            row['failed_builds'],
            f"{row['success_rate']:.1f}%",
            f"{row['failure_rate']:.1f}%",
            f"{row['avg_build_time_minutes']:.1f}" if pd.notna(row['avg_build_time_minutes']) else "N/A",
            f"{row['median_build_time_minutes']:.1f}" if pd.notna(row['median_build_time_minutes']) else "N/A"
        ]
        
        for j, value in enumerate(row_data):
            cell = ws.cell(row=start_row + 1 + i, column=j + 1, value=value)
            cell.border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
            cell.alignment = Alignment(horizontal='center', vertical='center')
            
            # Color code success rate column
            if j == 4:  # Success rate column
                success_rate = row['success_rate']
                if success_rate >= 90:
                    cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                elif success_rate >= 70:
                    cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
                else:
                    cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    
    # Add summary totals at the bottom
    total_row = start_row + len(filter_summary) + 2
    ws.cell(row=total_row, column=1, value="OVERALL TOTALS:").font = Font(bold=True, size=12)
    
    overall_total_builds = filter_summary['total_builds'].sum()
    overall_successful = filter_summary['successful_builds'].sum()
    overall_failed = filter_summary['failed_builds'].sum()
    overall_success_rate = (overall_successful / overall_total_builds * 100) if overall_total_builds > 0 else 0
    
    # Calculate weighted average build time
    total_build_time = 0
    total_builds_with_time = 0
    for _, row in filter_summary.iterrows():
        if pd.notna(row['avg_build_time_minutes']):
            total_build_time += row['avg_build_time_minutes'] * row['total_builds']
            total_builds_with_time += row['total_builds']
    
    overall_avg_build_time = total_build_time / total_builds_with_time if total_builds_with_time > 0 else 0
    
    totals_data = [
        ["Total Builds:", overall_total_builds],
        ["Total Successful:", f"{overall_successful} ({overall_success_rate:.1f}%)"],
        ["Total Failed:", f"{overall_failed} ({100-overall_success_rate:.1f}%)"],
        ["Overall Avg Build Time:", f"{overall_avg_build_time:.1f} minutes" if overall_avg_build_time > 0 else "N/A"]
    ]
    
    for i, (label, value) in enumerate(totals_data):
        ws.cell(row=total_row + 1 + i, column=1, value=label).font = Font(bold=True)
        ws.cell(row=total_row + 1 + i, column=2, value=value)
    
    # Auto-adjust column widths
    for column in ws.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 50)
        ws.column_dimensions[column_letter].width = adjusted_width
```