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

2. *Fine-tuning a pretrained model*: optimizinf a model on a smaller domain-specific dataset
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

