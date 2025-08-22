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

==**Prompting**=:
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

##==**OCI Generative Services**==

###Chat Models

