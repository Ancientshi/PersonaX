API_KEY=your_api_key  # you need to apply for an API key from OpenAI

# recent+distill, sampling most recent 5 historical behaviors for online persona construction
python client_agent.py \
    --method recent \
    --persona_learning_type distill \
    --k 5 \
    --api_key $API_KEY

# relevance+distill, sampling top 5 relevant historical behaviors for online persona construction
python client_agent.py \
    --method relevance \
    --persona_learning_type distill \
    --k 5 \
    --api_key $API_KEY

# personax+distill, using PersonaX for offline persona construction, generating multiple persona snippets. In the online stage, choose the most relevant snippet used for the client agent do recommendation task.
python client_agent.py \
    --method personax \
    --persona_learning_type distill \
    --distance_threshold 0.7 \
    --alpha 1.06 \
    --ratio 0.6 \
    --api_key $API_KEY

#You can also use other persona learning types, in this work we provide train(reflect) and distill.
#The evaluation result will be stored in result folder. The offline persona construction result will be stored in storage folder.