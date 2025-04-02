

# PersonaX
PersonaX provides two interfaces (`Offline_Profiling(UGC: list) -> None`, `Online_Retrieve(Item) -> Relevant Persona Snippet`), through which the `Recommendation Agent` can ultlize `PersonaX`.

```
 -----------------                                     ----------------------
|                 |<---- Offline_Profiling (UGC) -----|                      |
|     PersonaX    |                                   | Recommendation Agent |
|                 |<---- Online_Retrieve (Item) ------|                      |
|_________________|----> Relevant Persona Snippet --->|______________________|
                                                                  |
                                                                  v
                                                        Recommendation Result
```


## Configuration Instructions
1. Clone the [EasyRec repository](https://github.com/HKUDS/EasyRec) to your local machine.
2. Move the `app_easyrec.py` file into the EasyRec directory. Set the `cache_dir` variable to specify the installation path of the model. Then, start the Flask service in `app_easyrec.py`.
3. Start the Flask service in `app.py`.
4. Configure the `OPENAI_KEY` in the `CBS_relevance.sh` script.

## Running PersonaX and Baseline Methods
- The `CBS_relevance.sh` script provides an example of running the Summarization-based LLM-UM method on the `Books_480` dataset.  
- The `recent.sh` and `relevance.sh` scripts demonstrate sampling methods using the `recent` and `relevance` approaches, respectively, while still implementing the Summarization-based LLM-UM method on the `Books_480` dataset.

## Viewing Results
The transparent persona learning process, along with runtime and evaluation results, will be saved in the `result` folder.

## Additional Information
This code also supports large models provided by the Silicon Flow platform, including ChatGLM, DeepSeek, and Qwen.
