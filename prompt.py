import os
import re


Decoupling_Prompt='''
### Task
You task is to decouple [User Profile] into positive part and negative part. The [User Profile] is a collection of user's preferences, dislikes, and other relevant information. You need to extract the positive part and negative part from the [User Profile]. The positive part is the user's preferences, and the negative part is the user's dislikes.

### User Profile
{profile}

### Steps to Follow
1. Extract the user's preferences and dislikes from the [User Profile].
2. The positive part is the user's preferences, and the negative part is the user's dislikes.
3. Generate the positive part and negative part separately.

### Important Notes
1. Your output should strictly be in the following format:
    Positive Part: {{User's preferences}}
    Negative Part: {{User's dislikes}}
2. When identifying user's likes and dislikes, do not fabricate them! If your [User Profile] doesn’t specify any relevant preferences or dislikes.

### Response Example
Positive Part: I like ...
Negative Part: I dislike ...
'''

Inference_Prompt='''
### Task
We provide a user's personal profile in [User Profile], which includes the user's preferences, dislikes, and other relevant information. You need play the role of the user. And we also provide two candidate items, A and B, with their features in [Item Feature].  You need to choice between the two item candidates based on your profile and the features of the items. Furthermore, you must articulate why you’ve chosen that particular item while rejecting the other.

### User Profile
{profile}

### Item Feature
Item A: {item_a}
Item B: {item_b}

### Steps to Follow
1. Extract your preferences and dislikes from your self-introduction.
2. Evaluate the two candidate in light of your preferences and dislikes. Make your choice by considering the correlation between your preferences/dislikes and the features of the candidates.
3. Explain why you made such choices, from the perspective of the relationship between your preferences/dislikes and the features of these candidate items.

### Important Notes
1. Your output should strictly be in the following format:
    Chosen Item: {{Item A or Item B}}
    Explanation: {{Your detailed rationale behind your choice and reasons for rejecting the other item.}}

2. When identifying user's likes and dislikes, do not fabricate them! If your [User Profile] doesn’t specify any relevant preferences or dislikes, use common knowledge to inform your decision.

3. You **must** choose one of these two candidates, and **cannot** choose both.

4. Your explanation needs to be comprehensive and specific. Your reasoning should delve into the finer attributes of the items.

5. Base your explanation on facts. For instance, if your self-introduction doesn’t reveal any specific preferences or dislikes, justify your decision using available or common knowledge.

6. Please ignore the effect of Item position and length, they do not affect your decision.

### Response Example
Chosen Item: Item A
Explanation: I chose Item A because...
'''

Validate_Prompt='''
### Task
We provide a user's personal profile in [User Profile], which includes the user's preferences, dislikes, and other relevant information. You need play the role of the user. And we also provide candidate item A's features in [Item Feature]. You need to judge whether you will like the item or not based on your profile and the features of the item. Furthermore, you must articulate why you’ve chosen that particular item while rejecting the other.

### User Profile
{profile}

### Item Feature
Item: {item}

### Steps to Follow
1. Extract your preferences and dislikes from your self-introduction.
2. Evaluate the item in light of your preferences and dislikes. Make your choice by considering the correlation between your preferences/dislikes and the features of the candidate item.
3. Explain why you made such choices, from the perspective of the relationship between your preferences/dislikes and the features of the candidate item.

### Important Notes
1. Your output should strictly be in the following format:
    Decision: {{Like or Dislike}}
    Explanation: {{Your detailed rationale behind your decision.}}

2. When identifying user's likes and dislikes, do not fabricate them! If your [User Profile] doesn’t specify any relevant preferences or dislikes, use common knowledge to inform your decision.

3. You **must** choose one of 'Like' and 'Dislike', and **cannot** give other responses.

4. Your explanation needs to be comprehensive and specific. Your reasoning should delve into the finer attributes of the items.

5. Base your explanation on facts. For instance, if your self-introduction doesn’t reveal any specific preferences or dislikes, justify your decision using available or common knowledge.

### Response Example
Decision: Like
Explanation: I like this item because...
'''


Reflect_Prompt='''
### Background
We provide a user's personal profile in [User Profile], which includes the user's preferences, dislikes, and other relevant information. You need play the role of the user. Recently, you considered choosing one more prefered Item from two candidates. The features of these two candidate are provided in [Item Feature]. And your choice and explanation is in [Choice and Explanation], which reveals your previous judgment for these two candidates.

### User Profile
{profile}

### Item Feature
Item A: {item_a}
Item B: {item_b}

### Choice and Explanation
{response}

### Task
However, The user in the real world actually prefer to choose Item B, and reject the Item A that you initially chose. This indicates that you made an incorrect choice, the [Choice and Explanation] was mistaken. Therefore, you need to reflect and update [User Profile]. 

### Steps to Follow
1. Analyze the misconceptions in your previous [Choice and Explanation] about your preferences and dislikes, as recorded in your explanation, and correct these mistakes.  
2. Explore your new preferences based on the Item B you really enjoy, and determine your dislikes based on the Item a you truly don’t enjoy.  
3. Summarize your past preferences and dislikes from your previous [User Profile]. Combine your newfound preferences and dislikes with your past ones. Filter and remove any conflicting or repetitive parts in your past [User Profile] that contradict your current preferences and dislikes.  
4. Generate a update profile use the following format: 
My updated profile: {{Please write your updated profile here}}

### Important Notes
1. Keep your updated profile under 180 words.  
3. Any overall assessments or summarization in your profile are forbidden.  
4. Your updated profile should only describe the features of items you prefer or dislike, without mentioning your wrong choice or your thinking process in updating your profile.  
5. Your profile should be specific and personalized. Any preferences and dislikes that cannot distinguish you from others are not worth recording.

### Response Example
My updated profile: I ...
'''


Distillation_Prompt='''
### Task
We provide a user's personal profile in [User Profile], which includes the user's preferences and other relevant information. Additionally, we provide a sequence of liked items in [Sequence Item Profile] that the user has interacted with. Your task is to analyze these items in the context of the user's existing profile and produce an updated profile that reflects any new preferences, or insights inferred from the user's interactions with these items.

### User Profile
{profile}

### Sequence Item Profile
{sequence_item_profile}

### Steps to Follow
1. Carefully review the user's existing profile to understand their stated preferences and dislikes.
2. Analyze the features of the items in the provided sequence, noting any common themes, attributes, or patterns.
3. Identify any new preferences that can be inferred from the user's interactions with these items.
4. Summarize and update the user's profile by incorporating the new insights, adding new preferences or dislikes, and highlighting any changes or developments in the user's tastes.
Important Notes
5. Your output should strictly be in the following format:
Summarization: {{Your updated profile.}}
6. Do not contradict the user's existing preferences unless there is clear evidence from the sequence items that their tastes have changed.
7. Base your summary on facts and logical inferences drawn from the items in the sequence.
8. Be comprehensive and specific in your summarization, focusing on the finer attributes and features of the items that relate to the user's preferences.
9. Avoid fabricating any information not supported by the user's profile or the sequence items.

### Response Example
Summarization: You've developed a new interest in ....
'''



Rating_Prompt = '''
### Task
We provide a user's personal profile in [User Profile], which includes the user's preferences and other relevant information. Additionally, we provide a candidate item's profile  in [Candidate Item Profile]. Your task is to analyze the item's profile in the context of the user's existing profile and produce an predicted score that reflects the user's preference towards the item, the score must be at the integer range from 1~5, 1 means not interested and 5 means very interested.

### User Profile
{profile}

### Candidate Item Profile
{item_profile}

### Steps to Follow
1. Carefully review the user's existing profile to understand their stated preferences and dislikes.
2. Analyze the provided profile of the candidate item, noting any common themes, attributes, or patterns.
3. Identify any new preferences that can be inferred from the user's profile.
4. Give a predicted integer score that ranges from 1 to 5, 1 means the user will not be interested in the item, 5 means the user will be very intersted in the item.
Important Notes
5. Your output should strictly be in the following format:
Score: The predicted score is {{Your predicted score}}
6. Do not contradict the user's existing preferences unless there is clear evidence from the sequence items that their tastes have changed.
7. Avoid fabricating any information not supported by the user's profile or the sequence items.

### Response Example
Score: The predicted score is ....
'''