# InstructGPT
Project: Adapting InstructGPT to korean dataset

Purpose: 

## **Dataset**

- comparison dataset은 chosen, rejected, prompt 3가지 features로 구성되있으며 92.5k (약 92,500개의 예제)/ 83.6k train/test로 나누어져있다.
- 커뮤니티 사이트 Reddit post와 각 post에 대한 2개의 summary로 구성되있다. Human labeler는 2개의 summary 중 더 선호하는 쪽을 선택해서 'chosen'으로, 나머지 하나를 'rejected'로 라벨링한다.  


- dataset =load_dataset("CarperAI/openai_summarize_comparisons", split="train")
- Dataset({
    features: ['prompt', 'chosen', 'rejected'],
    num_rows: 92534
})

- Ko:

출처: https://huggingface.co/datasets/CarperAI/openai_summarize_comparisons
