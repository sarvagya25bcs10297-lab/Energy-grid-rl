import os
from openai import OpenAI

def log_start():
    print("[START]")

def log_step(step_number, action):
    print(f"[STEP] Step: {step_number}, Action: {action}")

def log_end():
    print("[END]")

def main():
    # Mandatory environment variables
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    hf_token = os.getenv("HF_TOKEN")

    if not all([api_base_url, model_name, hf_token]):
        print("Error: API_BASE_URL, MODEL_NAME, and HF_TOKEN must be set in environment.")
        return

    # Use OpenAI Client for all LLM calls
    client = OpenAI(
        base_url=api_base_url,
        api_key=hf_token
    )

    log_start()

    try:
        # This script serves as the inference entry point.
        # In a real RL scenario, the LLM might decide the action or optimize the agent's policy.
        # For compliance with the prompt, we demonstrate a loop that emits the required logs.

        # Dummy inference loop to simulate environment steps
        for i in range(1, 6):
            # Call the LLM to get a decision/action
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an energy grid optimizer."},
                    {"role": "user", "content": f"Decide the energy action for step {i}."}
                ]
            )
            action = response.choices[0].message.content
            log_step(i, action)

    except Exception as e:
        print(f"Inference error: {e}")
    finally:
        log_end()

if __name__ == "__main__":
    main()
