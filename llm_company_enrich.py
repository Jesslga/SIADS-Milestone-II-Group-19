import requests
import json
import time
import pandas as pd
from bs4 import BeautifulSoup

# Ensure your OpenRouter API key is set
openrouter_api_key = ""

# List of free models to try in order
models_to_try = [
    "google/gemini-2.0-flash-lite-001",
    "google/gemma-3n-e4b-it:free",
    "deepseek/deepseek-prover-v2:free",
    "sarvamai/sarvam-m:free",
    "mistralai/devstral-small:free",
    "meta-llama/llama-4-maverick:free"
]

# System prompt for the summarization
system_prompt = (
    "You are an expert B2B lead generation assistant. "
    "Your job is to extract key business information from a company’s website and provide a concise overview. "
    "Summarize the company in 5–6 sentences. "
    "Always respond **only** with that summary—do **not** include any greetings, explanations, or additional content before or after."
)

headers = {
    "Authorization": f"Bearer {openrouter_api_key}",
    "Content-Type": "application/json"
}

def fetch_html(domain, timeout=10):
    print(f"[Info] Fetching HTML for domain: {domain}")
    try:
        resp = requests.get(f'https://{domain}', timeout=timeout)
        resp.raise_for_status()
        print(f"[Info] Retrieved {len(resp.text)} characters of HTML.")
        return resp.text
    except Exception as e:
        print(f"[Error] Failed to fetch HTML for {domain}: {e}")
        return None

def extract_text(html):
    print("[Info] Extracting visible text from HTML")
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    print(f"[Info] Extracted {len(text)} characters of clean text.")
    return text

def summarize_text(text):
    print(f"[Info] Summarizing text of length {len(text)}")
    for model in models_to_try:
        print(f"[Info] Trying model: {model}")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
        }
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload)
            )
            resp_json = resp.json()
            if resp.status_code == 200 and "choices" in resp_json:
                summary = resp_json["choices"][0]["message"]["content"]
                print(f"[Info] Success with model {model}")
                return summary
            else:
                error_msg = resp_json.get('error', {}).get('message', 'Unknown error')
                print(f"[Warning] Model {model} failed: {error_msg}")
                time.sleep(1)
        except Exception as e:
            print(f"[Error] Exception for model {model}: {e}")
    print("[Error] All models failed to generate a summary.")
    return None

# Example DataFrame setup (replace with your actual df)
# df = pd.DataFrame({"domain": ["example.com", "anotherexample.com"]})

# Initialize the summary column only
df["summary"] = None

# Process each domain
for idx, row in df.iterrows():
    domain = row["domain"]
    print(f"\n==== Processing row {idx}: {domain} ====")
    html_content = fetch_html(domain)

    if html_content:
        clean_text = extract_text(html_content)
        summary = summarize_text(clean_text)
        df.at[idx, "summary"] = summary
        print(f"[Result] Summary for {domain}: {summary}")
    else:
        print(f"[Skip] No HTML content for {domain}, skipping.")
        df.at[idx, "summary"] = None

# Keep only domain and summary columns in the final output
df = df[["domain", "summary"]]

print("\n[Done] Completed processing all domains.")
print(df)
