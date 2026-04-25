import json
import os
from collections import defaultdict

def main():
    input_file = os.path.join(os.path.dirname(__file__), "qwen25_finetune_data.jsonl")
    
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    # Group episodes by character
    episodes_by_char = defaultdict(list)
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            char = data.get("metadata", {}).get("target_character", "unknown")
            episodes_by_char[char].append(data)
            
    print(f"Loaded {sum(len(eps) for eps in episodes_by_char.values())} episodes across {len(episodes_by_char)} characters.")
    
    # Write a readable JSON file for each character (saving only the first 3 episodes to avoid huge files)
    output_dir = os.path.join(os.path.dirname(__file__), "readable_samples")
    os.makedirs(output_dir, exist_ok=True)
    
    for char, episodes in episodes_by_char.items():
        # Let's format the JSON beautifully. We'll grab up to 3 examples.
        sample = episodes[:3]
        
        # We can also parse the assistant messages to show the actual JSON stroke dict instead of a raw string
        for ep in sample:
            for msg in ep["messages"]:
                if msg["role"] == "assistant":
                    try:
                        msg["content"] = json.loads(msg["content"])
                    except:
                        pass
                        
        out_filename = os.path.join(output_dir, f"{char}_sample.json")
        with open(out_filename, "w", encoding="utf-8") as f:
            json.dump(sample, f, indent=4)
            
    print(f"Saved readable JSON samples for each character in: {output_dir}")

if __name__ == "__main__":
    main()
