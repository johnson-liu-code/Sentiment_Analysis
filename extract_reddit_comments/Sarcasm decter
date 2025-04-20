import json
from transformers import pipeline

# Load the sarcasm detection model
sarcasm_detector = pipeline("text-classification", model="mrm8488/t5-base-finetuned-sarcasm-twitter")

COMMENTS_FILE = "comments.json"
TEXT_FILE = "sarcasmfile.txt"

# Load existing comments from JSON
def load_comments():
    try:
        with open(COMMENTS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Save updated comments to JSON
def save_comments(comments):
    with open(COMMENTS_FILE, "w") as f:
        json.dump(comments, f, indent=4)

# Add a single comment (manual input)
def add_comment(text):
    comments = load_comments()
    comments.append({"text": text, "sarcastic": None, "confidence": None})
    save_comments(comments)
    print("📝 Comment added.")

# Load comments from a .txt file
def load_comments_from_txt(file_path):
    comments = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):  # skip empty or comment lines
                continue
            if ":" in line:
                _, comment = line.split(":", 1)
                comments.append({"text": comment.strip(), "sarcastic": None, "confidence": None})
    return comments

# Analyze a list of comments
def analyze_comments(comments):
    for comment in comments:
        if comment["sarcastic"] is None:
            result = sarcasm_detector(comment["text"])[0]
            comment["sarcastic"] = result["label"].lower() == "sarcasm"
            comment["confidence"] = round(result["score"], 3)
            print(f"🎭 '{comment['text']}' → Sarcastic: {comment['sarcastic']} (Confidence: {comment['confidence']})")
    return comments

# CLI menu
if __name__ == "__main__":
    print("💬 Sarcasm Detector")
    while True:
        print("\nOptions:\n1. Add single comment\n2. Analyze all stored (JSON) comments\n3. Show all stored comments\n4. Analyze from .txt file\n5. Exit")
        choice = input("Select an option: ").strip()

        if choice == "1":
            text = input("Enter your comment: ")
            add_comment(text)

        elif choice == "2":
            comments = load_comments()
            comments = analyze_comments(comments)
            save_comments(comments)

        elif choice == "3":
            comments = load_comments()
            for c in comments:
                print(f"- {c['text']} | Sarcastic: {c['sarcastic']} | Confidence: {c['confidence']}")

        elif choice == "4":
            print(f"📂 Reading from {TEXT_FILE}...")
            new_comments = load_comments_from_txt(TEXT_FILE)
            analyzed = analyze_comments(new_comments)
            save_comments(load_comments() + analyzed)  # merge with existing
            print("✅ Done analyzing and saving new comments!")

        elif choice == "5":
            print("👋 Byeee!")
            break

        else:
            print("⚠️ Invalid option, try again.")
